from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from functorch import vmap
from torch import nn
from torch.utils.data import DataLoader

from ...datasets.dataset import Dataset
from ...datasets.variables import Variables
from ..imodel import (
    IModelForInterventions,
)
from ..torch_model import TorchModel
from ...preprocessing.data_processor import DataProcessor
from ...utils.causality_utils import (
    intervene_graph,
    intervention_to_tensor,
)
from ...utils.fast_data_loader import FastTensorDataLoader
from ...utils.helper_functions import to_tensors
from ...utils.nri_utils import edge_prediction_metrics_multisample, mmd_true_posterior, nll
from .base_distributions import BinaryLikelihood, CategoricalLikelihood, DiagonalFLowBase, GaussianBase
from .generation_functions import ContractiveInvertibleGNN, combine_state_for_ensemble_icgnn


class BayesDAG(
    TorchModel,
    IModelForInterventions,
):
    """
    Base class for all BayesDAG models.
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        lambda_sparse: float = 1.0,
        scale_noise: float = 1.0,
        scale_noise_p: float = 1.0,
        base_distribution_type: str = "spline",
        spline_bins: int = 8,
        norm_layers: bool = False,
        res_connection: bool = False,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        num_chains: Optional[int] = None,
        dense_init: bool = False,
        embedding_size: Optional[int] = None,
        model_type: str = "nonlinear",
    ):
        """
        Args:
            model_id: Unique identifier for the model.
            variables: Variables object containing information about the variables.
            save_dir: Directory where the model will be saved.
            device: Device to use for training and evaluation.
            lambda_sparse: Sparsity penalty coefficient.
            scale_noise: Scale of the Gaussian noise for SG-MCMC preconditioning for theta.
            scale_noise_p: Scale of the Gaussian noise for SG-MCMC preconditioning for p.
            base_distribution_type: Type of base distribution to use for the additive noise SEM.
            spline_bins: Number of bins to use for the spline flow.
            norm_layers: Whether to use normalization layers in the encoder and decoder.
            res_connection: Whether to use residual connections in the encoder and decoder.
            encoder_layer_sizes: List of layer sizes for the encoder.
            decoder_layer_sizes: List of layer sizes for the decoder.
            num_chains: Number of SG-MCMC chains to use.
            dense_init: Whether to use dense initialization for the invertible GNN.
            embedding_size: Size of the embedding layer.
            model_type: Type of model to use for the invertible GNN.
        """
        super().__init__(model_id, variables, save_dir, device)
        self.base_distribution_type = base_distribution_type
        self.dense_init = dense_init
        self.embedding_size = embedding_size
        self.device = device
        self.lambda_sparse = lambda_sparse
        self.scale_noise = scale_noise
        self.scale_noise_p = scale_noise_p
        self.num_particles = num_chains
        self.num_nodes = variables.num_groups
        self.model_type = model_type
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.processed_dim_all = variables.num_processed_non_aux_cols
        self.spline_bins = spline_bins
        self.variables = variables
        # Set up the Neural Nets
        self.res_connection = res_connection
        self.norm_layer = nn.LayerNorm if norm_layers else None
        ICGNN = self._create_ICGNN_for_bayesdag(num_chains=num_chains)
        self.ICGNN, self.icgnn_params, self.icgnn_buffers = combine_state_for_ensemble_icgnn(ICGNN)
        _ = [p.requires_grad_() for p in self.icgnn_params]
        self.likelihoods = nn.ModuleDict(
            self._generate_error_likelihoods(
                self.base_distribution_type, self.variables, num_particles=num_chains
            )
        )
        self.posterior_graph_dist, self.posterior_graph_params = None, nn.ParameterDict({})
        # Adding a buffer to hold the log likelihood. This will be saved with the state dict.
        self.register_buffer("log_p_x", torch.tensor(-np.inf))
        self.log_p_x: torch.Tensor  # This is simply a scalar.

    def get_extra_state(self) -> Dict[str, Any]:
        """Return extra state for the model, including the data processor."""
        return {
            "data_processor": self.data_processor
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """Load the state dict including data processor.

        Args:
            state: State to load.
        """
        if "data_processor" in state:
            self.data_processor = state["data_processor"]
        if "neg_constraint_matrix" in state:
            self.neg_constraint_matrix = state["neg_constraint_matrix"]
        if "pos_constraint_matrix" in state:
            self.pos_constraint_matrix = state["pos_constraint_matrix"]

    def _create_ICGNN_for_bayesdag(self, num_chains: int = 1) -> List[ContractiveInvertibleGNN]:
        """
        This creates the SEM used for BayesDAG. 
        Returns:
            A list of the ICGNN networks corresponding to each SG-MCMC chain.
        """
        return [
            ContractiveInvertibleGNN(
                torch.tensor(self.variables.group_mask),
                self.device,
                norm_layer=self.norm_layer,
                res_connection=self.res_connection,
                encoder_layer_sizes=self.encoder_layer_sizes,
                decoder_layer_sizes=self.decoder_layer_sizes,
                embedding_size=self.embedding_size,
                model_type=self.model_type,
            )
            for _ in range(num_chains)
        ]

    def _generate_error_likelihoods(
        self, base_distribution_string: str, variables: Variables, num_particles: Optional[int]
    ) -> Dict[str, nn.Module]:
        """
        Instantiate error likelihood models for each variable in the SEM.
        For continuous variables, the likelihood is for an additive noise model whose exact form is determined by
        the `base_distribution_string` argument (see below for exact options). For vector-valued continuous variables,
        we assume the error model factorises over dimenions.
        For discrete variables, the likelihood directly parametrises the per-class probabilities.
        For sampling these variables, we rely on the Gumbel max trick. For this to be implemented correctly, this method
        also returns a list of subarrays which should be treated as single categorical variables and processed with a
        `max` operation during sampling.

        Args:
            base_distribution_string: which type of base distribution to use for the additive noise SEM
            Options:
                fixed_gaussian: Gaussian with non-leanrable mean of 0 and variance of 1
                gaussian: Gaussian with fixed mean of 0 and learnable variance
                spline: learnable flow transformation which composes an afine layer a spline and another affine layer
        Returns:
            A dictionary from variable type to the likelihood distribution(s) for variables of that type.
        """

        conditional_dists: Dict[str, nn.Module] = {}
        typed_regions = variables.processed_cols_by_type
        # Continuous
        continuous_range = [i for region in typed_regions["continuous"] for i in region]
        if continuous_range:
            dist: nn.Module
            if base_distribution_string == "fixed_gaussian":
                dist = GaussianBase(len(continuous_range), device=self.device, train_base=False)
            if base_distribution_string == "gaussian":
                dist = GaussianBase(
                    len(continuous_range), device=self.device, train_base=True, num_particles=num_particles
                )
            elif base_distribution_string == "spline":
                dist = DiagonalFLowBase(
                    len(continuous_range),
                    device=self.device,
                    num_bins=self.spline_bins,
                    flow_steps=1,
                )
            else:
                raise NotImplementedError("Base distribution type not recognised")
            conditional_dists["continuous"] = dist

        # Binary
        binary_range = [i for region in typed_regions["binary"] for i in region]
        if binary_range:
            conditional_dists["binary"] = BinaryLikelihood(len(binary_range), device=self.device)

        # Categorical
        if "categorical" in typed_regions:
            conditional_dists["categorical"] = nn.ModuleList(
                [CategoricalLikelihood(len(region), device=self.device) for region in typed_regions["categorical"]]
            )

        return conditional_dists

    @classmethod
    def name(cls) -> str:
        return "BayesDAG"

    def extract_icgnn_weights(self, num_particles):
        assert num_particles == 1
        return self.icgnn_params, self.icgnn_buffers

    def get_weighted_adj_matrix(
        self,
        samples: int = 100,
        squeeze: bool = False,
    ) -> torch.Tensor:
        """
        Returns the weighted adjacency matrix (or several) as a tensor.
        Args:
            samples: Number of samples to draw.
            squeeze: Whether to squeeze the first dimension if samples=1.
        """
        A_samples, _ = self.get_adj_matrix_tensor(samples)
        params, buffers = self.extract_icgnn_weights(num_particles=A_samples.shape[0], use_param_weights=False)
        W_adjs = A_samples * vmap(self.ICGNN.get_weighted_adjacency)(params, buffers)

        if squeeze and samples == 1:
            W_adjs = W_adjs.squeeze(0)
        return W_adjs, params, buffers


    def dagness_factor(self, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the dag penalty for matrix A as trace(expm(A)) - dim.

        Args:
            A: Binary adjacency matrix, size (input_dim, input_dim).
        """
        return torch.diagonal(torch.matrix_exp(A), dim1=-1, dim2=-2).sum(-1) - self.num_nodes

    def _log_prob(
        self, x: torch.Tensor, predict: torch.Tensor, intervention_mask: Optional[torch.Tensor] = None, **_
    ) -> torch.Tensor:
        """
        Computes the log probability of the observed data given the predictions from the SEM.

        Args:
            x: Array of size (processed_dim_all) or (batch_size, processed_dim_all), works both ways (i.e. single sample
            or batched).
            predict: tensor of the same shape as x.
            intervention_mask (num_nodes): optional array containing indicators of variables that have been intervened upon.
            These will not be considered for log probability computation.

        Returns:
            Log probability of non intervened samples. A number if x has shape (input_dim), or an array of
            shape (batch_size) is x has shape (batch_size, input_dim).
        """
        typed_regions = self.variables.processed_cols_by_type
        x = x.unsqueeze(1)  # N x 1 x D predict: N x china x D
        if predict.dim() == 5:
            return self._double_eltwise_log_prob(x, predict=predict, intervention_mask=intervention_mask)
        # Continuous
        cts_bin_log_prob = torch.zeros_like(predict)
        continuous_range = [i for region in typed_regions["continuous"] for i in region]
        if continuous_range:
            cts_bin_log_prob[..., continuous_range] = self.likelihoods["continuous"].log_prob(
                x[..., continuous_range] - predict[..., continuous_range]
            )

        if intervention_mask is not None:
            cts_bin_log_prob[..., intervention_mask] = 0.0

        log_prob = cts_bin_log_prob.sum(-1)

        return log_prob

    def _double_eltwise_log_prob(
        self, x: torch.Tensor, predict: torch.Tensor, intervention_mask: Optional[torch.Tensor] = None, **_
    ) -> torch.Tensor:
        """
        Computes the log probability of the observed data given the predictions from the SEM.

        Args:
            x: Array of size (processed_dim_all) or (batch_size, processed_dim_all), works both ways (i.e. single sample
            or batched).
            predict: tensor of the same shape as x.
            intervention_mask (num_nodes): optional array containing indicators of variables that have been intervened upon.
            These will not be considered for log probability computation.

        Returns:
            Log probability of non intervened samples. A number if x has shape (input_dim), or an array of
            shape (batch_size) is x has shape (batch_size, input_dim).
        """
        typed_regions = self.variables.processed_cols_by_type
        if x.dim() == 2:
            x = x.unsqueeze(1)  # N x 1 x D predict: N x china x D
        # Continuous
        cts_bin_log_prob = torch.zeros_like(predict)
        continuous_range = [i for region in typed_regions["continuous"] for i in region]
        if continuous_range:
            cts_bin_log_prob[..., continuous_range] = vmap(self.likelihoods["continuous"].log_prob)(
                x[..., continuous_range] - predict[..., continuous_range]
            )

        # Binary
        binary_range = [i for region in typed_regions["binary"] for i in region]
        if binary_range:
            cts_bin_log_prob[..., binary_range] = vmap(self.likelihoods["binary"].log_prob)(
                x[..., binary_range], predict[..., binary_range]
            )

        if intervention_mask is not None:
            cts_bin_log_prob[..., intervention_mask] = 0.0

        log_prob = cts_bin_log_prob.sum(-1)

        # Categorical
        if "categorical" in typed_regions:
            for region, likelihood, idx in zip(
                typed_regions["categorical"],
                self.likelihoods["categorical"],
                self.variables.var_idxs_by_type["categorical"],
            ):
                # Can skip likelihood computation completely if intervened
                if (intervention_mask is None) or (intervention_mask[idx] is False):
                    log_prob += likelihood.log_prob(x[..., region], predict[..., region])

        return log_prob

    def _sample_base(self, Nsamples: int) -> torch.Tensor:
        """
        Draw samples from the base distribution.

        Args:
            Nsamples: Number of samples to draw

        Returns:
            torch.Tensor z of shape (batch_size, input_dim).
        """
        sample = self.likelihoods["continuous"].sample(Nsamples)
        return sample

    def sample(
        self,
        Nsamples: int = 100,
        most_likely_graph: bool = False,
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        samples_per_graph: int = 1,
    ) -> torch.Tensor:
        """
        Draws samples from the causal flow model. Optionally these can be subject to an intervention of some variables

        Args:
            Nsamples: int containing number of samples to draw
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior or to draw a new graph for every sample
            intervention_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) optional array containing values for variables that have been intervened.
            samples_per_graph: how many samples to draw per graph sampled from the posterior. If most_likely_graph is true, this variable will default to 1.
        Returns:
            Samples: torch.Tensor of shape (Nsamples, input_dim).
        """
        if most_likely_graph:
            assert (
                Nsamples == samples_per_graph
            ), "When setting most_likely_graph=True, the number of samples should equal the number of samples per graph"
        else:
            assert Nsamples % samples_per_graph == 0, "Nsamples must be a multiple of samples_per_graph"

        (intervention_idxs, intervention_mask, intervention_values,) = intervention_to_tensor(
            intervention_idxs,
            intervention_values,
            self.variables.group_mask,
            device=self.device,
        )
        gumbel_max_regions = self.variables.processed_cols_by_type["categorical"]
        gt_zero_region = [j for i in self.variables.processed_cols_by_type["binary"] for j in i]

        num_graph_samples = Nsamples // samples_per_graph
        W_adj_samples, params, buffers = self.get_weighted_adj_matrix(
            samples=num_graph_samples,
        )
        Z = self._sample_base(Nsamples)
        Z = Z.view(num_graph_samples, samples_per_graph, -1)
        samples = vmap(self.ICGNN.simulate_SEM, in_dims=(0, 0, 0, 0, None, None))(
            params,
            buffers,
            Z,
            W_adj_samples,
            intervention_mask,
            intervention_values,
        ).detach()

        return samples

    def log_prob(
        self,
        X: Union[torch.Tensor, np.ndarray],
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        Ngraphs: Optional[int] = 1000,
        fixed_seed: Optional[int] = None,
    ):

        """
        Evaluate log-probability of observations X. Optionally this evaluation can be subject to an intervention on our causal model.
        Then, the log probability of non-intervened variables is computed.

        Args:
            X: torch.Tensor of shape (Nsamples, input_dim) containing the observations we want to evaluate
            Nsamples: int containing number of graph samples to draw.
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior instead of sampling graphs
            intervention_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) optional array containing values for variables that have been intervened.
            conditioning_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that we condition on.
            conditioning_values: torch.Tensor of shape (input_dim) optional array containing values for variables that we condition on.
            Nsamples_per_graph: int containing number of samples to draw
            Ngraphs: Number of different graphs to sample for graph posterior marginalisation. If None, defaults to Nsamples
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior or to draw a new graph for every sample
            fixed_seed: The integer to seed the random number generator (unused)

        Returns:
            log_prob: torch.tensor  (Nsamples)
        """

        if fixed_seed is not None:
            raise NotImplementedError("Fixed seed not supported by BayesDAG.")

        (X,) = to_tensors(X, device=self.device, dtype=torch.float)

        (intervention_idxs, intervention_mask, intervention_values,) = intervention_to_tensor(
            intervention_idxs,
            intervention_values,
            self.variables.group_mask,
            device=self.device,
        )

        if intervention_mask is not None and intervention_values is not None:
            X[:, intervention_mask] = intervention_values

        log_prob_samples = []
        batch_size = 10

        W_adj_samples, params, buffers = self.get_weighted_adj_matrix( samples=Ngraphs)
        # This sets certain elements of W_adj to 0, to respect the intervention
        W_adj = intervene_graph(W_adj_samples, intervention_idxs, copy_graph=False)
        log_prob_samples = []
        curr = 0
        while curr < X.shape[0]:
            curr_batch_size = min(batch_size, X.shape[0] - curr)
            with torch.no_grad():
                if params is None:
                    predict = self.ICGNN.predict(X[curr : curr + curr_batch_size], W_adj).transpose(0, 1).unsqueeze(-2)
                    W = W_adj_samples if self.base_distribution_type == "conditional_spline" else None
                    log_prob_samples.append(
                        self._log_prob(X[curr : curr + curr_batch_size], predict, intervention_mask, W=W).squeeze()
                    )  # (B)
                else:
                    predict = vmap(self.ICGNN.predict, in_dims=(0, 0, None, 0))(
                        params, buffers, X[curr : curr + curr_batch_size], W_adj
                    )
                    predict_shape = predict.shape
                    predict = predict.reshape(-1, self.num_particles, *predict_shape[1:])
                    log_prob_samples.append(
                        self._log_prob(X[curr : curr + curr_batch_size], predict, intervention_mask).reshape(
                            Ngraphs, -1
                        )
                    )  # (B)
            curr += curr_batch_size
        # Note that the W input is for AR-DECI, DECI will not use W.

        log_prob = torch.cat(log_prob_samples, dim=-1)
        return log_prob.detach().cpu().numpy().astype(np.float64)

    def print_tracker(self, inner_step: int, tracker: dict) -> None:
        """Prints formatted contents of loss terms that are being tracked."""
        tracker_copy = tracker.copy()

        loss = np.mean(tracker_copy.pop("loss")[-100:])
        log_p_x = np.mean(tracker_copy.pop("log_p_x")[-100:])
        penalty_dag = np.mean(tracker_copy.pop("penalty_dag")[-100:])
        log_p_A_sparse = np.mean(tracker_copy.pop("log_p_A_sparse")[-100:])
        log_q_A = np.mean(tracker_copy.pop("log_q_A")[-100:])
        h_filled = np.mean(tracker_copy.pop("imputation_entropy")[-100:])
        reconstr = np.mean(tracker_copy.pop("reconstruction_mse")[-100:])

        out = (
            f"Inner Step: {inner_step}, loss: {loss:.2f}, log p(x|A): {log_p_x:.2f}, dag: {penalty_dag:.8f}, "
            f"log p(A)_sp: {log_p_A_sparse:.2f}, log q(A): {log_q_A:.3f}, H filled: {h_filled:.3f}, rec: {reconstr:.3f}"
        )

        for k, v in tracker_copy.items():
            out += f", {k}: {np.mean(v[-100:]):.3g}"

        print(out)

    def process_dataset(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        variables: Optional[Variables] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the training data and mask.
        Args:
            dataset: Dataset to use.
            train_config_dict: Dictionary with training hyperparameters.
        Returns:
            Tuple with data and mask tensors.
        """
        if train_config_dict is None:
            train_config_dict = {}
        if variables is None:
            variables = self.variables

        self.data_processor = DataProcessor(
            variables,
            unit_scale_continuous=False,
            standardize_data_mean=train_config_dict.get("standardize_data_mean", False),
            standardize_data_std=train_config_dict.get("standardize_data_std", False),
        )
        processed_dataset = self.data_processor.process_dataset(dataset)
        true_adj = dataset.get_adjacency_data_matrix().astype(np.float32)
        self.posterior_graph_params["adjacency_matrix"] = torch.tensor(true_adj, dtype=torch.float32)

        data, mask = processed_dataset.train_data_and_mask
        data = data.astype(np.float32)
        return data, mask

    def _create_dataset_for_bayesdag(
        self, dataset: Dataset, train_config_dict: Dict[str, Any]
    ) -> Tuple[Union[DataLoader, FastTensorDataLoader], int]:
        """
        Create a data loader. For static deci, return an instance of FastTensorDataLoader, along with the number of samples.
        Consider override this if require customized dataset and dataloader.
        Args:
            dataset: Dataset to generate a dataloader for.
            train_config_dict: Dictionary with training hyperparameters.
        Returns:
            dataloader: Dataloader for the dataset during training.
            num_samples: Number of samples in the dataset.
        """
        data, mask = self.process_dataset(dataset, train_config_dict)
        dataloader = FastTensorDataLoader(
            *to_tensors(data, mask, device=self.device),
            batch_size=train_config_dict["batch_size"],
            shuffle=True,
        )
        return dataloader, data.shape[0]

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        Runs training.
        """
        pass

    def evaluate_metrics(self, dataset: Dataset):
        """
        Evluate the metrics for a given dataset on the model (self)

        Args:
            dataset (_type_): _description_

        Returns:
            _type_: _description_
        """
        adj_matrix, is_dag = self.get_adj_matrix(samples=100)
        adj_true = dataset.get_adjacency_data_matrix()
        subgraph_mask = dataset.get_known_subgraph_mask_matrix()
        adj_metrics = edge_prediction_metrics_multisample(
            adj_true,
            adj_matrix,
            adj_matrix_mask=subgraph_mask,
            compute_mean=True,
            is_dag = is_dag,
        )
        adj_metrics["num_dags"] = int(is_dag.sum())
        if dataset.true_posterior:
            adj_metrics["mmd-tp"] = mmd_true_posterior(
                log_marginal_likelihood_true_posterior=dataset.true_posterior["log_marginal_likelihood"],
                all_graphs=dataset.true_posterior["all_graphs"],
                graph_samples_from_model=adj_matrix,
                exp_edges_per_node=dataset.exp_edges_per_node,
                graph_type=dataset.graph_type,
            )
        else:
            adj_metrics["mmd-tp"] = -1
        val_data, _ = dataset.test_data_and_mask
        train_data, _ = dataset.train_data_and_mask
        adj_metrics["nll_val"] = nll(log_prob=self.log_prob, held_out_data=val_data)
        adj_metrics["nll_train"] = nll(log_prob=self.log_prob, held_out_data=train_data)
        return adj_metrics
