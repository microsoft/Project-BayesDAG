from __future__ import annotations

import os
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from functorch import vmap
from functorch._src.make_functional import transpose_stack
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ...datasets.dataset import CausalDataset, Dataset
from ...datasets.variables import Variables
from ...preprocessing.data_processor import DataProcessor
from ...utils.helper_functions import to_tensors, fill_triangular
from ...utils.torch_utils import generate_fully_connected
from ..optimizers import Adam_SGMCMC
from .bayesdag import BayesDAG
from .generation_functions import untranspose_stack
from functools import partial

from scipy.optimize import linear_sum_assignment


class BayesDAGNonLinear(BayesDAG):
    """
    Approximate Bayesian inference over the graph in a Gaussian nonlinear ANM based on the BayesDAG result. Any DAG G is represented as G = W * Step(grad (p))
    where W is a discrete matrix W in {0, 1} ^ {d, d} and p in R^d. Inference over DAGs G then corresponds to inference over W and p
    as the transformation is determinstic. This can be converted to inference over W and p by using the Gumbel-Sinkhorn trick.
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        lambda_sparse: float = 1.0,
        norm_layers: bool = False,
        res_connection: bool = False,
        num_chains: int = 10,
        sinkhorn_n_iter: int = 3000,
        scale_noise: float = 0.1,
        scale_noise_p: float = 1.0,
        model_type: str = "nonlinear",
        sparse_init: bool = False,
        input_perm: bool = False,
        VI_norm: bool = False,
    ):
        """
        Args:
            model_id: Unique model ID for this model instance of training.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
            device: Device to load model to.
            lambda_sparse: Coefficient for the prior term that enforces sparsity.
            norm_layers: bool indicating whether all MLPs should use layer norm
            res_connection:  bool indicating whether all MLPs should use layer norm
            num_chains: Number of chains to use for SG-MCMC
            sinkhorn_n_iter: Number of iterations for Sinkhorn
            scale_noise: Hyperparameter of the Adam SGMCMC for sampling theta
            scale_noise_p: Hyperparameter of the Adam SGMCMC for sampling p
            model_type: Type of model to use. Admits {"linear", "nonlinear"}
            sparse_init: Whether to initialize the W matrix to be sparse
            input_perm: Whether to use the input permutation to generate the adjacency matrix
            VI_norm: Whether to use layer norm in the helper network
        """
        super().__init__(
            model_id=model_id,
            variables=variables,
            save_dir=save_dir,
            device=device,
            lambda_sparse=lambda_sparse,
            base_distribution_type="gaussian",
            norm_layers=norm_layers,
            res_connection=res_connection,
            num_chains=num_chains,
            scale_noise=scale_noise,
            scale_noise_p=scale_noise_p,
            model_type=model_type,
        )
        self.input_perm = input_perm
        self.sparse_init = sparse_init
        self.VI_norm = VI_norm
        if self.sparse_init:
            self.logit_const = -1
        else:
            self.logit_const = 0
        
        if self.input_perm:
            hidden_size = 128
            layer_norm = partial(torch.nn.LayerNorm, elementwise_affine=True)
            self.helper_network = generate_fully_connected(
                input_dim=self.num_nodes*self.num_nodes,
                output_dim=(self.num_nodes * self.num_nodes),
                hidden_dims=[hidden_size, hidden_size],
                non_linearity=nn.ReLU,
                activation=None,
                device=device,
                res_connection=True,
                normalization=layer_norm
            )
        else:
            layer_norm = partial(torch.nn.LayerNorm, elementwise_affine=True)
            hidden_size = 48
            self.helper_network = generate_fully_connected(
                input_dim=self.num_nodes,
                output_dim=(self.num_nodes * self.num_nodes),
                hidden_dims=[hidden_size, hidden_size],
                non_linearity=nn.ReLU,
                activation=None,
                device=device,
                res_connection=True,
                normalization=layer_norm if self.VI_norm else None
            )
            
        if self.VI_norm:
            self.layer_norm = nn.LayerNorm(self.num_nodes, elementwise_affine=False)
        else:
            self.layer_norm = lambda x: x
        
        self.num_chains = num_chains
        self.o_scale = 10
        self.p_scale = 0.01
        self.p_buffer = deque(maxlen=5000)
        self.weights_buffer = deque(maxlen=5000)
        self.buffers_buffer = deque(maxlen=5000)
        self.p_steps = 0
        self.weights_steps = 0
        self.sinkhorn_n_iter = sinkhorn_n_iter
        self.num_burnin_steps = 1
        self.p = self.p_scale * torch.randn((self.num_chains, self.num_nodes), device=self.device)
        self.p.requires_grad = True

        self.p_opt = Adam_SGMCMC([self.p],
                                 lr=0.0003,
                                 betas=(0.9,0.99),
                                 dataset_size=5000,
                                 scale_noise=self.scale_noise_p,
                                 )
        #
        self.W_opt = torch.optim.Adam(list(self.helper_network.parameters())+[self.likelihoods["continuous"].logscale_base] , lr=0.005)
        self.weights_opt = Adam_SGMCMC(self.icgnn_params,
                                 lr=0.0003,
                                 betas=(0.9,0.99),
                                 dataset_size=5000,
                                 scale_noise=self.scale_noise,
                                 )

    @classmethod
    def name(cls) -> str:
        return "bayesdag_nonlinear"

    def compute_perm_hard(self, p:torch.Tensor):
        def log_sinkhorn_norm(log_alpha: torch.Tensor, tol= 1e-3):
            for _ in range(self.sinkhorn_n_iter):
                log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
                log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
                exp_log_alpha = log_alpha.exp()
                if torch.abs(1.-exp_log_alpha.sum(-1)).max()<tol and torch.abs(1.-exp_log_alpha.sum(-2)).max()<tol:
                    break
            return log_alpha.exp()

        O = self.o_scale * torch.arange(1, self.num_nodes+1, dtype=p.dtype, device=p.device).expand(1, -1)
        X = torch.matmul(p.unsqueeze(-1), O.unsqueeze(-2))
        
        perm = log_sinkhorn_norm(X / 0.2)
        
        perm_matrix = torch.zeros_like(perm)
        for i in range(perm.shape[0]):
            row_ind, col_ind = linear_sum_assignment(-perm[i].squeeze().cpu().detach().numpy())
            perm_indices = list(zip(row_ind, col_ind))            
            perm_indices = [(i,) + idx for idx in perm_indices]
            perm_indices = tuple(zip(*perm_indices))
            perm_matrix[perm_indices] = 1.0
        perm_matrix_hard = (perm_matrix - perm).detach() + perm # Straight Through
        return perm_matrix_hard, perm
    
    def transform_adj(self, p: torch.Tensor, detach_W: bool = False):
        """
        Takes in p and returns the adjacency matrix G = W * Step(grad (p)), equivalent to doing G = W* [sigma(p) x L x sigma(p)^T]
        See Theorem 3.2 in https://arxiv.org/abs/2307.13917.
        Args:
            p: Tensor of shape (num_chains, num_nodes) representing the p vector
            detach_W: Whether to detach the W matrix from the computation graph
        """
        perm_matrix_hard, perm = self.compute_perm_hard(p)
        
        if self.input_perm:
            helper_input = perm_matrix_hard.view(perm_matrix_hard.shape[0],-1)
        else:
            helper_input = self.layer_norm(p)

        W_vec_ = torch.distributions.RelaxedBernoulli(logits=self.helper_network(helper_input)+self.logit_const, temperature=0.2).rsample()
        if detach_W:
            W_vec_.detach()
        W_vec_hard = W_vec_.round()
        W_vec = (W_vec_hard - W_vec_).detach() + W_vec_
        W = W_vec.reshape(perm.shape[0], self.num_nodes, self.num_nodes)
        full_lower = torch.ones(perm.shape[0], int((self.num_nodes - 1) * self.num_nodes / 2)).to(p.device)
        adj_matrix = W * torch.matmul(
            torch.matmul(perm_matrix_hard, fill_triangular(full_lower)), perm_matrix_hard.transpose(-1, -2)
        )

        return adj_matrix
    
    def extract_icgnn_weights(self, use_param_weights, num_particles):
        if use_param_weights or len(self.weights_buffer)==0:
            params, buffers  = self.icgnn_params, self.icgnn_buffers
        else:
            tuple_tensor=tuple(self.weights_buffer.pop() for _ in range(num_particles))
            params = transpose_stack(tuple_tensor)
            tuple_tensor=tuple(self.buffers_buffer.pop() for _ in range(num_particles))
            buffers = transpose_stack(tuple_tensor)
        return params, buffers

    def data_likelihood(self, X: torch.Tensor, A_samples: torch.Tensor,  dataset_size:int, use_param_weights=False, return_prior: bool = False):
        """
        Computes the log likelihood of the data under the model, i.e. p(X | G:={w,p}, theta, sigma)
        Args:
            X: Tensor of shape (batch_size, num_nodes) representing the data
            A_samples: Tensor of shape (num_chains, num_nodes, num_nodes) representing the adjacency matrix
            return_prior: Whether to return the prior term as well
        """
        params, buffers = self.extract_icgnn_weights(use_param_weights=use_param_weights, num_particles=A_samples.shape[0])
        params_flat = torch.cat([param.view(A_samples.shape[0],-1) for param in params], dim=-1)
        if return_prior:
            theta_prior = 1/ dataset_size * (torch.distributions.normal.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.)).log_prob(
                params_flat) ).sum(-1) # num_chains
            p_prior = 1/dataset_size * (torch.distributions.normal.Normal(loc=torch.tensor(0.), scale=torch.tensor(0.1)).log_prob(self.p)).sum(-1) # chain

            sparse_loss = -(1/dataset_size) * self.lambda_sparse* A_samples.abs().sum(-1).sum(-1) # chain

        W_adj = A_samples * vmap(self.ICGNN.get_weighted_adjacency)(params, buffers)
        predict = vmap(self.ICGNN.predict, in_dims=(0, 0, None, 0))(params, buffers, X, W_adj)# N x num_chain x D #chain x  N x 1  x D

        log_p_base = self._log_prob(
            X, predict, W=A_samples if self.base_distribution_type == "conditional_spline" else None
        ).transpose(0,1)  #  N x num_chains
        if return_prior:
            return log_p_base, theta_prior, p_prior, sparse_loss

        return log_p_base

    def compute_W_prior_entropy(self, p: torch.Tensor, dataset_size: int):
        """
        Computes the prior and entropy terms for the W matrix (for VI).
        Args:
            p: Tensor of shape (num_chains, num_nodes) representing the p vector
            dataset_size: Size of the dataset
        """
        if self.input_perm:
            perm_matrix_hard, _ = self.compute_perm_hard(p)
            helper_input = perm_matrix_hard.view(perm_matrix_hard.shape[0],-1)
            logits = self.helper_network(helper_input)
        else:
            logits = self.helper_network(self.layer_norm(p)) # chain x( D x D)

        # draw hard samples
        W_vec_ = torch.distributions.RelaxedBernoulli(logits=logits+self.logit_const, temperature=0.2).rsample()
        W_vec_hard = W_vec_.round()
        W_vec = (W_vec_hard - W_vec_).detach() + W_vec_
        W = W_vec.reshape(p.shape[0], self.num_nodes, self.num_nodes) # chain x D x D
        prior = 1./dataset_size*torch.distributions.Bernoulli(probs=torch.tensor(0.5).to(device=W.device)).log_prob(W).sum(-1).sum(-1) # chain
        # compute entropy
        entropy = 1./dataset_size*torch.distributions.Bernoulli(logits=logits).entropy().sum(-1).sum(-1) # chain
        return prior, entropy

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
            variables: Information about variables/features used by this model.
        Returns:
            Tuple with data and mask arrays.
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

        data, mask = processed_dataset.train_data_and_mask
        data = data.astype(np.float32)
        return data, mask

    def _posterior_p_sample(self, data: torch.Tensor, dataset_size:int, num_samples: int = 1, writer: Optional[SummaryWriter] = None, interval:int=1) -> torch.Tensor:
        """
        SG-MCMC step for sampling p.
        Args:
            data: Tensor of shape (batch_size, num_nodes) representing the data
            num_samples: Number of samples to return.
            writer: Optional tensorboard SummaryWriter.
            dataset_size: Size of the dataset
            interval: Number of steps between logging to tensorboard
        """
        num_steps = num_samples * interval
        if self.p_steps < self.num_burnin_steps:
            num_steps = self.num_burnin_steps - self.p_steps + num_samples
        total_loss = 0.
        for cur_step in range(num_steps):
            self.weights_opt.zero_grad()
            self.W_opt.zero_grad()
            self.p_opt.zero_grad()
            A_samples = self.transform_adj(self.p, detach_W=False)
            ll_eltwise, _, p_prior, sparse_loss = self.data_likelihood(data, A_samples, dataset_size=dataset_size, use_param_weights=True, return_prior=True)
            loss = -(ll_eltwise+p_prior+1*sparse_loss).mean()  # 1 averaged over num_chains, batch sizes
            total_loss+= loss.detach()
            loss.backward()
            self.p_opt.step()
            if writer is not None:
                for jj in range(self.p.shape[0]):
                    writer_dict = {}
                    for j in range(self.p.shape[-1]):
                        writer_dict[str(j)] = self.p[jj, j].detach().cpu().numpy()
                    writer.add_scalars(f"p_chain_{jj}", writer_dict, self.p_steps)  # tensorboard
            self.p_steps += 1
        
            if self.p_steps >= self.num_burnin_steps and (cur_step+1)%interval == 0:
                for i in range(len(self.p)):
                    self.p_buffer.append(self.p[i].detach().clone())

        return total_loss/num_steps
        
    def _posterior_weights_sample(self, data: torch.Tensor, dataset_size:int, num_samples: int = 1) -> torch.Tensor:
        """
        SG-MCMC step for sampling the weights.
        Args:
            data: Tensor of shape (batch_size, num_nodes) representing the data
            num_samples: Number of samples to return.
            dataset_size: Size of the dataset
        """
        num_steps = num_samples
        if self.weights_steps < self.num_burnin_steps:
            num_steps = self.num_burnin_steps - self.weights_steps + num_samples

        total_loss = 0.
        for _ in range(num_steps):
            self.weights_opt.zero_grad()
            self.W_opt.zero_grad()
            self.p_opt.zero_grad()
            A_samples = self.transform_adj(self.p)
            ll_eltwise, theta_prior, _, sparse_loss = self.data_likelihood(data, A_samples, dataset_size=dataset_size, use_param_weights=True, return_prior=True)# batch x chain, num_chain
            
            loss = -(ll_eltwise+theta_prior+sparse_loss).mean()  #[]
            total_loss += loss.detach()
            loss.backward()
            self.weights_opt.step()
            self.weights_steps += 1

            if self.weights_steps >= self.num_burnin_steps:
                for i in range(self.num_chains):
                    self.weights_buffer.append(untranspose_stack(self.icgnn_params, i, clone=True))
                    self.buffers_buffer.append(untranspose_stack(self.icgnn_buffers, i, clone=True))
        return total_loss/num_steps
    
    def _train_helper_network(self, data: torch.Tensor, dataset_size, num_iters: int = 1)-> torch.Tensor:
        """
        VI step for training the helper network to generate W conditioned on p.
        Args:
            data: Tensor of shape (batch_size, num_nodes) representing the data
            num_iters: Number of iterations to train for.
            dataset_size: Size of the dataset
        """
        total_loss = 0.
        for _ in range(num_iters):
            self.weights_opt.zero_grad()
            self.W_opt.zero_grad()
            self.p_opt.zero_grad()
            A_samples = self.transform_adj(self.p)
            ll_eltwise,_,_,sparse_loss = self.data_likelihood(data, A_samples, dataset_size=dataset_size, use_param_weights=True, return_prior=True) # batch x chain
            prior, entropy = self.compute_W_prior_entropy(self.p, dataset_size=dataset_size) # chain
            loss = -(ll_eltwise+prior + entropy+sparse_loss).mean()  #
            total_loss += loss.detach()
            loss.backward()
            self.W_opt.step()
        return total_loss/num_iters

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        Runs training.
        Args:
            dataset: Dataset to use.
            train_config_dict: Dictionary with training hyperparameters.
            report_progress_callback: Optional callback function to report training progress.
        """
        if train_config_dict is None:
            train_config_dict = {}
        dataloader, _ = self._create_dataset_for_bayesdag(dataset, train_config_dict)

        # initialise logging machinery
        train_output_dir = os.path.join(self.save_dir, "train_output")
        os.makedirs(train_output_dir, exist_ok=True)
        log_path = os.path.join(train_output_dir, "summary")
        writer = SummaryWriter(log_path, flush_secs=1)

        print("Saving logs to", log_path, flush=True)
        tracker_loss_terms: Dict = defaultdict(list)
        
        self.dataset_size = dataset._train_data.shape[0]
        self.train_data, _ = to_tensors(*dataset.train_data_and_mask,device=self.device)

        best_loss = np.inf
        self.p_opt.update_dataset_size(self.dataset_size)
        self.weights_opt.update_dataset_size(self.dataset_size)
        # Outer optimization loop
        inner_opt_count = 0
        prev_best = 0
        for step in range(train_config_dict["max_epochs"]):
            loss_epoch = 0.
            
            for (x, _) in dataloader:
                p_loss = self._posterior_p_sample(data=x, dataset_size=self.dataset_size, num_samples=1, writer=writer)
                
                W_loss = self._train_helper_network(data=x, dataset_size=self.dataset_size,num_iters=1)
                weights_loss = self._posterior_weights_sample(data=x, dataset_size=self.dataset_size ,num_samples=1)
                loss = (p_loss+W_loss+weights_loss)/3
                loss_epoch += loss
            tracker_loss_terms["loss"].append(loss.mean().item())
            tracker_loss_terms["p_loss"].append(p_loss.item())
            tracker_loss_terms["W_loss"].append(W_loss.item())
            tracker_loss_terms["weights_loss"].append(weights_loss.item())
            inner_opt_count+=1
            if loss_epoch.item() < best_loss:
                best_loss = loss_epoch.item()
                print("New best model found. Saving Checkpoint")
                prev_best = 0
                self.save(best=True)
            else:
                prev_best +=1
            if step % 4 == 0:
                if (
                    isinstance(dataset, CausalDataset)
                    and dataset.has_adjacency_data_matrix
                    and not hasattr(self, "latent_variables")
                ):
                    
                    adj_metrics = self.evaluate_metrics(dataset=dataset)
                else:
                    adj_metrics = None
                self.print_tracker_sgld(step, tracker_loss_terms, adj_metrics)
                
            _log_epoch_metrics(writer=writer, tracker_loss_terms=tracker_loss_terms, adj_metrics=adj_metrics, step=step)

                
    def print_tracker_sgld(self, step: int, tracker: dict, adj_metrics: Optional[dict]) -> None:
        """Prints formatted contents of loss terms that are being tracked.
        Args:
            inner_step: Current step.
            tracker: Dictionary with loss terms.
            adj_metrics: Dictionary with adjacency matrix discovery metrics.
        """
        tracker_copy = tracker.copy()

        loss = np.mean(tracker_copy.pop("loss")[-100:])
        if adj_metrics is not None:
            adj_metrics_cp = adj_metrics.copy()
            shd = adj_metrics_cp.pop("shd")
            nnz = adj_metrics_cp.pop("nnz")
            cpdag_shd = adj_metrics_cp.pop("cpdag-shd", float("nan"))
            nll_val = adj_metrics_cp.pop("nll_val", float("nan"))
            nll_train = adj_metrics_cp.pop("nll_train", float("nan"))
            o_fscore = adj_metrics_cp.pop("orientation_fscore", float("nan"))
            mmd_tp = adj_metrics_cp.pop("mmd-tp", float("nan"))
        else:
            shd = float("nan")
            nnz = float("nan")
            nll_val = float("nan")
            cpdag_shd = float("nan")
            nll_train = float("nan")
            mmd_tp =float("nan")
            o_fscore = float("nan")

        print(
            f"Step: {step}, loss: {loss:.2f}, shd: {shd:.2f}, o_fscore:{o_fscore:.2f}, cpdag-shd: {cpdag_shd:.2f} nnz: {nnz:.2f} NLL-Validation: {nll_val:.4f} NLL-Train: {nll_train:.4f} MMD-TP: {mmd_tp:.4f} P_grad_norm:{torch.norm(self.p.grad).item()}", flush=True
        )
        
        
    def get_adj_matrix_tensor(
        self,
        samples: int = 5,
    ) -> torch.Tensor: 
        """
        Returns the adjacency matrix (or several) as a torch tensor.
        Args:
            samples: Number of samples to return.
        """
        batch_size = 500
        num_steps = int(np.ceil(samples/self.num_chains))
        for _ in range(num_steps):
            indices = torch.randperm(self.train_data.shape[0])[:batch_size]
            input_data = self.train_data[indices]
            self._posterior_p_sample(data=input_data, num_samples=1, dataset_size=self.dataset_size, interval=1)
            self._posterior_weights_sample(data=input_data,dataset_size=self.dataset_size, num_samples=1)

        p_vec= []
        for _ in range(samples):
            p_vec.append(self.p_buffer.pop())
        p_eval = torch.stack(p_vec)
        adj_matrix = self.transform_adj(p_eval) != 0.0
        return adj_matrix, torch.ones(samples)

    def get_adj_matrix(
        self,
        samples: int = 100,
        squeeze: bool = False,
    ) -> np.ndarray:
        """
        Returns the adjacency matrix (or several) as a numpy array.
        Args:
            samples: Number of samples to return.
            squeeze: Whether to squeeze the first dimension if samples == 1.
        """
        adj_matrix, is_dag = self.get_adj_matrix_tensor(samples=samples)
        if squeeze and samples == 1:
            adj_matrix = adj_matrix.squeeze(0)
        return adj_matrix.detach().cpu().numpy().astype(np.float64), is_dag.detach().cpu().numpy().astype(bool)


def _log_epoch_metrics(writer: SummaryWriter, tracker_loss_terms: dict, adj_metrics: Optional[dict], step: int):
    """
    Logging method for BayesDAG training loop
    Args:
        writer: tensorboard summarywriter used to log experiment results
        tracker_loss_terms: dictionary containing arrays with values generated at each inner-step during the inner optimisation procedure
        adj_metrics: Optional dictionary with adjacency matrix discovery metrics
        step: step number
    """

    # iterate over tracker vectors
    for key, value_list in tracker_loss_terms.items():
        writer.add_scalar(key, value_list[-1], step)  # tensorboard

    # log adjacency matrix metrics
    if adj_metrics is not None:
        for key, value in adj_metrics.items():
            writer.add_scalar(key, value, step)  # tensorboard