import copy
from typing import Dict, List, Optional, Tuple, Type

import torch
from functorch import FunctionalModuleWithBuffers
from functorch._src.make_functional import _swap_state, extract_weights, transpose_stack, extract_buffers
from torch import nn

from ...utils.torch_utils import generate_fully_connected


class ContractiveInvertibleGNN(nn.Module):
    """
    Given x, we can easily compute the exog noise z that generates it.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        device: torch.device,
        norm_layer: Optional[Type[nn.LayerNorm]] = None,
        res_connection: bool = True,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        embedding_size: Optional[int] = None,
        model_type: str = "nonlinear",
    ):
        """
        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1 when col j is in group i.
            device: Device used.
        """
        super().__init__()
        assert model_type in ["linear", "nonlinear"]
        self.model_type = model_type
        self.group_mask = group_mask.to(device)
        self.num_nodes, self.processed_dim_all = group_mask.shape
        self.device = device
        self.W = self._initialize_W()
        if self.model_type == "nonlinear":
            self.f = FGNNI(
                self.group_mask,
                self.device,
                norm_layer=norm_layer,
                res_connection=res_connection,
                layers_g=encoder_layer_sizes,
                layers_f=decoder_layer_sizes,
                embedding_size=embedding_size,
            )


    def _initialize_W(self) -> torch.Tensor:
        """
        Creates and initializes the weight matrix for adjacency.

        Returns:
            Matrix of size (num_nodes, num_nodes) initialized with zeros.

        Question: Initialize to zeros??
        """
        W = 0.1*torch.randn(self.num_nodes, self.num_nodes, device=self.device)
        return nn.Parameter(W, requires_grad=True)

    def get_weighted_adjacency(self) -> torch.Tensor:
        """
        Returns the weights of the adjacency matrix.
        """
        W_adj = self.W * (1.0 - torch.eye(self.num_nodes, device=self.device))  # Shape (num_nodes, num_nodes)
        return W_adj

    def predict(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """
        Gives the prediction of each variable given its parents.

        Args:
            X: Output of the GNN after reaching a fixed point, batched. Array of size (batch_size, processed_dim_all).
            W_adj: Weighted adjacency matrix, possibly normalized.

        Returns:
            predict: Predictions, batched, of size (B, n) that reconstructs X using the SEM.
        """

        #return torch.matmul(X,W_adj).transpose(0,1)
        if len(W_adj.shape) == 2:
            W_adj = W_adj.unsqueeze(0)
        if self.model_type == "linear":
            return torch.matmul(X, W_adj).transpose(-2,-3)
        else:
            return self.f.feed_forward(X, W_adj)
    def simulate_SEM(
        self,
        Z: torch.Tensor,
        W_adj: torch.Tensor,
        intervention_mask: Optional[torch.Tensor] = None,
        intervention_values: Optional[torch.Tensor] = None,
    ):
        """
        Given exogenous noise Z, computes the corresponding set of observations X, subject to an optional intervention
        generates it.
        For discrete variables, the exogeneous noise uses the Gumbel max trick to generate samples.

        Args:
            Z: Exogenous noise vector, batched, of size (B, num_mc, n)
            W_adj: Weighted adjacency matrix, possibly normalized. (n, n) if a single matrix should be used for all batch elements. Otherwise (num_mc, n, n)
            intervention_mask: torch.Tensor of shape (num_nodes) optional array containing binary flag of nodes that have been intervened.
            intervention_values: torch.Tensor of shape (processed_dim_all) optional array containing values for variables that have been intervened.
            gumbel_max_regions: a list of index lists `a` such that each subarray X[a] represents a one-hot encoded discrete random variable that must be
                sampled by applying the max operator.
            gt_zero_region: a list of indices such that X[a] should be thresholded to equal 1, if positive, 0 if negative. This is used to sample
                binary random variables. This also uses the Gumbel max trick implicitly

        Returns:
             X: Output of the GNN after reaching a fixed point, batched. Array of size (batch_size, processed_dim_all).
        """

        X = torch.zeros_like(Z)
        for _ in range(self.num_nodes):
            if intervention_mask is not None and intervention_values is not None:
                X[..., intervention_mask] = intervention_values.unsqueeze(0)
            X = self.predict(X, W_adj).squeeze() + Z

        if intervention_mask is not None and intervention_values is not None:
            if intervention_values.shape == X.shape:
                X[..., intervention_mask] = intervention_values
            else:
                X[..., intervention_mask] = intervention_values.unsqueeze(0)
        return X


class FGNNI(nn.Module):
    """
    Defines the function f for the SEM. For each variable x_i we use
    f_i(x) = f(e_i, sum_{k in pa(i)} g(e_k, x_k)), where e_i is a learned embedding
    for node i.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        device: torch.device,
        embedding_size: Optional[int] = None,
        out_dim_g: Optional[int] = None,
        norm_layer: Optional[Type[nn.LayerNorm]] = None,
        res_connection: bool = False,
        layers_g: Optional[List[int]] = None,
        layers_f: Optional[List[int]] = None,
    ):
        """
        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1 when col j is in group i.
            device: Device used.
            embedding_size: Size of the embeddings used by each node. If none, default is processed_dim_all.
            out_dim_g: Output dimension of the "inner" NN, g. If none, default is embedding size.
            layers_g: Size of the layers of NN g. Does not include input not output dim. If none, default
                      is [a], with a = max(2 * input_dim, embedding_size, 10).
            layers_f: Size of the layers of NN f. Does not include input nor output dim. If none, default
                      is [a], with a = max(2 * input_dim, embedding_size, 10)
        """
        super().__init__()
        self.group_mask = group_mask
        self.num_nodes, self.processed_dim_all = group_mask.shape
        self.device = device
        # Initialize embeddings
        self.embedding_size = embedding_size or self.processed_dim_all
        self.embeddings = self.initialize_embeddings()  # Shape (input_dim, embedding_size)
        # Set value for out_dim_g
        out_dim_g = out_dim_g or self.embedding_size
        # Set NNs sizes
        a = max(4 * self.processed_dim_all, self.embedding_size, 64)
        layers_g = layers_g or [a]
        layers_f = layers_f or [a]
        in_dim_g = self.embedding_size + self.processed_dim_all
        in_dim_f = self.embedding_size + out_dim_g
        self.g = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=out_dim_g,
            hidden_dims=layers_g,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )
        self.f = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=self.processed_dim_all,
            hidden_dims=layers_f,
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )

    def feed_forward(self, X: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """
        Computes non-linear function f(X, W) using the given weighted adjacency matrix.

        Args:
            X: Batched inputs, size (batch_size, processed_dim_all).
            W_adj: Weighted adjacency matrix, size (processed_dim_all, processed_dim_all) or size (num_mc, n, n).
        """

        if len(W_adj.shape) == 2:
            W_adj = W_adj.unsqueeze(0)
        W_adj = W_adj.unsqueeze(0)  # (1, num_mc. n, n)

        # Generate required input for g (concatenate X and embeddings)
        X = X.unsqueeze(1) 
        X_masked = X * self.group_mask 
        E = self.embeddings.expand(X.shape[0], -1, -1)
        X_in_g = torch.cat([X_masked, E], dim=2) 
        X_emb = self.g(X_in_g).unsqueeze(1)
        X_aggr_sum = torch.matmul(W_adj.transpose(-1, -2), X_emb)  
        X_in_f = torch.cat(
            [X_aggr_sum, E.unsqueeze(1).expand(-1, W_adj.shape[1], -1, -1)], dim=-1
        )  # Shape (batch_size, num_mc, num_nodes, out_dim_g + embedding_size)
        # Run f
        X_rec = self.f(X_in_f)  # Shape (batch_size, num_mc, num_nodes, processed_dim_all)
        # Mask and aggregate
        X_rec = X_rec * self.group_mask  # Shape (batch_size, num_mc, num_nodes, processed_dim_all)
        return X_rec.sum(-2)  # Shape (batch_size, num_mc, processed_dim_all)

    def initialize_embeddings(self) -> torch.Tensor:
        """
        Initialize the node embeddings.
        """
        aux = torch.randn(self.num_nodes, self.embedding_size, device=self.device) * 0.01  # (N, E)
        return nn.Parameter(aux, requires_grad=True)

def untranspose_stack(tuple_of_tensors, index, clone=False):
    if clone:
        results = tuple(shards[index].detach().clone() for shards in tuple_of_tensors)
    else:
        results = tuple(shards[index] for shards in tuple_of_tensors)
    return results

class FunctionalModuleCustomWithBuffersICGNN(FunctionalModuleWithBuffers):
    """
    Wrapper class for ICGNN that allows stateless modules, thus making it possible to use the module in an ensemble (as in across SG-MCMC chains).
    See https://github.com/pytorch/functorch/blob/1.10/functorch/_src/make_functional.py#L185
    """
    @staticmethod
    def _create_from(model, disable_autograd_tracking=False):
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = copy.deepcopy(model)
        params, param_names, param_names_map = extract_weights(model_copy)
        buffers, buffer_names, buffer_names_map = extract_buffers(model_copy)
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        return (
            FunctionalModuleCustomWithBuffersICGNN(
                model_copy, param_names, buffer_names, param_names_map, buffer_names_map
            ),
            params,
            buffers
        )

    def get_weighted_adjacency(self, params, buffers):
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(self.stateless_model, self.all_names_map, tuple(params) + tuple(buffers))
        try:
            return self.stateless_model.get_weighted_adjacency()
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.all_names_map, old_state)

    def predict(self, params, buffers, X, W_adj):
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(self.stateless_model, self.all_names_map, tuple(params) + tuple(buffers))
        try:
            return self.stateless_model.predict(X, W_adj)
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.all_names_map, old_state)

    def simulate_SEM(self, params, buffers, Z, W_adj,intervention_mask,intervention_values):
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(self.stateless_model, self.all_names_map, tuple(params) + tuple(buffers))
        try:
            return self.stateless_model.simulate_SEM(Z, W_adj,intervention_mask,intervention_values)
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.all_names_map, old_state)


def combine_state_for_ensemble_icgnn(models):
    """
    Combine the state and params of a list of models into a single state and params for an ensemble.
    See https://github.com/pytorch/functorch/blob/1.10/functorch/_src/make_functional.py#L339
    """
    if len(models) == 0:
        raise RuntimeError("combine_state_for_ensemble: Expected at least one model, got 0.")
    if not (all(m.training for m in models) or all(not m.training for m in models)):
        raise RuntimeError("combine_state_for_ensemble: Expected all models to " "have the same training/eval mode.")
    model0_typ = type(models[0])
    if not all(type(m) == model0_typ for m in models):
        raise RuntimeError("combine_state_for_ensemble: Expected all models to " "be of the same class.")
    funcs, params, buffers = zip(
        *[
            FunctionalModuleCustomWithBuffersICGNN._create_from(model)  # pylint: disable=protected-access
            for model in models
        ]
    )
    params = transpose_stack(params)
    buffers = transpose_stack(buffers)
    return funcs[0], params, buffers
