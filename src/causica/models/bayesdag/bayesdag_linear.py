from __future__ import annotations

import torch
from ...datasets.variables import Variables
from .bayesdag_nonlinear import BayesDAGNonLinear


class BayesDAGLinear(BayesDAGNonLinear):
    """
    Approximate Bayesian inference over the graph in a Gaussian linear ANM based on the BayesDAG result. Any DAG G is represented as G = W * Step(grad (p))
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
        num_chains: int = 10,
        sinkhorn_n_iter: int = 3000,
        scale_noise: float = 0.1,
        scale_noise_p: float = 1.0,
        VI_norm: bool = False,
        input_perm:bool = False,
        sparse_init:bool = False,
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
            num_chains=num_chains,
            sinkhorn_n_iter=sinkhorn_n_iter,
            scale_noise=scale_noise,
            scale_noise_p=scale_noise_p,
            model_type="linear",
            VI_norm=VI_norm,
            input_perm=input_perm,
            sparse_init=sparse_init
        )
    @classmethod
    def name(cls) -> str:
        return "bayesdag_linear"