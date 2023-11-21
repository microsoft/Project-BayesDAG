from typing import Optional, Tuple

import torch
import torch.distributions as td
from torch import nn
from functorch import vmap
from math import pi

from .diagonal_flows import create_diagonal_spline_flow


# TODO: Add tests for base distributions, and ensure we are not duplicating any pytorch.distributions functionality unnecessarily
class GaussianBase(nn.Module):
    def __init__(self, input_dim: int, device: torch.device, train_base: bool = True, num_particles: Optional[int] = None):
        """
        Gaussian base distribution with 0 mean and optionally learnt variance. The distribution is factorised to ensure SEM invertibility.
            The mean is fixed. This class provides an interface analogous to torch.distributions, exposing .sample and .log_prob methods.

        Args:
            input_dim: dimensionality of observations
            device: torch.device on which object should be stored
            train_base: whether to fix the variance to 1 or learn it with gradients
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.num_particles = num_particles
        self.mean_base, self.logscale_base = self._initialize_params_base_dist(train=train_base)

    def _initialize_params_base_dist(self, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the parameters of the base distribution.

        Args:
            train: Whether the distribution's parameters are trainable or not.
        """
        if self.num_particles is None:
            mean = nn.Parameter(torch.zeros(self.input_dim, device=self.device), requires_grad=False)
            logscale = nn.Parameter(torch.zeros(self.input_dim, device=self.device), requires_grad=train)
        else:
            mean = nn.Parameter(torch.zeros(self.num_particles, self.input_dim, device=self.device), requires_grad=False)
            logscale = nn.Parameter(torch.zeros(self.num_particles, self.input_dim, device=self.device), requires_grad=train)
        # logscale = nn.Parameter(torch.tensor([-1.6676, -0.5876, -1.1042, -1.1596], device=self.device), requires_grad=train)
        # logscale = nn.Parameter(torch.tensor([-1.2065e-3, -1.3657, -3.1529e-1, -1.1952], device=self.device), requires_grad=train)

        return mean, logscale

    def log_prob(self, z: torch.Tensor):
        """
        Returns a the log-density per sample and dimension of z

        Args:
            z (batch, input_dim)

        Returns:
            log_prob (batch, input_dim)
        """
        if self.num_particles:
            return vmap(self.log_prob_vmap)(z, self.mean_base, self.logscale_base)
        else:
            dist = td.Normal(self.mean_base, torch.exp(self.logscale_base))
            return dist.log_prob(z)
    
    def log_prob_vmap(self, z, mean, logscale):
        logvar = 2*logscale

        return -0.5*(torch.log(torch.tensor(2*pi))+logvar + ((mean - z)**2/(torch.exp(logvar)+1e-7)))

    def sample(self, Nsamples: int):
        """
        Draw samples

        Args:
            Nsamples

        Returns:
            samples (Nsamples, input_dim)
        """
        dist = td.Normal(self.mean_base, torch.exp(self.logscale_base))
        if self.num_particles:
            n_actual_sample=int(torch.ceil(torch.tensor(Nsamples/self.num_particles)).item())
            samples = dist.sample((n_actual_sample,))
            samples = samples.reshape(-1,samples.shape[-1])
            return samples[:Nsamples]
        return dist.sample((Nsamples,))


class DiagonalFLowBase(nn.Module):
    def __init__(self, input_dim: int, device: torch.device, num_bins: int = 8, flow_steps: int = 1) -> None:
        """
        Learnable base distribution based on a composite affine-spline transformation of a standard Gaussian. The distribution is factorised to ensure SEM invertibility.
           This means that the flow acts dimension-wise, without sharing information across dimensions.
           This class provides an interface analogous to torch.distributions, exposing .sample and .log_prob methods.

        Args:
            input_dim: dimensionality of observations
            device: torch.device on which object should be stored
            num_bins: ow many bins to use for spline transformation
            flow_steps: how many affine-spline steps to take. Recommended value is 1.
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.transform = create_diagonal_spline_flow(
            flow_steps=flow_steps, features=self.input_dim, num_bins=num_bins, tail_bound=3
        ).to(self.device)
        self.base_dist = td.Normal(
            loc=torch.zeros(self.input_dim, device=self.device),
            scale=torch.ones(self.input_dim, device=self.device),
            validate_args=None,
        )

    def log_prob(self, z: torch.Tensor):
        """
        Returns a the log-density per sample and dimension of z

        Args:
            z (batch, input_dim)

        Returns:
            log_prob (batch, input_dim)
        """
        u, logdet = self.transform.inverse(z)
        log_pu = self.base_dist.log_prob(u)
        return logdet + log_pu

    def sample(self, Nsamples: int):
        """
        Draw samples

        Args:
            Nsamples

        Returns:
            samples (Nsamples, input_dim)
        """
        with torch.no_grad():
            u = self.base_dist.sample((Nsamples,))
            z, _ = self.transform.forward(u)
        return z


class CategoricalLikelihood(nn.Module):
    def __init__(self, input_dim: int, device: torch.device):
        """
        Discrete likelihood model. This model learns a base probability distribution.
        At evaluation time, it takes in an additional input which is added to this base distribution.
        This allows us to model both conditional and unconditional nodes.

        Args:
            input_dim: dimensionality of observations
            device: torch.device on which object should be stored
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.base_logits = nn.Parameter(torch.zeros(self.input_dim, device=self.device), requires_grad=True)
        self.softplus = nn.Softplus()

    def log_prob(self, x: torch.Tensor, logit_deltas: torch.Tensor):
        """
        The modelled probability distribution for x is given by `self.base_logits + logit_deltas`.
        This method returns the log-density per sample.

        Args:
            x (batch, input_dim): Should be one-hot encoded
            logit_deltas (batch, input_dim)

        Returns:
            log_prob (batch, input_dim)
        """
        dist = td.OneHotCategorical(logits=self.base_logits + logit_deltas, validate_args=False)
        return dist.log_prob(x)

    def sample(self, n_samples: int):
        """
        Samples Gumbels that can be used to sample this variable using the Gumbel max trick.
        This method does **not** return hard-thresholded samples.
        Args:
            n_samples

        Returns:
            samples (Nsamples, input_dim)
        """
        dist = td.Gumbel(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        return dist.sample((n_samples, self.input_dim)) + self.base_logits

    def posterior(self, x: torch.Tensor, logit_deltas: torch.Tensor):
        """
        A posterior sample of the Gumbel noise random variables given observation x and probabilities
        `self.base_logits + logit_deltas`.
        This methodology is described in https://arxiv.org/pdf/1905.05824.pdf.
        See https://cmaddis.github.io/gumbel-machinery for derivation of Gumbel posteriors.
        For a derivation of this exact algorithm using softplus, see https://www.overleaf.com/8628339373sxjmtvyxcqnx.

        Args:
            x (batch, input_dim): Should be one-hot encoded
            logit_deltas (batch, input_dim)

        Returns:
            z (batch, input_dim)
        """
        logits = self.base_logits + logit_deltas
        dist = td.Gumbel(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        top_sample = dist.sample((x.shape[0], 1)) + logits.logsumexp(-1, keepdim=True)
        lower_samples = dist.sample((x.shape[0], self.input_dim)) + logits
        lower_samples[x == 1] = float("inf")
        samples = top_sample - self.softplus(top_sample - lower_samples) - logits
        return samples + self.base_logits


class BinaryLikelihood(nn.Module):
    def __init__(self, input_dim: int, device: torch.device):
        """
        Binary likelihood model. This model learns a base probability distribution.
        At evaluation time, it takes in an additional input which is added to this base distribution.
        This allows us to model both conditional and unconditional nodes.

        Args:
            input_dim: dimensionality of observations
            device: torch.device on which object should be stored
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.base_logits = nn.Parameter(torch.zeros(self.input_dim, device=self.device), requires_grad=True)
        self.softplus = nn.Softplus()

    def log_prob(self, x: torch.Tensor, logit_deltas: torch.Tensor):
        """
        The modelled probability distribution for x is given by `self.base_logits + logit_deltas`.
        This method returns the log-density per sample.

        Args:
            x (batch, input_dim): Should be one-hot encoded
            logit_deltas (batch, input_dim)

        Returns:
            log_prob (batch, input_dim)
        """
        dist = td.Bernoulli(logits=self.base_logits + logit_deltas, validate_args=False)
        return dist.log_prob(x)

    def sample(self, n_samples: int):
        """
        Samples a Logistic random variable that can be used to sample this variable.
        This method does **not** return hard-thresholded samples.
        Args:
            n_samples

        Returns:
            samples (Nsamples, input_dim)
        """
        # The difference of two independent Gumbel(0, 1) variables is a Logistic random variable
        dist = td.Gumbel(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        g0 = dist.sample((n_samples, self.input_dim))
        g1 = dist.sample((n_samples, self.input_dim))
        return g1 - g0 + self.base_logits

    def posterior(self, x: torch.Tensor, logit_deltas: torch.Tensor):
        """
        A posterior sample of the logistic noise random variables given observation x and probabilities
        `self.base_logits + logit_deltas`.

        Args:
            x (batch, input_dim): Should be one-hot encoded
            logit_deltas (batch, input_dim)

        Returns:
            z (batch, input_dim)
        """
        logits = self.base_logits + logit_deltas
        dist = td.Gumbel(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        top_sample = dist.sample(x.shape)
        neg_log_prob_non_sampled = self.softplus(logits * x - logits * (1 - x))
        positive_sample = self.softplus(top_sample - dist.sample(x.shape) + neg_log_prob_non_sampled)
        sample = positive_sample * x - positive_sample * (1 - x) - logits
        return sample + self.base_logits
