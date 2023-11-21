import math
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required


class RBFKernel:
    """
    A RBF kernel for use in the SVGD inference algorithm. The bandwidth of the kernel is chosen from the
    particles using a simple heuristic as in reference [1].

    References

    [1] "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,"
        Qiang Liu, Dilin Wang

    Implementation mainly based on https://docs.pyro.ai/en/stable/_modules/pyro/infer/svgd.html.
    """

    def __init__(self, bandwidth_factor: Optional[float] = None):
        """
        :param float bandwidth_factor: Optional factor by which to scale the bandwidth
        """
        self.bandwidth_factor = bandwidth_factor

    def _bandwidth(self, norm_sq: torch.Tensor):
        """
        Compute the bandwidth along each dimension using the median pairwise squared distance between particles.
        """
        num_particles = norm_sq.size(0)
        index = torch.arange(num_particles)
        norm_sq = norm_sq[index > index.unsqueeze(-1), ...]
        median = norm_sq.median(dim=0)[0]
        if self.bandwidth_factor is not None:
            median = self.bandwidth_factor * median
        assert median.shape == norm_sq.shape[-1:]
        return median / math.log(num_particles + 1)

    @torch.no_grad()
    def log_kernel_and_grad(self, particles: torch.Tensor):
        delta_x = particles.unsqueeze(0) - particles.unsqueeze(1)  # N N D
        assert delta_x.dim() == 3
        norm_sq = delta_x.pow(2.0)  # N N D
        h = self._bandwidth(norm_sq)  # D
        log_kernel = -(norm_sq / h)  # N N D
        grad_term = 2.0 * delta_x / h  # N N D
        assert log_kernel.shape == grad_term.shape
        return log_kernel, grad_term


class SVGD(Optimizer):
    """
    An optimizer for taking a single SVGD step using kernels.
    References

    [1] "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,"
        Qiang Liu, Dilin Wang

    Args:
        params: An iterable of tensors or dicts corresponding to particles with their corresponding optimisation hyperparams.
        lr: Default learning rate
        num_particles: Default number of particles in each of the param groups.
        kernel: Default kernel to be used for all param groups.
        maximize: Whether to maximize the function wrt each parameter group or minimize it.

    Implementation mainly based on https://docs.pyro.ai/en/stable/_modules/pyro/infer/svgd.html.
    """

    def __init__(
        self,
        params: Union[List, Tuple],
        lr: float = 0.1,
        num_particles: int = 20,
        kernel: str = "rbf",
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, maximize=maximize, num_particles=num_particles, kernel=kernel)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is not None:
                    self._svgd(
                        p.view(group["num_particles"], -1),
                        p.grad.view(group["num_particles"], -1),
                        kernel=group["kernel"],
                        lr=group["lr"],
                        num_particles=group["num_particles"],
                        maximize=group["maximize"],
                    )
        return loss

    def _svgd(
        self,
        params_with_grad: torch.Tensor,
        d_p: torch.Tensor,
        kernel: str,
        lr: float,
        num_particles: int,
        maximize: bool,
    ):
        """
        Update the parameters with gradient as in SVGD.

        Args:

            params_with_grad: Parameters which are to be updated with SVGD, i.e they are to be particles.
            d_p: the gradient of the loss (possibly ELBO) with respect to the parameter p.
            kernel: Kernel to be used for repulsion in SVGD.
            lr: Learning rate
            num_particles: Number of particles in the params.
            maximize: If the final objective is to maximized or minimized wrt the parameter.
        """
        if kernel == "rbf":
            _kernel = RBFKernel()
        else:
            raise NotImplementedError

        # compute kernel ingredients
        log_kernel, kernel_grad = _kernel.log_kernel_and_grad(params_with_grad)

        kxx = log_kernel.sum(-1).exp()
        assert kxx.shape == (num_particles, num_particles)
        attractive_grad = torch.mm(kxx, d_p)
        repulsive_grad = torch.einsum("nm,nm...->n...", kxx, kernel_grad)

        assert attractive_grad.shape == repulsive_grad.shape
        alpha = lr / num_particles if maximize else -lr / num_particles
        params_with_grad.add_((attractive_grad + repulsive_grad), alpha=alpha)

class Adam_SGMCMC(Optimizer):
    """ Adam inspired SGMCMC"""
    def __init__(self,
                 params,
                 dataset_size,
                 lr=0.001,
                 betas=(0.9,0.99),
                 scale_noise = 0.1
                 ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(
            lr=lr,
            betas=betas,
            dataset_size = dataset_size,
            scale_noise=scale_noise,
        )
        super().__init__(params, defaults)
    
    def update_dataset_size(self, new_dataset_size: int):  
        """Update the dataset_size attribute for all parameter groups."""  
        for param_group in self.param_groups:  
            param_group['dataset_size'] = new_dataset_size  

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = np.sqrt(group["lr"])
                dataset_size = group["dataset_size"]
                beta1,beta2 = group["betas"]
                gradient = parameter.grad.data
                scale_noise = group["scale_noise"]

                if len(state) == 0:
                    state["iteration"] = 0
                    state["p"] = torch.zeros_like(parameter)
                    state["V"] = torch.zeros_like(parameter)

                state["iteration"] += 1
                # Adam SGMCMC
                p = state["p"]
                V = state["V"]
                beta = (1-beta1)/lr

                V.mul_(beta2).add_((1-beta2)*(gradient*dataset_size)**2)
                G = 1./torch.sqrt(1e-8+torch.sqrt(1e-8+V))
                # change the constant here to adjust the added noise, 0 for optimization.
                p.mul_(beta1).add_(lr*G*gradient*dataset_size+scale_noise*torch.randn_like(parameter)*np.sqrt(2*lr*(beta-0.5*lr)))
                parameter.data.add_(-lr*G*p)
                mean_scale = (lr*G*p).abs().mean()
                # parameter.data.add_(0.1*mean_scale*torch.randn_like(parameter)*scale_noise)

                # Actual Adam
                # p.mul_(beta1).add_(gradient,alpha=1-beta1)
                # V.mul_(beta2).addcmul_(gradient,gradient,value=1-beta2)
                # p_unbias = p/1#(1-beta1**state["iteration"])
                # V_unbias = V/1#(1-beta2**state["iteration"])
                # parameter.data.addcdiv_(p_unbias,torch.sqrt(V_unbias)+1e-8, value=-lr)

        return loss




class pSGLD_v1(Optimizer):
    """Stochastic Gradient Langevin Dynamics Sampler with preconditioning."""

    def __init__(
        self,
        params,
        lr=1e-2,
        precondition_decay_rate=0.95,
        num_pseudo_batches=1,
        num_burn_in_steps=10,
        diagonal_bias=1e-8,
        scale_noise = 1.0
    ) -> None:
        """Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : Base learning rate for this optimizer.
        precondition_decay_rate : Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
        num_pseudo_batches : Effective number of minibatches in the data set. Trades off noise and prior with the SGD likelihood term.
            Note: Assumes loss is taken as mean over a minibatch.
            Otherwise, if the sum was taken, divide this number by the batch size.
            Default: `1`.
        num_burn_in_steps : Number of iterations to collect gradient statistics to update the preconditioner before starting to draw noisy samples.
        diagonal_bias : Term added to the diagonal of the preconditioner to prevent it from degenerating.

        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if num_burn_in_steps < 0:
            raise ValueError(f"Invalid num_burn_in_steps: {num_burn_in_steps}")

        defaults = dict(
            lr=lr,
            precondition_decay_rate=precondition_decay_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=diagonal_bias,
            scale_noise = scale_noise,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                precondition_decay_rate = group["precondition_decay_rate"]
                gradient = parameter.grad.data

                #  State initialization

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                state["iteration"] += 1

                momentum = state["momentum"]

                #  Momentum update
                momentum.add_((1.0 - precondition_decay_rate) * ((gradient**2) - momentum))

                if state["iteration"] > group["num_burn_in_steps"]:
                    sigma = 1.0 / torch.sqrt(torch.tensor(lr))
                else:
                    sigma = torch.zeros_like(parameter)

                preconditioner = 1.0 / torch.sqrt(momentum + group["diagonal_bias"])

                scaled_grad = 0.5 * preconditioner * gradient + 1*group["scale_noise"]/num_pseudo_batches* torch.normal(
                    mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)
                ) * sigma * torch.sqrt(preconditioner)
                parameter.data.add_(-lr * scaled_grad)

        return loss

class pSGLD_v2(Optimizer):
    """Implements pSGLD algorithm based on https://arxiv.org/pdf/1512.07666.pdf
    Built on the PyTorch RMSprop implementation
    (https://pytorch.org/docs/stable/_modules/torch/optim/rmsprop.html#RMSprop)
    Code implementation from https://github.com/alisiahkoohi/Langevin-dynamics/blob/master/langevin_sampling/precondSGLD.py
    """

    def __init__(self, params, lr=1e-2, beta=0.99, Lambda=1e-15, weight_decay=0,
                 centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= Lambda:
            raise ValueError("Invalid epsilon value: {}".format(Lambda))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".
                             format(weight_decay))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta value: {}".format(beta))

        defaults = dict(lr=lr, beta=beta, Lambda=Lambda, centered=centered,
            weight_decay=weight_decay)
        super(pSGLD_v2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pSGLD_v2, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('pSGLD does not support sparse '
                                       'gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['V'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                V = state['V']
                beta = group['beta']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                V.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(beta).add_(1 - beta, grad)
                    G = V.addcmul(
                        grad_avg, grad_avg, value=-1).sqrt_().add_(
                            group['Lambda'])
                else:
                    G = V.sqrt().add_(group['Lambda'])

                p.data.addcdiv_(grad, G, value=-group['lr'])

                noise_std = 2*group['lr']/G
                noise_std = noise_std.sqrt()
                noise = p.data.new(
                    p.data.size()).normal_(mean=0, std=1)*noise_std
                p.data.add_(noise)

        return G

class SGLD(Optimizer):
    """Implements SGLD algorithm based on
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
    Built on the PyTorch SGD implementation
    (https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/sgd.py)
    Implementation obtained from https://github.com/alisiahkoohi/Langevin-dynamics/blob/master/langevin_sampling/SGLD.py
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero ''dampening")
        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state[
                            'momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])
                noise_std = torch.Tensor([2*group['lr']])
                noise_std = noise_std.sqrt()
                noise = p.data.new(
                    p.data.size()).normal_(mean=0, std=1)*noise_std
                p.data.add_(noise)

        return 1.0