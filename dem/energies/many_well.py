from __future__ import annotations

import math
from typing import Callable
import torch

from .base import Distribution, rejection_sampling
from .gauss import GMM, IsotropicGauss
import numpy as np

def rejection_sampling_gfn(shape: tuple, proposal: torch.distributions.Distribution,
                       target_log_prob_fn: Callable, k: float) -> torch.Tensor:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    n_samples = math.prod(shape)
    z_0 = proposal.sample((n_samples*10,))
    u_0 = torch.distributions.Uniform(0, k*torch.exp(proposal.log_prob(z_0)))\
        .sample().to(z_0)
    accept = torch.exp(target_log_prob_fn(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = (n_samples - samples.shape[0],)
        new_samples = rejection_sampling_gfn(required_samples, proposal, target_log_prob_fn, k)
        samples = torch.concat([samples, new_samples], dim=0)
        return samples


class DoubleWell(Distribution):
    def __init__(
        self,
        dim: int = 1,
        grid_points: int = 2001,
        rejection_sampling_scaling: float = 3.0,
        domain_delta: float = 2.5,
        **kwargs,
    ):
        if not dim == 1:
            raise ValueError("`dim` needs to be `1`. Consider using `MultiWell`.")
        super().__init__(dim=1, grid_points=grid_points, **kwargs)
        self.rejection_sampling_scaling = rejection_sampling_scaling

        self.log_norm_const = np.log(11784.50927)

        if self.domain is None:
            domain =  domain_delta * torch.tensor([[-1.0, 1.0]])
            self.set_domain(domain)
    def log_prob(self, x:torch.Tensor) -> torch.Tensor:
        return self.unnorm_log_prob(x) - self.log_norm_const
    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -x ** 4 + 6 * x ** 2 + 1 / 2 * x

    # def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    #     return -4 * x**3 + 12 * x + 0.5

    def marginal(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.pdf(x)

    def get_proposal_distr(self):
        device = self.domain.device
        loc = torch.tensor([-1.7, 1.7], device=device)     # shape: (2,)
        scale = torch.tensor([0.5, 0.5], device=device)    # shape: (2,)
        mixture_weights = torch.tensor([0.2, 0.8], device=device)

        mix = torch.distributions.Categorical(mixture_weights)               # batch_shape: ()
        com = torch.distributions.Normal(loc, scale)                         # batch_shape: (2,)
        proposal = torch.distributions.MixtureSameFamily(mix, com)   
        return proposal

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        proposal = self.get_proposal_distr()
        k = self.rejection_sampling_scaling*np.exp(self.log_norm_const)
        return rejection_sampling_gfn(
            shape=shape,
            proposal=proposal,
            target_log_prob_fn=self.unnorm_log_prob,
            k=k,
        )

    # def plots(self, samples, nbins=100) -> torch.Tensor:
    #     samples = self.sample((samples.shape[0],))
    #     fig = plot_marginal(
    #         x=samples,
    #         marginal=lambda x, **kwargs: self.pdf(x),
    #         dim=0,
    #         nbins=nbins,
    #         domain=self.domain,
    #     )

    #     x = torch.linspace(*self.domain[0], steps=nbins, device=self.domain.device)
    #     y = (
    #         self.get_proposal_distr().pdf(x.unsqueeze(-1))
    #         * self.rejection_sampling_scaling
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             x=x.cpu(),
    #             y=y.squeeze(-1).cpu(),
    #             mode="lines",
    #             name="proposal",
    #         )
    #     )
    #     return {"plots/rejection_sampling": fig}


class ManyWell(Distribution):
    def __init__(
        self,
        dim: int = 32,
        domain_dw_delta: float = 2.5,
        domain_gauss_scale: float = 5.0,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        self.n_double_wells = dim//2

        # Initialize distributions
        self.double_well = DoubleWell(domain_delta=domain_dw_delta)
        

        self.Z_x1 = 11784.50927
        self.logZ_x2 = 0.5 * np.log(2 * np.pi)
        self.double_well_log_norm_const = np.log(self.Z_x1) + self.logZ_x2
        self.log_norm_const = self.double_well_log_norm_const * self.n_double_wells
        # Set domain
        

    def compute_stats(self):
        # Double well
        self.double_well.compute_stats()
        self.log_norm_const = self.double_well.log_norm_const * self.n_double_wells
        self.expectations = {
            name: exp * self.n_double_wells
            for name, exp in self.double_well.expectations.items()
        }
        self.stddevs = torch.cat([self.double_well.stddevs] * self.n_double_wells)

        # Gauss
        if self.gauss is not None:
            self.gauss.compute_stats()
            self.log_norm_const += self.gauss.log_norm_const
            for name in self.expectations:
                # This assumes that the expectations are reducing the dim via a sum
                self.expectations[name] += self.gauss.expectations[name]
            self.stddevs = torch.cat([self.stddevs, self.gauss.stddevs])

        assert (self.pdf(self.domain.T) < 1e-5).all()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = torch.zeros((x.shape[0],), device=x.device)
        for i in range(self.dim):
            if i % 2 == 0:
                log_prob = log_prob + self.double_well.unnorm_log_prob(x[:, i:i+1]).squeeze(-1)
            else:
                log_prob += -0.5 * (x[:, i] ** 2)

        log_prob = log_prob.unsqueeze(-1)
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    # def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    #     score = torch.zeros_like(x, device=x.device)
    #     for i in range(self.dim):
    #         if i % 2 == 0:
    #             score[:, i] = self.double_well.score(x[:, i:i+1]).squeeze(-1)
    #         else:
    #             score[:, i] = -x[:, i]
    #     return score

    def marginal(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if dim < self.n_double_wells:
            return self.double_well.marginal(x)
        return self.gauss.marginal(x)

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()

        samples = []
        for i in range(self.dim):
            if i%2 ==0:
                samples.append(self.double_well.sample(shape))
            else:
                samples.append(torch.randn_like(samples[i-1]))
        
        samples = torch.stack(samples, dim=1)
        return samples