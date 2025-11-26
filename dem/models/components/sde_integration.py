from contextlib import contextmanager

import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.sdes import VEReverseSDE
from dem.utils.data_utils import remove_mean


@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


def grad_E(x, energy_function):
    with torch.enable_grad():
        x = x.requires_grad_()
        return torch.autograd.grad(torch.sum(energy_function(x)), x)[0].detach()


def negative_time_descent(x, energy_function, num_steps, dt=1e-4, clipper=None):
    samples = []
    for _ in range(num_steps):
        drift = grad_E(x, energy_function)

        if clipper is not None:
            drift = clipper.clip_scores(drift)

        x = x + drift * dt

        if energy_function.is_molecule:
            x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)

        samples.append(x)
    return torch.stack(samples)


def euler_maruyama_step(
    sde: VEReverseSDE, t: torch.Tensor, x: torch.Tensor, dt: float, diffusion_scale=1.0
):
    # Calculate drift and diffusion terms
    drift = sde.f(t, x) * dt
    diffusion = diffusion_scale * sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x)

    # Update the state
    x_next = x + drift + diffusion
    return x_next, drift


def integrate_pfode(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    reverse_time: bool = True,
):
    start_time = 1.0 if reverse_time else 0.0
    end_time = 1.0 - start_time

    times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []
    with torch.no_grad():
        for t in times:
            x, f = euler_maruyama_step(sde, t, x, 1 / num_integration_steps)
            samples.append(x)

    return torch.stack(samples)


def integrate_sde(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    energy_function: BaseEnergyFunction,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    no_grad=True,
    time_range=1.0,
    negative_time=False,
    num_negative_time_steps=100,
    clipper=None,
    return_full_trajectory=False,
):
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time

    times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]

    x = x0
    samples = []
    log_pi_r = []
    log_q_r = []
    log_pi_f = []
    log_q_f = []
    with conditional_no_grad(no_grad):
        for t in times:
            # Euler-Maruyama update
            dt = time_range / num_integration_steps
            mu = sde.f(t, x) * dt
            sigma = diffusion_scale * sde.g(t, x) * np.sqrt(dt)
            diffusion = sigma * torch.randn_like(x)
            x_next = x + mu + diffusion
            sigma_next = diffusion_scale * sde.g(t - dt, x_next) * np.sqrt(dt)

            # Log probabilities
            if return_full_trajectory:
                log_pi_r.append(-0.5 * (((x_next - x - mu) / sigma) ** 2 + torch.log(sigma)).sum(dim=-1))
                log_q_r.append(-0.5 * (((x - x_next) / sigma_next) ** 2 + torch.log(sigma_next)).sum(dim=-1))

            x = x_next

            if energy_function.is_molecule:
                x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
            samples.append(x)

    samples = torch.stack(samples)
    if negative_time:
        print("doing negative time descent...")
        samples_langevin = negative_time_descent(
            x, energy_function, num_steps=num_negative_time_steps, clipper=clipper
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)

    if return_full_trajectory:
        with conditional_no_grad(no_grad):
            y = energy_function.energy.sample((samples.shape[1],))
            log_q_f.append(-energy_function(y))
            for t in reversed(times):
                # Euler-Maruyama update
                dt = time_range / num_integration_steps
                sigma = diffusion_scale * sde.g(t, y) * np.sqrt(dt)
                diffusion = sigma * torch.randn_like(y)
                y_prev = y + diffusion
                sigma_prev = diffusion_scale * sde.g(t + dt, y_prev) * np.sqrt(dt)
                mu_prev = sde.f(t + dt, y_prev) * dt

                # Log probabilities
                log_pi_f.append(-0.5 * (((y - y_prev - mu_prev) / sigma_prev) ** 2 + torch.log(sigma_prev)).sum(dim=-1))
                log_q_f.append(-0.5 * (((y_prev - y) / sigma) ** 2 + torch.log(sigma)).sum(dim=-1))

                y = y_prev

        return samples, y, log_pi_r, log_q_r, log_pi_f, log_q_f
    else:
        return samples