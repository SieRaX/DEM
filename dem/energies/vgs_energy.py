from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from fab.target_distributions import gmm
from fab.utils.plotting import plot_contours, plot_marginal_pair
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.replay_buffer import ReplayBuffer
from dem.utils.logging_utils import fig_to_image

from dem.energies.gauss import GMM
from dem.energies.funnel import Funnel
from dem.energies.many_well import ManyWell


class vgs_energy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        name = "25_gmm",
        device="cpu",
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        should_unnormalize=False,
        data_normalization_factor=None,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
        data_path_train=None,
    ):
        use_gpu = device != "cpu"
        torch.manual_seed(0)  # seed of 0 for GMM problem
        if name == "25_gmm":
            self.energy = GMM(
                name=name,
            )
        elif name == "funnel":
            self.energy = Funnel(
                dim=dimensionality,
            )
        elif name == "manywell":
            self.energy = ManyWell(
                dim=dimensionality,
            )
        else:
            raise ValueError("Unknown energy function name.")
        if use_gpu:
            self.energy.to(device)

        self.curr_epoch = 0
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.should_unnormalize = should_unnormalize
        self.data_normalization_factor = data_normalization_factor

        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size

        self.data_path_train = data_path_train

        self.name = name

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )

    def setup_test_set(self):
        test_sample = self.energy.sample((self.test_set_size,))
        return test_sample

    def setup_train_set(self):
        if self.data_path_train is None:
            train_samples = self.normalize(self.energy.sample((self.train_set_size,)))

        else:
            # Assume the samples we are loading from disk are already normalized.
            # This breaks if they are not.

            if self.data_path_train.endswith(".pt"):
                data = torch.load(self.data_path_train).cpu().numpy()
            else:
                data = np.load(self.data_path_train, allow_pickle=True)

            data = torch.tensor(data, device=self.device)

        return train_samples

    def setup_val_set(self):
        val_samples = self.energy.sample((self.val_set_size,))
        return val_samples

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        if self.should_unnormalize:
            samples = self.unnormalize(samples)
        return self.energy.log_prob(samples).squeeze(-1)

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        pass