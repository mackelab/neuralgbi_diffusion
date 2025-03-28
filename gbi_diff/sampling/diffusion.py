from typing      import Tuple
from pathlib import Path

import h5py
import numpy as np
from torch import Tensor
import torch
from tqdm import tqdm

from gbi_diff.model.lit_module import DiffusionModel, Guidance
from gbi_diff.sampling.sampler import _PosteriorSampler
from gbi_diff.sampling.utils import get_sample_path, load_data_stats, load_observed_data
from gbi_diff.utils.plot import _pair_plot
from gbi_diff.utils.train_diffusion_config import Config as DiffusionConfig
from gbi_diff.utils.train_guidance_config import Config as GuidanceConfig


class DiffusionSampler(_PosteriorSampler):
    def __init__(
        self,
        diff_model_ckpt: str | Path,
        guidance_model_ckpt: str | Path,
        observed_data_file: str | Path,
        beta: float = 1,
        normalize_data: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(diff_model_ckpt, str):
            diff_model_ckpt = Path(diff_model_ckpt)
        self.diff_model_ckpt = diff_model_ckpt
        if isinstance(guidance_model_ckpt, str):
            guidance_model_ckpt = Path(guidance_model_ckpt)
        self.guidance_model_ckpt = guidance_model_ckpt

        self._observed_data_file = observed_data_file
        self._beta = beta
        self._normalize_data = normalize_data
        self._config = {
            "observed_data_file": self._observed_data_file,
            "beta": self._beta,
            "normalize_data": self._normalize_data
        }

        self._check_model_compatibility(diff_model_ckpt, guidance_model_ckpt)
        self._check_config(self._observed_data_file, guidance_model_ckpt)

        self._guidance_model = Guidance.load_from_checkpoint(guidance_model_ckpt)
        self._diff_model = DiffusionModel.load_from_checkpoint(diff_model_ckpt)
        self.x_o, self.theta_o = load_observed_data(self._observed_data_file)
        self._data_stats = load_data_stats(guidance_model_ckpt.parent.joinpath("data_stats.pt"))
        self._diff_beta_schedule = self._diff_model.diff_schedule.beta_schedule
    
    def _check_config(self, observed_data: str | Path, guidance_model_ckpt: Path):
        # check if dataset is compatible with guidance and therefor with diffusion model
        guidance_model_config: GuidanceConfig = GuidanceConfig.from_file(
            guidance_model_ckpt.parent / "config.yaml"
        )
        guidance_train_set_name = Path(guidance_model_config.dataset.train_file)
        guidance_train_set_name = "_".join(guidance_train_set_name.stem.split("_")[:-1])

        # observed_set_name = observed_data.rstrip("10.pt")
        observed_data = Path(observed_data)
        observed_set_name = "_".join(observed_data.stem.split("_")[:-1])
        msg = f"Observed dataset has to be compatible with training sets but: {observed_set_name=}, {guidance_train_set_name=}"
        print(guidance_train_set_name, observed_set_name)
        assert (
            guidance_train_set_name.split("/")[-1] == observed_set_name.split("/")[-1]
        ), msg

    def _check_model_compatibility(
        self, diff_model_ckpt: Path, guidance_model_ckpt: Path
    ):
        # load configs for both checkpoints
        diff_model_config: DiffusionConfig = DiffusionConfig.from_file(
            diff_model_ckpt.parent / "config.yaml"
        )
        guidance_model_config: GuidanceConfig = GuidanceConfig.from_file(
            guidance_model_ckpt.parent / "config.yaml"
        )

        # check if dataset is the same
        diff_train_set_name = diff_model_config.dataset.train_file
        diff_train_set_name = diff_train_set_name.rstrip("10.pt")
        guidance_train_set_name = guidance_model_config.dataset.train_file
        guidance_train_set_name = guidance_train_set_name.rstrip("10.pt")
        msg = f"Guidance and diffusion model have to be trained on the same datafile. Given: {diff_model_config.dataset.train_file=}, {guidance_model_config.dataset.train_file}"
        assert diff_train_set_name == guidance_train_set_name, msg

        # check if diffusion steps are the same
        diff_steps = diff_model_config.diffusion.steps
        guidance_steps = guidance_model_config.diffusion.steps
        msg = f"Both models have to be trained on the same number of diffusion steps. Given: {diff_steps=} {guidance_steps=}"
        assert diff_steps == guidance_steps, msg

        # check iff diffusion schedule is the same
        diff_schedule = getattr(
            diff_model_config.diffusion, diff_model_config.diffusion.diffusion_schedule
        ).to_container()
        guidance_schedule = getattr(
            guidance_model_config.diffusion,
            guidance_model_config.diffusion.diffusion_schedule,
        ).to_container()
        msg = f"Diffusion Schedules have to be the same. Given: {diff_schedule=}, {guidance_schedule=}"
        assert diff_schedule == guidance_schedule, msg
        # check if time representation dimension is the same
        diff_time_repr_dim = diff_model_config.model.TimeEncoder.input_dim
        guidance_time_repr_dim = guidance_model_config.model.TimeEncoder.input_dim
        msg = f"Both, diffusion and guidance have to use the same time encoding and representation dimension. Given {diff_time_repr_dim=}, {guidance_time_repr_dim=}"
        assert diff_time_repr_dim == guidance_time_repr_dim, msg

    def _get_default_path(self):
        return get_sample_path(self.diff_model_ckpt)

    def single_forward(
        self, x_o: Tensor, n_samples: int, quiet: bool = False
    ) -> Tensor:
        """_summary_

        Args:
            x_o (Tensor): (n_simulation_features, )
            n_samples (int): how many samples you would like to create per

        Returns:
            Tensor: (n_samples, theta_dim)
        """
        # n_samples per sample in the dataset
        theta_t = torch.normal(
            0, 1, size=(n_samples, self.theta_dim), requires_grad=True
        )

        T = self._diff_model.diffusion_steps
        T = np.arange(T)[::-1]

        if self._normalize_data:
            _, (x_mean, x_std) = self._data_stats
            x_o = (x_o - x_mean) / x_std
        
        iterator = T
        if not quiet:
            iterator = tqdm(T, desc="Step in diffusion process", leave=True)

        for t_idx in iterator:
            beta = self._diff_beta_schedule.forward(t_idx)
            alpha = self._diff_beta_schedule.get_alphas(t_idx)
            alpha_bar = self._diff_beta_schedule.get_alpha_bar(t_idx)

            time_repr = self._diff_model.get_diff_time_repr(np.array([t_idx]))
            time_repr = time_repr.repeat(n_samples, 1)

            diffusion_step = self._diff_model.forward(theta_t, time_repr)
            diffusion_step = diffusion_step.detach()
            guidance_grad = self.get_log_boltzmann_grad(theta_t, x_o, time_repr)
            guidance_grad = guidance_grad.detach()

            z = torch.normal(0, 1, size=theta_t.shape)
            z = torch.sqrt(beta) * z if t_idx > 0 else 0

            theta_t = (
                (1 / torch.sqrt(alpha))
                * (theta_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * diffusion_step
                + beta      * guidance_grad)
                + z
            )

            del diffusion_step
            del guidance_grad
            del time_repr

        if self._normalize_data:
            (theta_mean, theta_std), _ = self._data_stats
            theta_t = theta_t * theta_std + theta_mean
        return theta_t

    def get_log_boltzmann_grad(
        self, theta: Tensor, x_target: Tensor, time_repr: Tensor
    ) -> Tensor:
        """
        get the gradient of the log prob of a Boltzmann statistics
        Boltzmann statistics:
            $p_\psi(x) = \exp(-\beta E_\psi(x)) / Z(\psi)$
            $\nabla_x \log(p_\psi(x)) = -\beta \nabla_x E_\psi(x)$

        Note: batchsize = n_samples. n_samples is the number of samples we sample per given observation we put in.
        Args:
            theta (Tensor): theta tensor for guidance model (batch_size / n_samples, n_features)
            x_target (Tensor): guidance signal, observed data (n_target, n_sim_features)
            time_repr (Tensor): time representation (batch_size, time_features)

        Returns:
            Tensor: log gradient (batch_size, n_theta_features)
        """
        if len(x_target.shape) == 1:
            x_target = x_target[None]

        batch_size = len(theta)
        x_target = x_target[None].repeat(batch_size, 1, 1)

        theta = theta.detach()
        theta.requires_grad = True
        x_target = x_target.detach()
        x_target.requires_grad = True
        time_repr = time_repr.detach()
        time_repr.requires_grad = True

        log_prob = self._guidance_model.forward(theta, x_target, time_repr)
        grad = torch.autograd.grad(outputs=log_prob.sum(), inputs=theta)[0]
        grad = -self.beta * grad
        theta = theta.detach()
        return grad

    def forward(self, n_samples: int, quiet: int = 0, h5_file: Tuple[h5py.File, slice] = None) -> Tensor | h5py.File:
        """_summary_

        Args:
            n_samples (int): _description_
            quiet (optional, bool): 0 no progress bar at all. 1: only the upper progress bar. 2. all progress bars
            h5_file (optional, Tuple[h5py.File, slice]): is a combination of h5 file which has to contain the dataset 
                "theta" in the shape such that it fits the gained samples 
        Returns:
            Tensor: (n_samples, n_observed_data, param_dim) | h5py.File
        """

        if h5_file is None:
            res = torch.zeros((1, n_samples, len(self.x_o), self.theta_dim))
            s = slice(0, 1)
        else:
            file, s = h5_file
            res = file["theta"]

        iterator = self.x_o
        if not (quiet >= 2):
            iterator = tqdm(self.x_o, desc="Sample in observed data")

        for idx, x_o in enumerate(iterator):
            samples = self.single_forward(x_o, n_samples, quiet < 2).detach().cpu()
            if h5_file is not None:
                samples = samples.numpy().copy()
            res[s, :, idx] = samples

        if h5_file is None:
            # remove artificial dimension and return stacked tensor
            return res[0]
        return file

    def pair_plot(self, samples: Tensor, x_o: Tensor, output: str | Path = None):
        """_summary_

        Args:
            samples (Tensor): (batch_size, n_target, theta_dim)
            x_o (Tensor): observed data (n_target, sim_out_dim)
            output (str | Path): where to store the figures

        """
        batch_size, n_target, _ = samples.shape
        time_repr = self._guidance_model.get_diff_time_repr(np.zeros(batch_size))
        x_o = x_o[None].repeat(batch_size, 1, 1)

        if output is None:
            save_dir = self._get_default_path()
        elif isinstance(output, Path):
            save_dir = output
        else:
            save_dir = Path(output)

        for target_idx in range(n_target):
            log_prob = -self.beta * self._guidance_model.forward(
                samples[:, target_idx], x_o[:, [target_idx]], time_repr
            )
            sample = samples[:, target_idx]
            title = f"Index: {target_idx}, beta: {self.beta}"
            file_name = f"pair_plot_{target_idx}_beta_{self.beta}.png"
            save_path = save_dir / file_name
            _pair_plot(
                sample, torch.exp(log_prob), title=title, save_path=str(save_path)
            )

    def update_beta(self, value: int | float):
        assert isinstance(value, (int, float))
        self._beta = value

    @property
    def theta_dim(self) -> int:
        return self._guidance_model._net._theta_encoder._input_dim

    @property
    def beta(self) -> float:
        return self._beta
