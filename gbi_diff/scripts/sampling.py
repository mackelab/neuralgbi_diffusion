from pathlib import Path
from omegaconf import DictConfig
import torch
from gbi_diff.sampling.diffusion import DiffusionSampler
from gbi_diff.sampling.mcmc import MCMCSampler
from gbi_diff.sampling.utils import save_torch, get_datetime_str
from gbi_diff.utils.sampling_mcmc_config import Config as MCMCConfig
from gbi_diff.utils.sampling_diffusion_config import Config as DiffusionSamplingConfig
from gbi_diff.utils.train_diffusion_config import Config as DiffusionTrainConfig


# NOTE: does not work yet with multiple worker
"""class SampleThreadManager:   
    def __init__(self, checkpoint: str, config: MCMCConfig, num_worker: int = 1):
        self.num_worker = num_worker
        self.config = config
        # self.potential_func = create_potential_fn(checkpoint, self.config, x_o=None)

        self._data: Dict[int, Dict[str, torch.Tensor]] = {}

    def _sample_x_o(self, idx: int, x_o: torch.Tensor, size: int = 100):
        potential_fn = deepcopy(self.potential_func)
        potential_fn.update_x_o(x_o)

        kernel_cls = getattr(infer, self.config.kernel)
        kernel = kernel_cls(potential_fn=potential_fn)
        mcmc = MCMC(
            kernel,
            num_samples=size,
            warmup_steps=self.config.warmup_steps,
            initial_params={"theta": potential_fn.prior.sample()},
        )
        mcmc.run(x_o)
        print(f"Done: {idx}")

        samples = mcmc.get_samples()
        self._data[idx] = samples

    def reset(self):
        self._data = {}

    def sample(self, x_o: torch.Tensor, size: int = 100) -> Dict[str, torch.Tensor]:
        if len(x_o) == 0:
            raise ValueError("did not receive any observed data")

        self.reset()
        with ThreadPoolExecutor(
            max_workers=self.num_worker,
        ) as executor:
            futures = [
                executor.submit(self._sample_x_o, idx, x, size)
                for idx, x in enumerate(x_o)
            ]
            print(len(futures), futures)

            bar = tqdm(total=len(futures), desc="MCMC sample from for each x_o")
            for future in as_completed(futures):
                bar.n += 1
                bar.refresh()

        # reshape data
        data = OrderedDict(self._data)
        data = list(data.values())
        res = {k: [] for k in data[0].keys()}

        for k in res.keys():
            for sample in data:
                res[k].append(sample[k])
            res[k] = torch.stack(res[k])

        # add observed data
        res["x_o"] = x_o

        return res
"""


def mcmc_sampling(
    checkpoint: str | Path,
    config: MCMCConfig,
    output: str | Path,
    n_samples: int,
    plot: bool,
):
    sampler = MCMCSampler(checkpoint, config)
    samples = sampler.forward(n_samples)
    sampler.save_samples(
        sampler.x_o, samples, output, f"samples_{get_datetime_str()}.pt"
    )
    if plot:
        sampler.pair_plot(samples, sampler.x_o)


def diffusion_sampling(
    diffusion_ckpt: str,
    guidance_ckpt: str,
    config: DiffusionSamplingConfig,
    output: str,
    n_samples: int,
    plot: bool,
) -> torch.Tensor:
    train_config: DiffusionTrainConfig = DiffusionTrainConfig.from_file(
        Path(diffusion_ckpt).parent.joinpath("config.yaml")
    )
    sampler = DiffusionSampler(
        diffusion_ckpt,
        guidance_ckpt,
        observed_data_file=config.observed_data_file,
        gamma=config.beta,
        normalize_data=train_config.dataset.normalize,
    )
    samples = sampler.forward(n_samples)
    sampler.save_samples(
        sampler.x_o, samples, output, f"samples_{get_datetime_str()}.pt"
    )
    if plot:
        sampler.pair_plot(samples, sampler.x_o)
