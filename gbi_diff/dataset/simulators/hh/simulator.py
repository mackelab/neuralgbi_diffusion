import numpy as np
from torch import Tensor
from sbi.utils import BoxUniform
import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from gbi_diff.dataset.simulators.hh import HodgkinHuxley, HodgkinHuxleyStatsMoments
from gbi_diff.dataset.simulators.hh.HodgkinHuxley import param_transform
from gbi_diff.dataset.simulators.hh.utils import (
    allen_obs_data,
    allen_obs_stats,
    obs_params,
    syn_current,
    syn_obs_data,
    syn_obs_stats,
)


class HodgkinHuxleySimulator:
    def __init__(
        self,
        allen: bool = False,
        prior_uniform: bool = True,
        prior_extent: bool = False,
        prior_log: bool = False,
        seed: int = None,
    ):
        self.allen = allen
        self.prior_uniform = prior_uniform
        self.prior_extent = prior_extent
        self.prior_log = prior_log
        self.seed = seed

        self.n_xcorr = 0
        self.n_mom = 4
        self.n_summary = 7

        self.true_params, _ = obs_params(reduced_model=False)
        self.prior = self.get_prior(self.true_params)

        I, dt, t_on, t_off, obs, _ = self.get_observational_data(self.allen)
        self.sim = HodgkinHuxley(
            I,
            dt,
            V0=obs["data"][0],
            reduced_model=False,
            cython=True,
            prior_log=False,
        )
        self.stats = HodgkinHuxleyStatsMoments(
            t_on=t_on,
            t_off=t_off,
            n_xcorr=self.n_xcorr,
            n_mom=self.n_mom,
            n_summary=self.n_summary,
        )

    def get_observational_data(self, allen: bool):
        if allen:
            ephys_cell = 518290966
            sweep_number = 57
            A_soma = 0.0234 / 126
            junction_potential = -14

            obs = allen_obs_data(
                ephys_cell=ephys_cell, sweep_number=sweep_number, A_soma=A_soma
            )

            obs["data"] = obs["data"] + junction_potential
            I = obs["I"]
            dt = obs["dt"]
            t_on = obs["t_on"]
            t_off = obs["t_off"]

            obs_stats = allen_obs_stats(
                data=obs,
                ephys_cell=ephys_cell,
                sweep_number=sweep_number,
                n_xcorr=self.n_xcorr,
                n_mom=self.n_mom,
                n_summary=self.n_summary,
            )
        else:
            I, t_on, t_off, dt = syn_current()
            obs = syn_obs_data(I, dt, self.true_params, seed=self.seed, cython=True)

            obs_stats = syn_obs_stats(
                data=obs,
                I=I,
                t_on=t_on,
                t_off=t_off,
                dt=dt,
                params=self.true_params,
                seed=self.seed,
                n_xcorr=self.n_xcorr,
                n_mom=self.n_mom,
                cython=True,
                n_summary=self.n_summary,
            )

        return I, dt, t_on, t_off, obs, obs_stats

    def get_prior(self, true_params):
        if not self.prior_extent:
            range_lower = param_transform(self.prior_log, 0.5 * true_params)
            range_upper = param_transform(self.prior_log, 1.5 * true_params)
        else:
            range_lower = param_transform(
                self.prior_log,
                np.array([0.5, 1e-4, 1e-4, 1e-4, 50.0, 40.0, 1e-4, 35.0]),
            )
            range_upper = param_transform(
                self.prior_log,
                np.array([80.0, 15.0, 0.6, 0.6, 3000.0, 90.0, 0.15, 100.0]),
            )

            range_lower = range_lower[0 : len(true_params)]
            range_upper = range_upper[0 : len(true_params)]

        if self.prior_uniform:
            prior_min = range_lower
            prior_max = range_upper
            return BoxUniform(prior_min, prior_max)
        else:
            prior_mn = param_transform(self.prior_log, true_params)
            prior_cov = np.diag((range_upper - range_lower) ** 2) / 12
            return MultivariateNormal(prior_mn, prior_cov)

    def simulate(self, theta: Tensor, seeds: np.ndarray = None) -> Tensor:
        if seeds is None:
            seeds = np.ones(len(theta)) * self.seed
        else:
            assert len(theta) == len(seeds), "Every theta needs one dedicated seed"
        
        r = []
        for param, seed in tqdm(zip(theta, seeds), total=len(theta)):
            r.append(self.sim.gen_single(theta, seed=seed))    
        ss: np.ndarray = self.stats.calc(r)
        ss = torch.from_numpy(ss)
        return ss       
    
    
        
        