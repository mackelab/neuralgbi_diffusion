import pickle
import time

import numpy as np
import torch
from tqdm import tqdm

import gbi_diff.dataset.simulators.hh.utils as utils
from gbi_diff.dataset.simulators.hh.HodgkinHuxley import HodgkinHuxley
from gbi_diff.dataset.simulators.hh.HodgkinHuxleyStatsMoments import (
    HodgkinHuxleyStatsMoments,
)

seed = 1
n_xcorr = 0
n_mom = 4
n_summary = 7


true_params, labels_params = utils.obs_params(reduced_model=False)
I, t_on, t_off, dt = utils.syn_current()
obs = utils.syn_obs_data(I, dt, true_params, seed=seed, cython=True)
obs_stats = utils.syn_obs_stats(
    data=obs,
    I=I,
    t_on=t_on,
    t_off=t_off,
    dt=dt,
    params=true_params,
    seed=seed,
    n_xcorr=n_xcorr,
    n_mom=n_mom,
    cython=True,
    n_summary=n_summary,
)

with open(f"data/gt.pkl", "wb") as handle:
    pickle.dump(true_params, handle)

with open(f"data/xo.pkl", "wb") as handle:
    pickle.dump(obs_stats, handle)


p = utils.prior(
    true_params=true_params,
    prior_uniform=True,
    prior_extent=True,
    prior_log=False,
    seed=seed,
)


sim = HodgkinHuxley(
    I, dt, V0=obs["data"][0], reduced_model=False, cython=True, prior_log=False
)
stats = HodgkinHuxleyStatsMoments(
    t_on=t_on, t_off=t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary
)

n_runs = 100
print("Starting sims")

for run_ind in range(n_runs):
    n_sim = 10_000
    start_seed = n_sim * run_ind + 1
    end_seed = n_sim * (run_ind + 1) + 1
    seeds = np.arange(start_seed, end_seed, 1)
    _ = torch.manual_seed(start_seed)
    thetas = p.sample((n_sim,))

    start_time = time.time()
    r = []
    for theta, seed in tqdm(zip(thetas, seeds), total=len(thetas)):
        r.append(sim.gen_single(theta, seed=seed    ))      
    ss = stats.calc(r)
    print("time", time.time() - start_time)

    with open(f"data/theta_{run_ind}.pkl", "wb") as handle:
        pickle.dump(thetas, handle)

    with open(f"data/summstats_{run_ind}.pkl", "wb") as handle:
        pickle.dump(ss, handle)
    break
