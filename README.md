# neuralgbi
_Project owners_: Richard Gao & Michael Deistler

Amortized neural Generalized Bayesian Inference for SBI applications: using neural network-based regression and density estimation to do generalized Bayesian inference, i.e., using distance functions as pseudo-likelihood functions.

# Installing dependencies
`pip install -e .` to run setup.
`pip install -e packages/sbi/` to install local version of `sbi`.

# Generating figures

1. Run notebooks in `paper/fig1/01_generate_figure.ipynb`
2. Convert the svg via `invoke convert 1`
3. Upload to overleaf


# Generating benchmark results
1. Make x_o for each task with: 
    ```bash
    python gbi.benchmark.task.generate_xo --task-name <task-name> -n 1000
    ```
2. Generate ground-truth GBI posterior samples from x_os: 
    ```bash
    python -m gbi.benchmark.generate_gt.run_generate_gt -m task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified='specified','misspecified' task.is_known='known','unknown' task.beta=2.,10.,50. task.name=gaussian_mixture
    ```
    This command could run a while.
3. Train algorithms (can be done separately from step 2): `cd gbi/benchmark/run_algorithms/`, `python run_training.py -m task.name=gaussian_mixture algorithm=NPE,NLE,GBI`
4. Do inference with trained algorithms: `cd gbi/benchmark/run_algorithms/`, `python run_inference.py -m algorithm=GBI trained_inference_datetime='$YYYY_MM_DD__hh_mm_ss' task.name='gaussian_mixture' task.xo_index=0,1,2,3,4,5,6,7,8,9 task.is_specified=specified,misspecified task.is_known=known,unknown task.beta=2.,10.,50.`. Note for NPE and NLE there is no need to sweep over `beta`.
