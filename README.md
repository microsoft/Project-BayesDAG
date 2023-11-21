[![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=fff&style=for-the-badge)](https://arxiv.org/abs/2307.13917) 
# BayesDAG: Gradient-Based Posterior Inference for Causal Discovery

This is the code for the NeurIPS 2023 paper ["BayesDAG: Gradient-Based Posterior Inference for Causal Discovery"](https://openreview.net/forum?id=woptnU6fh1). BayesDAG is a fast, scalable structure inference method for causal discovery based on Stochastic-Gradient Markov Chain Monte Carlo (SG-MCMC) and Variational Inference (VI) that is made possible by unconstrained optimization over DAGs through a low-rank node potential vector. The approach is applicable to both linear and nonlinear causal models. 
## Installation
Installation requires [Poetry]((https://python-poetry.org/)) with Anaconda.
### Using Conda
Install the miniconda version corresponding to the python version 3.8 from https://docs.conda.io/en/latest/miniconda.html.

Once in the base environment of conda install poetry following the steps below. If you wish to create a new conda env, you can do so with `conda create -n mypy python=3.8` and use poetry as usual from that environment.

### Poetry

We use Poetry to manage the project dependencies, they're specified in the [pyproject.toml](pyproject.toml). To install poetry run:

```
    curl -sSL https://install.python-poetry.org | python3 -
```

To install the environment run `poetry install`, this will create a virtualenv that you can use by running either `poetry shell` or `poetry run {command}`. It's also a virtualenv that you can interact with in the normal way too.

Poetry also uses a lock file that exactly specifies all sub-dependencies. If you update the project dependencies, you can run either `poetry install` or `poetry update` to install the new dependencies, this will also modify the lockfile. To just modify the lockfile you can run `poetry lock`. This file must be committed to version control.

More information about poetry can be found [here](https://python-poetry.org/)

### Generating Synthetic Data
In order to generate synthetic data for ER and SF graphs for nonlinear case, please use the following command:

```
python -m open_source.causica.data_generation.large_synthetic.data_generation --num_nodes <num_nodes> --sem_type mlp --noise_type unequal --dataset_folder <dataset_folder>
```

For experiments for linear SCMs of size 5, please use the following command:

```
python -m open_source.causica.data_generation.large_synthetic.data_generation --num_nodes 5 --sem_type linear --noise_type unequal --dataset_folder <dataset_folder> --expected_edges_per_node 1 --num_samples_train 500 --num_samples_test 100
```

This command generates 30 datasets of both ER and SF graphs. 

### Running the Experiments

In order to run experiments for a single dataset (say ER 30 dataset with random seed 10) with nonlinear BayesDAG, run the following command:

```
python run_experiment.py run_ER_30_60_mlp_sem_unequal_noise_10_seed\
  --model_type bayesdag_nonlinear --model_config open_source/configs/bayesdag/bayesdag_nonlinear_er_30_60.json \
  --causal_discovery --device 0 --output_dir <results_dir> --data_dir <dataset_folder>
```

For running the experiments with linear BayesDAG, run the following command:

```
python run_experiment.py run_ER_5_5_mlp_sem_unequal_noise_10_seed\
  --model_type bayesdag_linear --model_config open_source/configs/bayesdag/bayesdag_nonlinear_er_30_60.json \
  --causal_discovery --device 0 --output_dir <results_dir> --data_dir <dataset_folder>
```


Inside `open_source/configs/bayesdag/`, there are config files for each setting which contain hyperparameters that have been tuned with some held-out dataset. See Appendix D for details. Use the corresponding config files for the datasets you are running.

## Running on Custom Data
If you have some custom data, which aloows for holding-out around 20% of the data, then some of thee hyperparameters are best tuned on this held-out set. In order to get an idea of the most important set of hyperparameters to tune, take a look at `open_source/configs/bayesdag/baysedag_linear.json` and `open_source/configs/bayesdag/bayesdag_nonlinear.json` for linear and nonlinear models respectively. You can directly use them in the `--model_config` option to do a hyperparameter search.

