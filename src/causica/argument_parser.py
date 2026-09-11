import argparse
import os
import textwrap

import numpy as np


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a partial VAE model.", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("dataset_name", help="Name of dataset to use.")
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default="data",
        help="Directory containing saved datasets. Defaults to ./data",
    )
    parser.add_argument(
        "--model_type",
        "-mt",
        type=str,
        default="bayesdag_nonlinear",
        choices=[
            "bayesdag_linear",
            "bayesdag_nonlinear",
        ],
        help=textwrap.dedent(
            """Type of model to train.
            """
        ),
    )
    parser.add_argument("--model_dir", "-md", default=None, help="Directory containing the model.")
    parser.add_argument("--model_config", "-m", type=str, help="Path to JSON containing model configuration.")
    parser.add_argument("--dataset_config", "-dc", type=str, default = "src/configs/dataset_config_causal_dataset.json", help="Path to JSON containing dataset configuration.")
    parser.add_argument("--impute_config", "-ic", type=str, help="Path to JSON containing impute configuration.")
    parser.add_argument("--objective_config", "-oc", type=str, help="Path to JSON containing objective configuration.")

    # Whether or not to run inference and active learning
    parser.add_argument("--run_inference", "-i", action="store_true", help="Run inference after training.")
    parser.add_argument("--extra_eval", "-x", action="store_true", help="Run extra eval tests that take longer.")
    parser.add_argument(
        "--max_steps", "-ms", type=int, default=np.inf, help="Maximum number of active learning steps to take."
    )
    parser.add_argument(
        "--max_al_rows",
        "-mar",
        type=int,
        default=np.inf,
        help="Maximum number of rows on which to perform active learning.",
    )
    parser.add_argument(
        "--active-learning",
        "-a",
        nargs="+",
        choices=[
            "eddi",
            "eddi_mc",
            "eddi_rowwise",
            "rand",
            "cond_sing",
            "sing",
            "ei",
            "b_ei",
            "variance",
            "all",
        ],
        help="""Run active learning after train and test.
                                eddi = personalized information acquisition from EDDI paper
                                eddi_mc = personalized information acquisition using Bayesian EDDI with stocastic weights
                                eddi_rowwise = same as eddi but with row-wise parallelization for information gain computation
                                rand = random strategy for information acquisition
                                cond_sing = conditional single order strategy across the whole test dataset where the next best step condition on existing observation
                                sing = single order strategy determinated by first step information gain
                                ei = expected improvement,
                                b_ei = batch ei""",
    )
    parser.add_argument(
        "--users_to_plot", "-up", default=[0], nargs="+", help="Indices of users to plot info gain bar charts for."
    )
    # Whether or not to evaluate causal discovery (only visl at the moment)
    parser.add_argument(
        "--causal_discovery",
        "-c",
        action="store_true",
        help="Whether to evaluate causal discovery against a ground truth during evaluation.",
    )
    parser.add_argument(
        "--latent_confounded_causal_discovery",
        "-lcc",
        action="store_true",
        help="Whether to evaluate latent confounded causal discovery against a ground truth during evaluation.",
    )
    parser.add_argument(
        "--treatment_effects",
        "-te",
        action="store_true",
        help="Whether to evaluate treatment effects against a ground truth.",
    )
    # Other options for saving output.

    parser.add_argument("--output_dir", "-o", type=str, default="runs", help="Output path. Defaults to ./runs/.")
    parser.add_argument("--name", "-n", type=str, help="Tag for this run. Output dir will start with this tag.")
    parser.add_argument(
        "--device", "-dv", default="cpu", help="Name (e.g. 'cpu', 'gpu') or ID (e.g. 0 or 1) of device to use."
    )
    parser.add_argument("--tiny", action="store_true", help="Use this flag to do a tiny run for debugging")
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for training. If not provided, a random seed will be taken from the model config JSON",
    )
    parser.add_argument(
        "--default_configs_dir",
        "-dcd",
        type=str,
        default="configs",
        help="Directory containing configs. Defaults to ./configs",
    )
    # Control the logger level
    parser.add_argument(
        "--logger_level",
        "-ll",
        type=str.upper,
        default="INFO",
        choices=["CRITICAL", "ERROR", "INFO", "DEBUG", "WARNING"],
        help="Control the logger level. Default: %(default)s .",
    )
    # Control whether evaluating the log-likelihood for Causal models
    parser.add_argument(
        "--eval_likelihood",
        "-el",
        action="store_false",
        help="Disable the likelihood computation for causal models during treatment effect estimation.",
    )
    parser.add_argument(
        "--conversion_type",
        "-ct",
        type=str.lower,
        default="full_time",
        choices=["ful_time", "auto_regressive"],
        help="The type of conversion used for converting the temporal adjacency matrix to a static adjacency matrix during causal discovery evaluation",
    )
    parser.add_argument(
        "--scale_noise",
        "-sn",
        type=float,
        help="Hyperparameter of the Adam SGMCMC",
    )
    parser.add_argument(
        "--scale_noise_p",
        "-snp",
        type=float,
        help="Hyperparameter of the Adam SGMCMC for p",
    )
    parser.add_argument(
        "--lambda_sparse",
        "-ls",
        type=float,
        help="Hyperparameter of the Sparsity Hyperparameter",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        help="Base Learning rate",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.

    """
    if not os.path.isdir(args.data_dir):
        print(f"{args.data_dir} is not a directory or does not exists, creating it.")
        os.makedirs(args.data_dir)

    # Config files
    for config in (args.model_config, args.dataset_config, args.impute_config, args.objective_config):
        if config is not None and not os.path.isfile(config):
            raise ValueError(f"Config file {config} does not exist.")
