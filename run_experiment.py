"""
Wraps open_source/common/run_experiment.py to enable running experiments in AML.
Check open_source/common/run_experiment.py for details what arguments to use to run an expeirment.
"""
import argparse
import os
import sys
from typing import List

from evaluation_pipeline.aml_run_context import setup_run_context_in_aml
from open_source.causica.run_experiment import get_parser, run_experiment_on_parsed_args, validate_args


def get_parser_with_aml_args() -> argparse.ArgumentParser:
    parser = get_parser()
    # Add AML arguments
    parser.add_argument(
        "--filter_metrics",
        nargs="*",
        help="If specified, only these metrics will be logged to AzureML.",
    )
    parser.add_argument(
        "--aml_experiment_name",
        type=str,
        help="Experiment name to use in AML workspace for non-local runs. Runs locally if not specified.",
    )
    parser.add_argument(
        "--compute_target",
        type=str,
        default="gpu-experiment-cluster",
        help="Name of AML compute target to use for remote runs.",
    )
    parser.add_argument(
        "--aml_config_filename",
        type=str,
        default="causica_config.json",
        help="Name of AML config file to use in './evaluation_pipeline/configs/'.",
    )
    return parser


def get_args(user_args: List[str]) -> argparse.Namespace:
    """
    Parses command line arguments and validates them.
    Returns: namespace of command line args.
    """
    parser = get_parser_with_aml_args()
    args = parser.parse_args(user_args)
    validate_args(args)  # Note lack of AML specific validation
    return args


def main(user_args):

    args = get_args(user_args)
    # Overwrite default_configs_dir to handle different relative path
    # depending whether we run from run_experiment.py or open_source/common/run_experiment.py
    # TODO: find cleaner way of doing it
    args.default_configs_dir = os.path.join("open_source", "configs")
    # Prepare AML context
    run_context = setup_run_context_in_aml(
        args.aml_experiment_name,
        compute_target=args.compute_target,
        aml_config_filename=args.aml_config_filename,
    )

    run_experiment_on_parsed_args(args=args, run_context=run_context)


if __name__ == "__main__":
    main(sys.argv[1:])
