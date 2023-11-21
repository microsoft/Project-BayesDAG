import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import psutil
import torch

from ..experiment.steps.eval_step import (
    eval_causal_discovery,
)
from ..experiment.steps.step_func import load_data, preprocess_configs
from ..experiment.steps.train_step import run_train_main
from ..models_factory import load_model
from ..utils.io_utils import save_json, save_txt
from .run_context import RunContext


class SystemMetricsLogger:
    def __init__(self):
        self._thread = None
        self._keep_running = True
        self._peak_cpu_memory_in_mb = 0.0
        self._peak_cpu_percent = 0.0

    def start_log(self):
        self._thread = threading.Thread(target=self._run, args=())
        # allows the main application to exit even though the thread is running
        self._thread.daemon = True
        self._thread.start()

    def _run(self):
        process = psutil.Process(os.getpid())
        while self._keep_running:
            memory_in_mb = process.memory_info().rss // (10**6)
            cpu_percent = psutil.cpu_percent()
            if memory_in_mb > self._peak_cpu_memory_in_mb:
                self._peak_cpu_memory_in_mb = memory_in_mb
            if cpu_percent > self._peak_cpu_percent:
                self._peak_cpu_percent = cpu_percent
            time.sleep(1)

    def end_log(self):
        if self._thread is not None:
            self._keep_running = False
            self._thread.join()
            return {"peak_cpu_memory_in_mb": self._peak_cpu_memory_in_mb, "peak_cpu_percent": self._peak_cpu_percent}


@dataclass
class ExperimentArguments:
    dataset_name: str
    data_dir: str
    model_type: str
    model_dir: str
    model_id: str
    run_inference: bool
    extra_eval: bool
    active_learning: Optional[List[str]]
    max_steps: int
    max_al_rows: int
    causal_discovery: bool
    latent_confounded_causal_discovery: bool
    treatment_effects: bool
    device: str
    quiet: bool
    active_learning_users_to_plot: List[int]
    tiny: bool
    dataset_config: Dict[str, Any]
    dataset_seed: Union[int, Tuple[int, int]]
    model_config: Dict[str, Any]
    train_hypers: Dict[str, Any]
    output_dir: str
    experiment_name: str
    model_seed: int
    aml_tags: Dict[str, Any]
    logger_level: str
    run_context: RunContext
    eval_likelihood: bool = True
    conversion_type: str = "full_time"


def run_single_seed_experiment(args: ExperimentArguments):
    # Set up loggers
    logger = logging.getLogger()
    log_format = "%(asctime)s %(filename)s:%(lineno)d[%(levelname)s]%(message)s"
    if args.quiet:
        level = logging.ERROR
    else:
        level_dict = {
            "ERROR": logging.ERROR,
            "INFO": logging.INFO,
            "CRITICAL": logging.CRITICAL,
            "WARNING": logging.WARNING,
            "DEBUG": logging.DEBUG,
        }
        level = level_dict[args.logger_level]
    logging.basicConfig(level=level, force=True, format=log_format)
    mlflow.set_tags(args.aml_tags)
    running_times: Dict[str, float] = {}

    _clean_partial_results_in_aml_run(args.output_dir, logger, args.run_context)

    # Log system's metrics
    system_metrics_logger = SystemMetricsLogger()
    system_metrics_logger.start_log()

    # Load data
    logger.info("Loading data.")
    dataset = load_data(
        args.dataset_name,
        args.data_dir,
        args.dataset_seed,
        args.dataset_config,
        args.model_config,
        args.tiny,
        args.run_context.download_dataset,
    )
    assert dataset.variables is not None

    # Preprocess configs based on args and dataset
    preprocess_configs(args.model_config, args.train_hypers, args.model_type, dataset, args.data_dir, args.tiny)

    # Loading/training model
    if args.model_id is not None:
        logger.info("Loading pretrained model")
        model = load_model(args.model_id, args.model_dir, args.device)
    else:
        start_time = time.time()
        model = run_train_main(
            logger=logger,
            model_type=args.model_type,
            output_dir=args.output_dir,
            variables=dataset.variables,
            dataset=dataset,
            device=args.device,
            model_config=args.model_config,
            train_hypers=args.train_hypers,
        )
        running_times["train/running-time"] = (time.time() - start_time) / 60
    save_json(args.dataset_config, os.path.join(model.save_dir, "dataset_config.json"))
    save_txt(args.dataset_name, os.path.join(model.save_dir, "dataset_name.txt"))
    eval_causal_discovery(dataset, model)
    try:
        model.load_state_dict(torch.load(os.path.join(model.save_dir, model.best_model_file)))
        eval_causal_discovery(dataset, model, best = True)
    except:
        print("ignoring best model evaluation")

    # Log speed/system metrics
    system_metrics = system_metrics_logger.end_log()
    mlflow.log_metrics(system_metrics)
    save_json(system_metrics, os.path.join(model.save_dir, "system_metrics.json"))
    mlflow.log_metrics(running_times)
    save_json(running_times, os.path.join(model.save_dir, "running_times.json"))

    _copy_results_in_aml_run(args.output_dir, args.run_context)

    return model, args.model_config


def _clean_partial_results_in_aml_run(output_dir: str, logger: logging.Logger, run_context: RunContext):
    if run_context.is_azureml_run():
        # If node is preempted (e.g. long running exp), it's possible
        # that there will be some partial results created in output directory
        # Those partial results shouldn't be aggregated, thus remove them
        logger.info("Checking if partial outputs are present for the run (if AML node was preempted before).")
        if os.path.isdir(output_dir):
            logger.info("Partial results are present.")
            for folder in os.listdir(output_dir):
                path = os.path.join(output_dir, folder)
                assert os.path.exists(path), "Partial results do not exist"
                if os.path.isdir(path):
                    logger.info(f"Removing partial results' directory: {path}.")
                    shutil.rmtree(path)
                else:
                    logger.info(f"Removing partial results' file: {path}.")
                    os.remove(path)


def _copy_results_in_aml_run(output_dir: str, run_context: RunContext):
    if run_context.is_azureml_run():
        # Copy the results to 'outputs' dir so that we can easily view them in AzureML.
        # Workaround for port name collision issue in AzureML, which sometimes prevents us from setting outputs_dir='outputs'.
        # See #16728
        shutil.copytree(output_dir, "outputs", dirs_exist_ok=True)
