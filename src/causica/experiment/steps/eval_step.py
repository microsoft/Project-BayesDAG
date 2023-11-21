"""
Run evaluation on a trained PVAE.

To run: python run_eval.py boston -ic parameters/impute_config.json -md runs/run_name/models/model_id
"""

import os

import mlflow

from ...datasets.dataset import (
    CausalDataset,
)
from ...models.imodel import IModelForInterventions
from ...utils.metrics import save_train_val_test_metrics

ALL_INTRVS = "all interventions"


def eval_causal_discovery(
    dataset: CausalDataset,
    model: IModelForInterventions,
    best: bool = False
):
    """
    Args:
        logger (`logging.Logger`): Instance of logger class to use.
        dataset: Dataset or SparseDataset object.
        model (IModelForInterventions): Model to use.
        conversion_type (str): It is used for temporal causal model evaluation. It supports "full_time" and "auto_regressive".
        "full_time" converts temporal adj matrix to full-time static graph and "auto_regressive" converts it to static graph only keeping
        the connections to the current timestep. For details, refer to docstring of `convert_temporal_adj_matrix_to_static`.

    This requires the model to have a method get_adjacency_data_matrix() implemented, which returns the adjacency
    matrix learnt from the data.
    """
    results = model.evaluate_metrics(dataset)
    # Log metrics
    mlflow.log_metric(f"orientation.f1_{best}", results["orientation_fscore"], True)
    mlflow.log_metric(f"causalshd_{best}", results["shd"])
    mlflow.log_metric(f"causalnnz_{best}", results["nnz"])
    mlflow.log_metric(f"nll_held_out_{best}", results["nll_val"])
    mlflow.log_metric(f"nll_train_{best}", results["nll_train"])

    if best:
        save_path =  os.path.join(model.save_dir, "target_results_causality_best.json")
    else:
        save_path = os.path.join(model.save_dir, "target_results_causality.json")
    # Save causality results to a file
    save_train_val_test_metrics(
        train_metrics={},
        val_metrics={},
        test_metrics=results,
        save_file=save_path,
    )
