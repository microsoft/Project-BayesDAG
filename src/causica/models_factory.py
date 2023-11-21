import os
from typing import Any, Dict, Type, Union
from uuid import uuid4


from .datasets.variables import Variables
from .models.bayesdag.bayesdag_linear import BayesDAGLinear
from .models.bayesdag.bayesdag_nonlinear import BayesDAGNonLinear
from .models.imodel import IModel

MODEL_SUBCLASSES: Dict[str, Type[IModel]] = {
    model.name(): model  # type: ignore
    for model in (
        # Models
        BayesDAGLinear,
        BayesDAGNonLinear,
    )
}


class ModelClassNotFound(NotImplementedError):
    pass


def create_model(
    model_name: str,
    models_dir: str,
    variables: Variables,
    device: Union[str, int],
    model_config_dict: Dict[str, Any],
    model_id: str = None,
) -> IModel:
    """
    Get an instance of an implementation of the `Model` class.

    Args:
        model_name (str): String corresponding to concrete instance of `Model` class.
        models_dir (str): Directory to save model information in.
        variables (Variables): Information about variables/features used
                by this model.
        model_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
            the form {arg_name: arg_value}. e.g. {"sample_count": 10}
        device (str or int): Name of Torch device to create the model on. Valid options are 'cpu', 'gpu', or a device ID
            (e.g. 0 or 1 on a two-GPU machine).
        model_id (str): String specifying GUID for model. A GUID will be generated if not provided.

    Returns:
        Instance of concrete implementation of `Model` class.
    """
    # Create anything needed for all model types.
    model_id = model_id if model_id is not None else str(uuid4())
    save_dir = os.path.join(models_dir, model_id)
    os.makedirs(save_dir)

    try:
        return MODEL_SUBCLASSES[model_name].create(model_id, save_dir, variables, model_config_dict, device=device)
    except KeyError as e:
        raise ModelClassNotFound() from e


def load_model(model_id: str, models_dir: str, device: Union[str, int]) -> IModel:
    """
    Loads an instance of an implementation of the `Model` class.

    Args:
        model_id (str): String corresponding to model's id.
        models_dir (str): Directory where mnodel information is saved.

    Returns:
        Deseralized instance of concrete implementation of `Model` class.
    """
    model_type_filepath = os.path.join(models_dir, "model_type.txt")
    with open(model_type_filepath, encoding="utf-8") as f:
        model_name = f.read()

    try:
        return MODEL_SUBCLASSES[model_name].load(model_id, models_dir, device)
    except KeyError as e:
        raise ModelClassNotFound() from e
