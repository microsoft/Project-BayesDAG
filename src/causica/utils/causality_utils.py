from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..utils.helper_functions import to_tensors


def intervene_graph(
    adj_matrix: torch.Tensor, intervention_idxs: Optional[torch.Tensor], copy_graph: bool = True
) -> torch.Tensor:
    """
    Simulates an intervention by removing all incoming edges for nodes being intervened

    Args:
        adj_matrix: torch.Tensor of shape (input_dim, input_dim) containing  adjacency_matrix
        intervention_idxs: torch.Tensor containing which variables to intervene
        copy_graph: bool whether the operation should be performed in-place or a new matrix greated
    """
    if intervention_idxs is None or len(intervention_idxs) == 0:
        return adj_matrix

    if copy_graph:
        adj_matrix = adj_matrix.clone()

    adj_matrix[:, intervention_idxs] = 0
    return adj_matrix


def intervention_to_tensor(
    intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]],
    intervention_values: Optional[Union[torch.Tensor, np.ndarray]],
    group_mask,
    device,
    is_temporal: bool = False,
) -> Tuple[Optional[torch.Tensor], ...]:
    """
    Maps empty interventions to nan and np.ndarray intervention data to torch tensors.
    Converts indices to a mask using the group_mask. If the intervention format is temporal, set is_temporal to True.
    Args:
        intervention_idxs: np.ndarray or torch.Tensor with shape [num_interventions] (for static data) or [num_interventions, 2] (for temporal data).
        intervention_values: np.ndarray or torch.Tensor with shape [proc_dims] storing the intervention values corresponding to the intervention_idxs.
        group_mask: np.ndarray, a mask of shape (num_groups, num_processed_cols) indicating which column
            corresponds to which group.
        is_temporal: Whether intervention_idxs in temporal 2D format.

    Returns:

    """

    intervention_mask = None

    if intervention_idxs is not None and intervention_values is not None:
        (intervention_idxs,) = to_tensors(intervention_idxs, device=device, dtype=torch.long)
        (intervention_values,) = to_tensors(intervention_values, device=device, dtype=torch.float)

        if intervention_idxs.dim() == 0:
            intervention_idxs = None

        if intervention_values.dim() == 0:
            intervention_values = None


        intervention_mask = get_mask_from_idxs(intervention_idxs, group_mask, device)

    assert intervention_idxs is None or isinstance(intervention_idxs, torch.Tensor)
    assert intervention_values is None or isinstance(intervention_values, torch.Tensor)
    return intervention_idxs, intervention_mask, intervention_values  # type: ignore


def get_mask_from_idxs(idxs, group_mask, device) -> torch.Tensor:
    """
    Generate mask for observations or samples from indices using group_mask
    """
    mask = torch.zeros(group_mask.shape[0], device=device, dtype=torch.bool)
    mask[idxs] = 1
    (group_mask,) = to_tensors(group_mask, device=device, dtype=torch.bool)
    mask = (mask.unsqueeze(1) * group_mask).sum(0).bool()
    return mask


