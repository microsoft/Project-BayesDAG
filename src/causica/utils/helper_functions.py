"""
Helper functions.
"""
import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Tuple, TypeVar, Union

import git
import numpy as np
import torch

T = TypeVar("T")


def convert_dict_of_lists_to_ndarray(dict_: Dict[T, List]) -> Dict[T, np.ndarray]:
    """
    Converts all list values in `dict_` to ndarrays.
    If no value is of type list, the passed dictionary is returned.
    """
    return {k: np.array(v) if isinstance(v, list) else v for k, v in dict_.items()}


def convert_dict_of_ndarray_to_lists(dict_):
    """
    Converts all ndarray values in `dict_` to (possibly nested) lists.
    If no value is of type ndarray, the passed dictionary is returned.
    """
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in dict_.items()}


def to_tensors(
    *arrays: Union[torch.Tensor, np.ndarray], device: torch.device, dtype: torch.dtype = torch.float
) -> Tuple[torch.Tensor, ...]:
    return tuple(torch.tensor(array, dtype=dtype, device=device) for array in arrays)


@contextmanager
def maintain_random_state(do_maintain=True):
    torch_rand_state = torch.get_rng_state()
    np_rand_state = np.random.get_state()
    if torch.cuda.is_available():
        cuda_rand_state = torch.cuda.get_rng_state()
    else:
        cuda_rand_state = None

    try:
        yield (torch_rand_state, np_rand_state, cuda_rand_state)
    finally:
        if do_maintain:
            if torch_rand_state is not None:
                torch.set_rng_state(torch_rand_state)
            if cuda_rand_state is not None:
                torch.cuda.set_rng_state(cuda_rand_state)
            if np_rand_state is not None:
                np.random.set_state(np_rand_state)


def get_random_state():
    """
    Get random states for PyTorch, PyTorch CUDA and Numpy.

    Returns:
        Dictionary of state type: state value.
    """
    states = {
        "torch_rand_state": torch.get_rng_state(),
        "np_rand_state": np.random.get_state(),
    }

    if torch.cuda.is_available():
        states["cuda_rand_state"] = torch.cuda.get_rng_state()

    return states


def write_git_info(directory: str, exist_ok: bool = False):
    """
    Write sys.argv, git hash, git diff to <directory>/git_info.txt

    directory: where to write git_info.txt.  This directory must already exist
    exist_ok: if set to True, may silently overwrite old git info
    """
    assert os.path.exists(directory)
    try:
        repo = git.Repo(search_parent_directories=True)

    except git.InvalidGitRepositoryError as exc:
        # Likely to happen if we are in an AzureML run.
        raise ValueError("Not running inside a Git repo.") from exc
    commit = repo.head.commit
    diff = repo.git.diff(None)
    mode = "w" if exist_ok else "x"
    with open(os.path.join(directory, "git_info.txt"), mode, encoding="utf-8") as f:
        f.write(f"sys.argv: {sys.argv}\n")
        f.write("Git commit: " + str(commit) + "\n")
        try:
            f.write("Active branch: " + str(repo.active_branch) + "\n")
        except TypeError:
            # Happens in PR build, detached head state
            pass
        f.write("Git diff:\n" + str(diff))

def fill_triangular(vec: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """
    Args:
        vec: A tensor of shape (..., n(n-1)/2)
        upper: whether to fill the upper or lower triangle
    Returns:
        An array of shape (..., n, n), where the strictly upper (lower) triangle is filled from vec
        with zeros elsewhere
    """
    num_nodes = num_lower_tri_elements_to_n(vec.shape[-1])
    if upper:
        idxs = torch.triu_indices(num_nodes, num_nodes, offset=1, device=vec.device)
    else:
        idxs = torch.tril_indices(num_nodes, num_nodes, offset=-1, device=vec.device)
    output = torch.zeros(vec.shape[:-1] + (num_nodes, num_nodes), device=vec.device)
    output[..., idxs[0, :], idxs[1, :]] = vec
    return output


def unfill_triangular(mat: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """
    Fill a vector of length n(n-1)/2 with elements from the strictly upper(lower) triangle.

    Args:
        mat: A tensor of shape (..., n, n)
        upper: whether to fill from the upper triangle
    Returns:
        A vector of shape (..., n(n-1)/2), filled from the upper triangle
    """
    num_nodes = mat.shape[-1]
    if upper:
        idxs = torch.triu_indices(num_nodes, num_nodes, offset=1, device=mat.device)
    else:
        idxs = torch.tril_indices(num_nodes, num_nodes, offset=-1, device=mat.device)
    return mat[..., idxs[0, :], idxs[1, :]]


def num_lower_tri_elements_to_n(x: int) -> int:
    """
    Calculate the size of the matrix from the number of strictly lower triangular elements.

    We have x = n(n - 1) / 2 for some n
    n² - n - 2x = 0
    so n = (1 + √(1 + 8x)) / 2
    """
    val = int(np.sqrt(1 + 8 * x) + 1) // 2
    if val * (val - 1) != 2 * x:
        raise ValueError("Invalid number of lower triangular elements")
    return val