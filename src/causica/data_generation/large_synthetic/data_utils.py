import os
import pickle as pkl
import random

import graphviz
import igraph as ig
import numpy as np
import pyro.distributions as dist
import torch
from scipy.special import softmax

from ...utils.nri_utils import BGe, enum_all_graphs
from ...utils.torch_utils import generate_fully_connected


def save_data(
    dataset_folder, name, directed_matrix, X_train_test, X_train, X_test, held_out_interventions, true_posterior, graph_args
):
    savedir = os.path.join(dataset_folder, name)
    os.makedirs(savedir, exist_ok=True)
    np.savetxt(os.path.join(savedir, "adj_matrix.csv"), directed_matrix, delimiter=",", fmt="%i")
    np.savetxt(os.path.join(savedir, "all.csv"), X_train_test, delimiter=",")
    np.savetxt(os.path.join(savedir, "train.csv"), X_train, delimiter=",")
    np.savetxt(os.path.join(savedir, "test.csv"), X_test, delimiter=",")
    with open(os.path.join(savedir, "held_out_interventions.pkl"), "wb") as b:
        pkl.dump(held_out_interventions, b)
    with open(os.path.join(savedir, "true_posterior.pkl"), "wb") as b:
        pkl.dump(true_posterior, b)
    with open(os.path.join(savedir, "graph_args.pkl"), "wb") as b:
        pkl.dump(graph_args, b)

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine="dot")
    names = labels if labels else [f"x{i}" for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d


def simulate_dag(np_seed, d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        np_seed (seed_iterator): Instance of class that tracks seeds to ensure reproducibility
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    np_seed.set_seed()

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif graph_type == "FC":
        # Fully connected DAG
        M = np.zeros((d, d))
        M[np.triu_indices(d, k=1)] = 1
        B = _random_permutation(M)

    else:
        raise ValueError("unknown graph type")
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_bidirected_matrix(np_seed, d, s0):
    """

    Args:
        np_seed (seed_iterator): Instance of class that tracks seeds to ensure reproducibility
        d (int): num of nodes
        s0 (int): expected num of latent edges

    Returns:
        a bidirected adj matrix

    """
    np_seed.set_seed()
    tril_graph = np.tril(np.random.binomial(1, s0 / (d * (d - 1) / 2), [d, d]), -1)
    return tril_graph + tril_graph.T


def simulate_single_equation(
    np_seed,
    X,
    scale,
    sem_type,
    noise_type,
    noise_mult_factor,
    output_classes=None,
    discrete_temperature=3,
    latent_parent_idxs=None,
):
    """X: [n, num of parents], x: [n]"""

    observed_parent_idxs = list(range(X.shape[1]))
    observed_parent_idxs = [d for d in observed_parent_idxs if d not in latent_parent_idxs]

    # Generate Noise
    np_seed.set_seed()
    n, pa_size = X.shape

    if output_classes is None:
        noise_size = (n, 1)
    else:
        noise_size = (n, output_classes)

    z = np.random.normal(scale=scale, size=noise_size)

    if noise_type == "uniform":
        half_side = np.sqrt(12) / 2  # this choice results in uniform variance noise
        z = np.random.uniform(low=-half_side, high=half_side, size=noise_size)

    elif noise_type == "mlp":
        noise_f = sample_mlp(noise_size[1], np_seed, noise_size[1])
        z = noise_f(z)
        z = z / z.std() * scale

    assert sem_type in ["mlp", "linear"]
    assert noise_type in ["uniform", "mlp", "fixed", "unequal"] or isinstance(noise_type, float)

    pa_size_observed = pa_size

    if pa_size == 0:
        z = z * noise_mult_factor

    f_observed = sample_function(sem_type, pa_size_observed, np_seed, output_classes, scale)

    def f(x):
        return f_observed(x[:, observed_parent_idxs])

    if output_classes is None:
        ff = f(X)
        # ff = np.clip(ff, a_min=-6, a_max=6)
        if isinstance(ff, np.ndarray):
            ff = ff.squeeze()
            assert len(ff.shape) == 1
        return ff + z[:, 0]
    else:
        logits = discrete_temperature * f(X) + z
        return categorical_sample_from_logits(logits, np_seed, argmax=False)


def simulate_nonlinear_sem(
    np_seed,
    B,
    n,
    sem_type,
    noise_type,
    noise_scale,
    noise_mult_factor=1.0,
    discrete_dims_list=None,
    intervention=None,
    discrete_temperature=3,
    latent_idxs=None,
):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): linear, mlp
        noise_scale (np.ndarray): scale parameter of additive noise
        noise_mul_factor (float): noise constant multiplicative factor for parent variables
        intervention (np.ndarray of floats or NaNs): which variables to intervene on and values to which to set them
        latent_idxs: list of latent confounders
    Returns:
        X (np.ndarray): [n, d] sample matrix
    """

    if latent_idxs is None:
        latent_idxs = []

    d = B.shape[0]

    X = np.zeros([n, d])

    if discrete_dims_list is None:
        discrete_dims_list = [None] * d
    else:
        assert len(discrete_dims_list) == d

    X_int, intervene_idxs = intervene_values(X, intervention)

    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d

    for j in ordered_vertices:
        output_classes = discrete_dims_list[j]

        parents = G.neighbors(j, mode=ig.IN)
        latent_parent_idxs = [parents.index(x) for x in (set(parents) & set(latent_idxs))]
        X[:, j] = simulate_single_equation(
            np_seed,
            X[:, parents],
            noise_scale[j],
            sem_type,
            noise_type,
            noise_mult_factor,
            output_classes=output_classes,
            discrete_temperature=discrete_temperature,
            latent_parent_idxs=latent_parent_idxs,
        )  # if a node has no parents it will be set to noise

        if j in intervene_idxs:  # we do this in this order to respect the random number sampling order
            if output_classes is not None:
                assert np.all(X_int[:, j] == 1)
            X[:, j] = X_int[:, j]
    return X


def get_name(gt, N, E, sem_type, noise_type, seed, is_tune=False):
    if is_tune:
        return f"tune_{gt}_{N}_{E}_{sem_type}_sem_{noise_type}_noise_{seed}_seed"
    else:
        return f"run_{gt}_{N}_{E}_{sem_type}_sem_{noise_type}_noise_{seed}_seed"


def sample_parent_node(G: ig.Graph.Adjacency, node: int, max_depth: int = 2, latent_idxs=None) -> int:
    """Sample a parent node from a DAG.

    Args:
        G (ig.Graph.Adjacency): DAG
        node (str): node to sample from
        max_depth (int): max depth of sampling
        latent_idxs: latent confounder indexes

    Returns:
        int: sampled parent node
    """
    if latent_idxs is None:
        latent_idxs = []
    parents = G.neighbors(node, mode=ig.IN)
    parents = [pa for pa in parents if pa not in latent_idxs]

    if len(parents) == 0:
        raise ValueError(f"node {node} has no observed parents")

    parent_idx = np.random.choice(parents)
    if max_depth > 1 and len(G.neighbors(parent_idx, mode=ig.IN)) > 0:
        go_deeper = np.random.random() < 0.5
        if go_deeper:
            return sample_parent_node(G, parent_idx, max_depth=max_depth - 1, latent_idxs=latent_idxs)
    return int(parent_idx)


def gen_dataset(
    base_seed,
    num_samples_train,
    num_samples_test,
    graph_type,
    N,
    E,
    sem_type,
    noise_type,
    N_interventions=5,
    max_parent_depth=2,
    use_quantile_references=False,
    generate_references=False,
    adj_matrix=None,
    noise_mult_factor=1,
    discrete_dims_list=None,
    discrete_temperature=3,
    expected_num_latent_confounders=0,
):
    seed = seed_iterator(base_seed)
    # Variances for noises
    graph_args = {"num_variables": N, "exp_edges": E, "seed": base_seed, "graph_type": graph_type, "exp_edges_per_node": E / N}
    if expected_num_latent_confounders > 0:
        # Number of potential bidirected edges between all possible pairs of the observed nodes
        N_all = N + N * (N - 1) // 2
    else:
        N_all = N

    if noise_type in ["fixed", "mlp", "spline", "uniform"]:
        noise_scales = np.ones(N_all)
    elif noise_type == "unequal":
        noise_scales = np.sqrt(
            dist.InverseGamma(torch.tensor([1.5] * N), torch.tensor([1.0] * N)).rsample().cpu().numpy()
        )
    else:
        noise_scales = np.random.uniform(low=0.2, high=noise_type, size=(N_all,))
    latent_idxs = []

    directed_matrix = None
    bidirected_matrix = None

    if adj_matrix is None:
        # True adjacency matrix
        directed_matrix = simulate_dag(seed, N, E, graph_type)
        adj_matrix = directed_matrix

    seed = seed_iterator(base_seed)
    X = simulate_nonlinear_sem(
        seed,
        adj_matrix,
        num_samples_train + num_samples_test,
        sem_type,
        noise_type,
        noise_scales,
        noise_mult_factor,
        discrete_dims_list,
        intervention=None,
        discrete_temperature=discrete_temperature,
        latent_idxs=latent_idxs,
    )

    X_test = X[:num_samples_test, :]
    X_train = X[num_samples_test:, :]

    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std

    G = ig.Graph.Adjacency(adj_matrix.tolist())

    N_interventions = min(N_interventions, N)

    X_train_test = np.concatenate([X_train, X_test], axis=0)
    # Sample held out interventions
    held_out_interventions = []

    for node in range(N):
        # for value in np.linspace(node_range[0], node_range[1], 5):
        for value in [-20.0, 20.0]:
            node_binary_v = np.ones(N) * np.nan
            node_binary_v[node] = value

            seed = seed_iterator(base_seed)
            X = simulate_nonlinear_sem(
                seed,
                adj_matrix,
                200,
                sem_type,
                noise_type,
                noise_scales,
                noise_mult_factor,
                discrete_dims_list,
                intervention=None,
                discrete_temperature=discrete_temperature,
                latent_idxs=latent_idxs,
            )
            held_out_interventions.append({"node": node, "value": value, "samples": X})
    true_posterior = None
    if sem_type == "linear" and noise_type=="unequal" and N<6:
        true_posterior = compute_true_posterior(num_nodes=N, data=torch.from_numpy(X_train[:, :N]))
    return (X_train_test[:, :N], X_train[:, :N], X_test[:, :N], directed_matrix, held_out_interventions, true_posterior, graph_args)


def intervene_graph(adj_matrix, intervention):
    if intervention is None:
        return adj_matrix

    intervene_idxs = np.where(np.logical_not(np.isnan(intervention)))[0]
    adj_matrix = adj_matrix.copy()
    adj_matrix[:, intervene_idxs] = 0
    return adj_matrix


def intervene_values(X, intervention):
    if intervention is None:
        return X, []

    intervene_idxs = np.where(np.logical_not(np.isnan(intervention)))[0]
    X = X.copy()
    X[:, intervene_idxs] = intervention[intervene_idxs]
    return X, intervene_idxs


class seed_iterator(object):
    def __init__(self, seed=0):
        self.seed = seed

    def set_seed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.seed += 1

    def set_seed_pt(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.seed += 1


def categorical_sample_from_logits(logits, np_seed, argmax=False):
    np_seed.set_seed()
    assert len(logits.shape) == 2
    assert logits.shape[1] > 1
    if argmax:
        return np.argmax(logits, axis=1)

    probabilities = softmax(logits, axis=-1)

    cum_prob = np.cumsum(probabilities, axis=-1)  # shape (n1, m)
    r = np.random.uniform(size=(cum_prob.shape[0], 1))

    # argmax finds the index of the first True value in the last axis.
    samples = np.argmax(cum_prob > r, axis=-1)

    # In case want to randomly shuffle meaning of labels
    #     permute_vec = np.random.choice(logits.shape[1], size=logits.shape[1], replace=False)
    return samples


def sample_mlp(input_dim, np_seed, output_classes=None):
    np_seed.set_seed()
    output_classes = 1 if output_classes is None else output_classes
    model = generate_fully_connected(
        input_dim=input_dim,
        output_dim=output_classes,
        hidden_dims=[
            5,
        ],
        non_linearity=torch.nn.ReLU,
        init_method="normal",
        device="cpu",
        activation=torch.nn.Identity,
    )

    def func(X):
        with torch.no_grad():
            return model(torch.from_numpy(X).to(torch.float32)).numpy()

    return func


def sample_linear(input_dim, np_seed, output_classes=None, conjugate=False, scale=None):
    np_seed.set_seed()
    if conjugate:
        if output_classes is not None:
            W = np.random.normal(scale=scale, loc=0.0, size=(input_dim, output_classes))
            b = np.zeros((1, output_classes))
        else:
            W = np.random.normal(scale=scale, loc=0.0, size=input_dim)
            b = np.zeros((1,))
    if output_classes is not None:
        W = np.random.uniform(low=0.5 / input_dim, high=1.5 / input_dim, size=(input_dim, output_classes))
        b = np.random.uniform(low=0.3, high=0.3, size=(1, output_classes))
    else:
        W = np.random.uniform(low=0.5, high=1.5, size=input_dim)
        b = np.random.uniform(low=0.3, high=0.3, size=(1,))

    def func(X):
        return X @ W

    return func


def sample_function(sem_type, pa_size, np_seed, output_classes, scale):
    def f(_):
        return 0

    if pa_size != 0:
        if sem_type == "mlp":
            f = sample_mlp(pa_size, np_seed, output_classes)
        elif sem_type == "linear":
            f = sample_linear(pa_size, np_seed, output_classes)
        elif sem_type == "conjugate":
            f = sample_linear(pa_size, np_seed, output_classes, conjugate=True, scale=scale)

    return f


def compute_true_posterior_linear_g(all_graph, intervention_mask, data):
    """_summary_
    Args:
        graph (torch.tensor): Graphs for which true posterior need to be computed for (L x d x d)
        intervention_mask (_type_): Boolean interventional targets indicating whether a particular node was intervened on for each sample (N x d)
        data (_type_): Dataset the model is trained on (N x d)
    """
    #!! for data, need to pass the data the model is trained on i.e. training data in entirety

    posterior = BGe(data=data)
    if len(all_graph.shape) == 3:
        graphs = all_graph
    elif len(all_graph.shape) == 2:
        graphs = all_graph.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected graph shape: {all_graph.shape}")
    mll = posterior.log_marginal_likelihood_given_g(w=graphs)

    return {"all_graphs": np.array([g.numpy() for g in graphs]), "log_marginal_likelihood": mll.detach().numpy()}


def compute_true_posterior(num_nodes, data, intervention_mask=None):
    if intervention_mask is None:
        intervention_mask = torch.zeros_like(data).to(torch.bool)

    all_graphs = torch.from_numpy(enum_all_graphs(num_nodes=num_nodes, dags_only=True)).to(data.device)

    return compute_true_posterior_linear_g(all_graph=all_graphs, intervention_mask=intervention_mask, data=data)
