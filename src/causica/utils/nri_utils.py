from itertools import product
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import torch.distributions as td
from sklearn.gaussian_process.kernels import RBF
from scipy.special import logsumexp
from sklearn import metrics as sklearn_metrics
from ..datasets.dataset import CausalDataset
import graphical_models as gm

def threshold_metrics(predicted_graphs, gt_graphs):
    g_flat = gt_graphs.reshape(-1)
    log_weights = np.array([-np.log(predicted_graphs.shape[0])])

    # P(G_ij = 1) = sum_G w_G 1[G = G] in log space
    log_edge_belief, log_edge_belief_sgn = logsumexp(
        log_weights[..., np.newaxis, np.newaxis], 
        b=predicted_graphs.astype(log_weights.dtype), 
        axis=0, return_sign=True)

    # L1 edge error
    p_edge = log_edge_belief_sgn * np.exp(log_edge_belief)
    p_edge_flat = p_edge.reshape(-1)

    # threshold metrics 
    fpr_, tpr_, _ = sklearn_metrics.roc_curve(g_flat, p_edge_flat)
    roc_auc_ = sklearn_metrics.auc(fpr_, tpr_)
    precision_, recall_, _ = sklearn_metrics.precision_recall_curve(g_flat, p_edge_flat)
    prc_auc_ = sklearn_metrics.auc(recall_, precision_)

    return {"roc_auc": roc_auc_, "prc_auc": prc_auc_}
def is_there_adjacency(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1)/2 indicating whether each edge is present or not (not
    considering orientation).
    """
    mask = np.tri(adj_matrix.shape[0], k=-1, dtype=bool)
    is_there_forward = adj_matrix[mask].astype(bool)
    is_there_backward = (adj_matrix.T)[mask].astype(bool)
    return is_there_backward | is_there_forward


def get_adjacency_type(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1)/2 indicating the type of each edge (that is, 0 if
    there is no edge, 1 if it is forward, -1 if it is backward and 2 if it is in both directions or undirected).
    """

    def aux(f, b):
        if f and b:
            return 2
        elif f and not b:
            return 1
        elif not f and b:
            return -1
        elif not f and not b:
            return 0

    mask = np.tri(adj_matrix.shape[0], k=-1, dtype=bool)
    is_there_forward = adj_matrix[mask].astype(bool)
    is_there_backward = (adj_matrix.T)[mask].astype(bool)
    out = np.array([aux(f, b) for (f, b) in zip(is_there_forward, is_there_backward)])
    return out


def is_there_edge(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1) indicating whether each edge is present or not (considering orientation).
    """
    mask = (np.ones_like(adj_matrix) - np.eye(adj_matrix.shape[0])).astype(bool)
    return adj_matrix[mask].astype(bool)


def edge_prediction_metrics(adj_matrix_true, adj_matrix_predicted, adj_matrix_mask=None):
    """
    Computes the edge predicition metrics when the ground truth DAG (or CPDAG) is adj_matrix_true and the predicted one
    is adj_matrix_predicted. Both are numpy arrays.
    adj_matrix_mask is the mask matrix for adj_matrices, that indicates which subgraph is partially known in the ground
    truth. 0 indicates the edge is unknwon, and 1 indicates that the edge is known.
    """
    results = {}
    if adj_matrix_mask is None:
        adj_matrix_mask = np.ones_like(adj_matrix_true)

    assert ((adj_matrix_true == 0) | (adj_matrix_true == 1)).all()
    assert ((adj_matrix_predicted == 0) | (adj_matrix_predicted == 1)).all()


    # Computing orientation precision/recall
    v_mask = is_there_adjacency(adj_matrix_mask)
    v_true = get_adjacency_type(adj_matrix_true) * v_mask
    v_predicted = get_adjacency_type(adj_matrix_predicted) * v_mask
    recall = ((v_true == v_predicted) & (v_true != 0)).sum() / (v_true != 0).sum()
    precision = (
        ((v_true == v_predicted) & (v_predicted != 0)).sum() / (v_predicted != 0).sum()
        if (v_predicted != 0).sum() != 0
        else 0.0
    )
    fscore = 2 * recall * precision / (precision + recall) if (recall + precision) != 0 else 0.0
    results["orientation_fscore"] = fscore

    # Compute SHD and number of nonzero edges
    results["shd"] = _shd(adj_matrix_true, adj_matrix_predicted)
    results["nnz"] = adj_matrix_predicted.sum()

    #Compute CPDAG metrics
    cpdag_true, _ = gm.DAG.from_amat(adj_matrix_true).cpdag().to_amat()
    try:
        cpdag_predicted, _ = gm.DAG.from_amat(adj_matrix_predicted).cpdag().to_amat()
    except:
        cpdag_predicted = adj_matrix_predicted
    results["cpdag-shd"] = _shd(cpdag_true, cpdag_predicted)
    return results


def _shd(adj_true, adj_pred):
    """
    Computes Structural Hamming Distance as E+M+R, where E is the number of extra edges,
    M the number of missing edges, and R the number os reverse edges.
    """
    E, M, R = 0, 0, 0
    for i in range(adj_true.shape[0]):
        for j in range(adj_true.shape[0]):
            if j <= i:
                continue
            if adj_true[i, j] == 1 and adj_true[j, i] == 0:
                if adj_pred[i, j] == 0 and adj_pred[j, i] == 0:
                    M += 1
                elif adj_pred[i, j] == 0 and adj_pred[j, i] == 1:
                    R += 1
                elif adj_pred[i, j] == 1 and adj_pred[j, i] == 1:
                    E += 1
            if adj_true[i, j] == 0 and adj_true[j, i] == 1:
                if adj_pred[i, j] == 0 and adj_pred[j, i] == 0:
                    M += 1
                elif adj_pred[i, j] == 1 and adj_pred[j, i] == 0:
                    R += 1
                elif adj_pred[i, j] == 1 and adj_pred[j, i] == 1:
                    E += 1
            if adj_true[i, j] == 0 and adj_true[j, i] == 0:
                E += adj_pred[i, j] + adj_pred[j, i]
    return E + M + R


def edge_prediction_metrics_multisample(
    adj_matrix_true, adj_matrices_predicted, adj_matrix_mask=None, compute_mean=True, is_dag=None,
):
    """
    Computes the edge predicition metrics when the ground truth DAG (or CPDAG) is adj_matrix_true and many predicted
    adjacencies are sampled from the distribution. Both are numpy arrays, adj_matrix_true has shape (n, n) and
    adj_matrices_predicted has shape (M, n, n), where M is the number of matrices sampled.
    """
    results = {}
    n_vars = adj_matrix_true.shape[-1]
    if is_dag is not None:
        if is_dag.sum()==0:
        
            results["orientation_fscore"] = 0.5
            results["shd"] = adj_matrix_predicted.shape[-1]*(adj_matrix_predicted.shape[-1]-1)
            results["nnz"] = float("nan")
            results["cpdag-shd"] = adj_matrix_predicted.shape[-1]*(adj_matrix_predicted.shape[-1]-1)
            return results
        
    for i in range(adj_matrices_predicted.shape[0]):
        if is_dag is not None:
            if not is_dag[i]:
                continue
        adj_matrix_predicted = adj_matrices_predicted[i, :, :]  # (n, n)
        results_local = edge_prediction_metrics(adj_matrix_true, adj_matrix_predicted, adj_matrix_mask=adj_matrix_mask)
        for k, result in results_local.items():
            if k not in results:
                results[k] = []
            results[k].append(result)
    if is_dag is None:
        is_dag = np.ones(adj_matrices_predicted.shape[0], dtype=bool)
    if compute_mean:
        return {key: np.mean(val) for key, val in results.items()}
    return results


def compute_dag_loss(vec, num_nodes):
    """
    vec is a n*(n-1) array with the flattened adjacency matrix (without the diag).
    """
    dev = vec.device
    adj_matrix = torch.zeros(num_nodes, num_nodes, device=dev)
    mask = (torch.ones(num_nodes, num_nodes, device=dev) - torch.eye(num_nodes, device=dev)).to(bool)
    adj_matrix[mask] = vec
    return torch.abs(torch.trace(torch.matrix_exp(adj_matrix * adj_matrix)) - num_nodes)


def get_feature_indices_per_node(variables):
    """
    Returns a list in which the i-th element is a list with the features indices that correspond to the i-th node.
    For each Variable in 'variables' argument, the node is specified through the group_name field.
    """
    nodes = [v.group_name for v in variables]
    nodes_unique = sorted(set(nodes))
    if len(nodes_unique) == len(nodes):
        nodes_unique = nodes
    output = []
    for node in nodes_unique:
        output.append([i for (i, e) in enumerate(nodes) if e == node])
    return output, nodes_unique


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    """
    preds: [num_sims, num_edges, num_edge_types]
    log_prior: [1, 1, num_edge_types]
    """
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def piecewise_linear(x, start, width, max_val=1):
    """
    Piecewise linear function whose value is:
        0 if x<=start
        max_val if x>=start+width
        grows linearly from 0 to max_val if start<=x<=(start+width)
    It is used to define the coefficient of the DAG-loss in NRI-MV.
    """
    return max_val * max(min((x - start) / width, 1), 0)


def enum_all_graphs(num_nodes: int, dags_only: Optional[bool] = False):
    """
    Enumerates all graphs of size num_nodes with no self-loops (all diagonals elements are strictly 0 in adj. matrix).
    Useful for computing the full posterior and true posterior.

    Args:
        num_nodes: An int specifying the number of nodes in the graphs (should be less than 6).
        dags_only: Whether to return only DAGs of size num_nodes.

    Returns: Adjacency matrices corresponding to all the graphs
    """

    assert (
        num_nodes < 6
    ), f"Enumeration of DAGs possible when No. of nodes is less than 6, received {num_nodes} instead."
    comb_ = list(product([0, 1], repeat=num_nodes * (num_nodes - 1)))  # Exclude diagonal
    comb = np.array(comb_)
    idxs_upper = np.triu_indices(num_nodes, k=1)
    idxs_lower = np.tril_indices(num_nodes, k=-1)
    output = np.zeros(comb.shape[:-1] + (num_nodes, num_nodes))
    output[..., idxs_upper[0], idxs_upper[1]] = comb[..., : (num_nodes * (num_nodes - 1)) // 2]
    output[..., idxs_lower[0], idxs_lower[1]] = comb[..., (num_nodes * (num_nodes - 1)) // 2 :]
    if dags_only:
        return output[dag_constraint(output) == 0]
    return output


def dag_constraint(A: np.ndarray):
    """
    Computes the DAG constraint based on the matrix exponential.
    Computes tr[e^(A * A)] - num_nodes.

    Args:
        A: Batch of adjacency matrices of size [batch_size, num_nodes, num_nodes]
    Returns: The DAG constraint values for each adj. matrix in the batch.
    """
    assert A.shape[-1] == A.shape[-2]

    num_nodes = A.shape[-1]
    expm_A = torch.linalg.matrix_exp(torch.from_numpy(A * A)).cpu().numpy()
    return np.trace(expm_A, axis1=-1, axis2=-2) - num_nodes



class BGe(torch.nn.Module):
    """
    Pytorch implementation of Linear Gaussian-Gaussian Model (Continuous Gaussian data)
    Supports batched version.
     Each variable distributed as Gaussian with mean being the linear combination of its parents
    weighted by a Gaussian parameter vector (i.e., with Gaussian-valued edges).
    The parameter prior over (mu, lambda) of the joint Gaussian distribution (mean `mu`, precision `lambda`) over x is Gaussian-Wishart,
    as introduced in
            Geiger et al (2002):  https://projecteuclid.org/download/pdf_1/euclid.aos/1035844981
    Computation is based on
            Kuipers et al (2014): https://projecteuclid.org/download/suppdf_1/euclid.aos/1407420013
    """

    def __init__(self, *, data, mean_obs=None, alpha_mu=None, alpha_lambd=None, device="cpu"):
        """
        mean_obs : [num_nodes] : Mean of each Gaussian distributed variable (Prior)
        alpha_mu : torch.float64 : Hyperparameter of Wishart Prior
        alpha_lambd :
        data : If provided, precomputes the posterior parameter R
        """
        super(BGe, self).__init__()
        self.N, self.d = data.shape
        self.mean_obs = mean_obs or torch.zeros(self.d, device=device)
        self.alpha_mu = alpha_mu or torch.tensor(1.0)
        self.alpha_lambd = alpha_lambd or torch.tensor(self.d + 2.)
        self.device = device

        # pre-compute matrices
        small_t = (self.alpha_mu * (self.alpha_lambd - self.d - 1)) / (self.alpha_mu + 1)
        T = small_t * torch.eye(self.d)

        x_bar = data.mean(axis=0, keepdims=True)
        x_center = data - x_bar
        s_N = torch.matmul(x_center.t(), x_center)  # [d, d]
        self.R = (
            T
            + s_N
            + ((self.N * self.alpha_mu) / (self.N + self.alpha_mu))
            * (torch.matmul((x_bar - self.mean_obs).t(), x_bar - self.mean_obs))
        ).to(self.device)

        all_l = torch.arange(self.d)

        self.log_gamma_terms = (
            0.5 * (np.log(self.alpha_mu) - np.log(self.N + self.alpha_mu))
            + torch.special.gammaln(0.5 * (self.N + self.alpha_lambd - self.d + all_l + 1))
            - torch.special.gammaln(0.5 * (self.alpha_lambd - self.d + all_l + 1))
            - 0.5 * self.N * np.log(np.pi)
            + 0.5 * (self.alpha_lambd - self.d + 2 * all_l + 1) * np.log(small_t)
        ).to(self.device)

    def slogdet_pytorch(self, parents, R=None):
        """
        Batched log determinant of a submatrix
        Done by masking everything but the submatrix and
        adding a diagonal of ones everywhere else for the
        valid determinant
        """

        if R is None:
            R = self.R.clone()
        batch_size = parents.shape[0]
        R = R.unsqueeze(0).expand(batch_size, -1, -1)
        parents = parents.to(torch.float64).to(self.device)
        mask = torch.matmul(parents.unsqueeze(2), parents.unsqueeze(1)).to(torch.bool)  # [batch_size, d,d]
        R = torch.where(mask, R, torch.tensor([np.nan], device=self.device).to(torch.float64))
        submat = torch.where(
            torch.isnan(R),
            torch.eye(self.d, dtype=torch.float64).unsqueeze(0).expand(batch_size, -1, -1).to(self.device),
            R,
        )
        return torch.linalg.slogdet(submat)[1]

    def log_marginal_likelihood_given_g_j(self, j, w):
        """
        Computes node specific terms of BGe metric
        j : Node to compute the marginal likelihood. Marginal Likelihood decomposes over each node.
        w : [batch_size, num_nodes, num_nodes] : {0,1} adjacency matrix
        """

        batch_size = w.shape[0]
        isj = (torch.arange(self.d) == j).unsqueeze(0).expand(batch_size, -1).to(self.device)
        parents = w[:, :, j] == 1
        parents_and_j = parents | isj
        n_parents = (w.sum(axis=1)[:, j]).long()
        n_parents_mask = n_parents == 0
        _log_term_r_no_parents = -0.5 * (self.N + self.alpha_lambd - self.d + 1) * torch.log(torch.abs(self.R[j, j]))

        _log_term_r = 0.5 * (self.N + self.alpha_lambd - self.d + n_parents[~n_parents_mask]) * self.slogdet_pytorch(
            parents[~n_parents_mask]
        ) - 0.5 * (self.N + self.alpha_lambd - self.d + n_parents[~n_parents_mask] + 1) * self.slogdet_pytorch(
            parents_and_j[~n_parents_mask]
        )  # log det(R_II)^(..) / det(R_JJ)^(..)

        log_term_r = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        log_term_r[n_parents_mask] = _log_term_r_no_parents
        log_term_r[~n_parents_mask] = _log_term_r
        return log_term_r + self.log_gamma_terms[n_parents]

    def log_marginal_likelihood_given_g(self, w, interv_targets=None):
        """Computes log p(x | G) in closed form using conjugacy properties
        w: 	{0,1} adjacency marix
        interv_targets: boolean mask of whether or not a node was intervened on
                        intervened nodes are ignored in likelihood computation
        """
        batch_size = w.shape[0]
        if interv_targets is None:
            interv_targets = torch.zeros(batch_size, self.d).to(torch.bool)
        interv_targets = (~interv_targets).to(self.device)
        # sum scores for all nodes
        mll = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        for i in range(self.d):
            # print(self.log_marginal_likelihood_given_g_j(i, w)[interv_targets[:,i]])
            mll[interv_targets[:, i]] += self.log_marginal_likelihood_given_g_j(i, w)[
                interv_targets[:, i]
            ]  ##TODO: Possible to use torch.vmap but should be okay for now
        return mll

def mmd_true_posterior(log_marginal_likelihood_true_posterior, all_graphs, graph_samples_from_model, exp_edges_per_node=1.0, graph_type="er"):
    """
    Computes the MMD between the true posterior and the model posterior.
    Args:
        log_marginal_likelihood_true_posterior: The log marginal likelihood of the true posterior.
        all_graphs: All the DAGs of size d.
        graph_samples_from_model: The graph samples from the model.
        exp_edges_per_node: The expected number of edges per node.
        graph_type: The type of graph (ER or SF).
    """
    all_graphs = torch.from_numpy(all_graphs)
    num_nodes = all_graphs.shape[-1]
    log_marginal_likelihood_true_posterior = torch.from_numpy(log_marginal_likelihood_true_posterior)
    graph_samples_from_model = torch.from_numpy(graph_samples_from_model)
    if graph_type.lower() == "er":
        def log_prob_prior_graph(g):
            """
            Computes log p(G) up the normalization constant
            Args:
                g: Bxdxd
            Returns:
                unnormalized log probability of :math:`G`
            """
            N = num_nodes* (num_nodes - 1) / 2.0
            E = torch.sum(g, dim=(-1,-2))
            p = torch.tensor(exp_edges_per_node*num_nodes/((num_nodes * (num_nodes - 1.0)) / 2))
            return E * torch.log(p) + (N - E) * torch.log1p(p)
    elif graph_type.lower() == "sf":
        def log_prob_prior_graph(g):
            """
            Computes log p(G) up the normalization constant
            Args:
                g: Bxdxd
            Returns:
                unnormalized log probability of :math:`G`
            """
            soft_indegree = g.sum(-2)
            return torch.sum(-3 * torch.log(1 + soft_indegree),-1)
    else:
        raise NotImplementedError

    log_prob_graph_prior_true_posterior = log_prob_prior_graph(all_graphs)
    marginal_posterior_true = torch.distributions.Categorical(logits=log_marginal_likelihood_true_posterior+log_prob_graph_prior_true_posterior)

    marginal_true_posterior_sample_indexes = marginal_posterior_true.sample(torch.Size([graph_samples_from_model.shape[0]]))
    marginal_true_posterior_samples = all_graphs[marginal_true_posterior_sample_indexes]

    from sklearn.metrics import DistanceMetric
    marginal_tp_np = marginal_true_posterior_samples.view(-1, num_nodes*num_nodes).cpu().numpy()
    graph_samples_from_model = graph_samples_from_model.view(-1, num_nodes*num_nodes).cpu().numpy()

    dist = DistanceMetric.get_metric('hamming')
    k_xx = 1 - dist.pairwise(marginal_tp_np)
    k_yy = 1 - dist.pairwise(graph_samples_from_model)
    k_xy = 1 - dist.pairwise(graph_samples_from_model, marginal_tp_np)

    selector = (np.ones_like(k_xx) - np.eye(k_xx.shape[-1])).astype(bool)
    return np.sqrt(k_xx[selector].mean() + k_yy[selector].mean() - 2*k_xy[selector].mean())

def mmd(x, y, kernel):

    n, d = x.shape
    m, d2 = y.shape
    assert d == d2, "x and y must have same dimensionality"
    k_x = kernel(x, x)
    k_y = kernel(y, y)
    k_xy = kernel(x, y)
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd

def i_mmd(sampler, held_out_interventions):
    """
    Computes the Interventional Maximum Mean Discrepancy of the model.
    Args:
        sampler: The sampler function of the model.
        held_out_interventions: The held out interventions.
    """
    # Dags x T x B x N x D
    mmds = []
    for i in range(len((held_out_interventions))):
        intervention = held_out_interventions[i]
        gt_samples = intervention['samples']
        #intervention_node = torch.zeros(gt_samples.shape[-1]).to(torch.bool)
        intervention_node = torch.tensor(intervention["node"]).unsqueeze(0)
        intervention_value = torch.tensor(intervention['value']).unsqueeze(0)
        _model_samples = sampler(Nsamples=20000,samples_per_graph=200,intervention_idxs=intervention_node,intervention_values=intervention_value)
        for dag in range(len(_model_samples)):
            model_samples = _model_samples[dag].cpu()

            # via https://torchdrift.org/notebooks/note_on_mmd.html
            # Gretton et. al. recommend to set the parameter to the median
            # distance between points.
            dists = torch.cdist(model_samples, torch.Tensor(gt_samples))
            sigma = (dists.median() / 2).item()
            kernel = RBF(length_scale=sigma)
            mmds.append(mmd(model_samples, gt_samples, kernel))

    return np.array(mmds).mean()

def logmeanexp(A, axis):
    return logsumexp(A, axis=axis) - np.log(A.shape[axis])

def i_nll(log_prob, held_out_interventions, device):
    """
    Computes the Interventional Negative Log Likelihood of the model.
    Args:
        log_prob: The log_prob function of the model.
        held_out_interventions: The held out interventions.
        device: The device to use.
    """
    # Dags x T x B x N x D
    log_probs = []
    for i in range(len((held_out_interventions))):
        intervention = held_out_interventions[i]
        gt_samples = intervention['samples']
        #intervention_node = torch.zeros(gt_samples.shape[-1]).to(torch.bool)
        intervention_node = torch.tensor(intervention["node"]).unsqueeze(0)
        intervention_value = torch.tensor(intervention['value']).unsqueeze(0)
        _log_probs = log_prob(X=torch.from_numpy(gt_samples).to(device), Ngraphs=100,intervention_idxs=intervention_node,intervention_values=intervention_value).squeeze()
        log_probs.append(-logmeanexp(logmeanexp(_log_probs,1),0))

    return np.array(log_probs).mean()

def nll(log_prob, held_out_data):
        """
        Computes the Negative Log Likelihood of the model.
        Args:
            log_prob: The log_prob function of the model.
            held_out_data: The held out data.
        """
        _log_probs = log_prob(X=held_out_data, Ngraphs=100).squeeze()
        return -logmeanexp(logmeanexp(_log_probs,1),0)