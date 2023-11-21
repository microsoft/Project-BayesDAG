import os

import numpy as np
import pickle as pkl
import subprocess

# How to generate dataset in correct format
# 1. Download https://github.com/kurowasan/GraN-DAG/blob/master/data/sachs.zip
# 2. Extract, and move the files from the continue sub-directory (DAG1.npy, data1.npy) and place them
# in the directory where this file is.
# 3. Run python generate.py. This will create all the datasets (standardized and non standardized,
# fully-observed and 30% of training set MCAR)


def save_data(savedir, adj_matrix, X, indices_train, indices_test, graph_args):
    np.savetxt(os.path.join(savedir, "adj_matrix.csv"), adj_matrix, delimiter=",", fmt="%i")
    np.savetxt(os.path.join(savedir, "all.csv"), X, delimiter=",")
    np.savetxt(os.path.join(savedir, "train.csv"), X[indices_train, :], delimiter=",")
    np.savetxt(os.path.join(savedir, "test.csv"), X[indices_test, :], delimiter=",")
    with open(os.path.join(savedir, "held_out_interventions.pkl"), "wb") as b:
        pkl.dump(None, b)
    with open(os.path.join(savedir, "true_posterior.pkl"), "wb") as b:
        pkl.dump(None, b)
    with open(os.path.join(savedir, "graph_args.pkl"), "wb") as b:
        pkl.dump(graph_args, b)


def main():
    np.random.seed(1)
    if not os.path.exists("sachs/continuous/data1.npy"):
        try:
            subprocess.run(
                ["wget https://github.com/kurowasan/GraN-DAG/raw/master/data/sachs.zip"], shell=True, check=True
            )
            subprocess.run(["unzip sachs.zip"], shell=True, check=True)
        except subprocess.CalledProcessError:
            print("Failed to download sachs.zip. Please download it manually and place it in the current directory.")
            return
    adj_matrix = np.load("sachs/continuous/DAG1.npy")
    print(adj_matrix.shape)

    X = np.load("sachs/continuous/data1.npy")  # Shape (500, 20)
    print(X.shape)
    folder_prefix = "/proj/berzelius-2021-89/users/x_yasan/data/graph_posterior_data/"
    num_samples_train = 800
    for i in range(5):
        graph_args = {
            "num_variables": X.shape[-1],
            "exp_edges": adj_matrix.sum(),
            "seed": i + 1,
            "graph_type": "sachs",
            "exp_edges_per_node": adj_matrix.sum() / X.shape[-1],
        }
        np.random.seed(i + 1)
        indices = np.arange(X.shape[0])
        indices = np.random.permutation(indices)
        indices_train = indices[:num_samples_train]
        indices_test = indices[num_samples_train:]

        mean = np.mean(X[:num_samples_train, :], axis=0)
        std = np.std(X[:num_samples_train, :], axis=0)
        X = X - mean  # Always remove mean
        X_std = X / std

        # Save data and std
        savedir = os.path.join(folder_prefix, f"sachs_protein_cells_seed_{i + 1}")
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X, indices_train, indices_test, graph_args)

        # Save std data
        savedir = os.path.join(folder_prefix, f"sachs_protein_cells_seed_{i + 1}_std")
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X_std, indices_train, indices_test, graph_args)


if __name__ == "__main__":
    main()
