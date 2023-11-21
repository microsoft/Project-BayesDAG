import os
import pickle as pkl
import subprocess

import numpy as np

# How to generate dataset in correct format
# Download https://github.com/kurowasan/GraN-DAG/blob/master/data/syntren_p20.zip
# Copy the 20 files (DAGi.py, datai.py, for i=1,...,10) to the directory where this file is.
# Run python generate.py. This will create all the datasets (standardized and non standardized, fully-observed and
# 30% of training set MCAR)


def save_data(savedir, adj_matrix, X, num_samples_train, graph_args):
    np.savetxt(os.path.join(savedir, "adj_matrix.csv"), adj_matrix, delimiter=",", fmt="%i")
    np.savetxt(os.path.join(savedir, "all.csv"), X, delimiter=",")
    np.savetxt(os.path.join(savedir, "train.csv"), X[:num_samples_train, :], delimiter=",")
    np.savetxt(os.path.join(savedir, "test.csv"), X[num_samples_train:, :], delimiter=",")
    with open(os.path.join(savedir, "held_out_interventions.pkl"), "wb") as b:
        pkl.dump(None, b)
    with open(os.path.join(savedir, "true_posterior.pkl"), "wb") as b:
        pkl.dump(None, b)
    with open(os.path.join(savedir, "graph_args.pkl"), "wb") as b:
        pkl.dump(graph_args, b)


def main():
    num_samples_train = 400
    if not os.path.exists("syntren/data1.npy"):
        try:
            subprocess.run(
                ["wget https://github.com/kurowasan/GraN-DAG/raw/master/data/syntren_p20.zip"], shell=True, check=True
            )
            subprocess.run(["unzip syntren_p20.zip"], shell=True, check=True)
        except subprocess.CalledProcessError:
            print("Failed to download sachs.zip. Please download it manually and place it in the current directory.")
            return
    folder_prefix = "/proj/berzelius-2021-89/users/x_yasan/data/graph_posterior_data/"
    for i, r in enumerate(["seed_1", "seed_2", "seed_3", "seed_4", "seed_5"]):
        index = i + 1
        adj_matrix = np.load(f"syntren/DAG{index}.npy")
        X = np.load(f"syntren/data{index}.npy")  # Shape (500, 20)
        graph_args = {
            "num_variables": X.shape[-1],
            "exp_edges": adj_matrix.sum(),
            "seed": index,
            "graph_type": "syntren",
            "exp_edges_per_node": adj_matrix.sum() / X.shape[-1],
        }
        mean = np.mean(X[:num_samples_train, :], axis=0)
        std = np.std(X[:num_samples_train, :], axis=0)
        X = X - mean  # Always remove mean
        X_std = X / std

        np.random.seed(index)

        # Save nonstd data
        savedir = os.path.join(folder_prefix, f"syntren_{r}")
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X, num_samples_train, graph_args)

        # Save std data
        savedir = os.path.join(folder_prefix, f"syntren_{r}_std")
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X_std, num_samples_train, graph_args)


if __name__ == "__main__":
    main()
