import numpy as np
import argparse
from .data_utils import gen_dataset, get_name, save_data

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_folder", type=str, default="data/")
    argparser.add_argument("--num_samples_train", type=int, default=5000)
    argparser.add_argument("--num_samples_test", type=int, default=1000)
    argparser.add_argument("--noise_mult_factor", type=float, default=1)   
    argparser.add_argument("--N_interventions", type=int, default=1)
    argparser.add_argument("--num_nodes", type=int, default=70)
    argparser.add_argument("--expected_edges_per_node", type=int, default=2)
    argparser.add_argument("--tune", type=bool, default=False)
    argparser.add_argument("--n_seeds", type=int, default=30)
    argparser.add_argument("--sem_type", type=str, default="mlp")
    argparser.add_argument("--noise_type", type=str, default="unequal")
    
    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    num_samples_train = args.num_samples_train
    num_samples_test = args.num_samples_test
    noise_mult_factor = args.noise_mult_factor
    expected_num_latent_confounders = 0
    N_interventions = args.N_interventions
    generate_references = True
    discrete_dims_list = None

    max_parent_depth = 2
    use_quantile_references = True

    partial_ratio = 0.3

    dataset_folder = args.dataset_folder

    is_tune = args.tune
    if is_tune:
        N_seeds = 5
        start = 0
    else:
        N_seeds = args.n_seeds
        start = 5
    sem_noise_pairs = [(args.sem_type, args.noise_type)]

    graph_types = ["ER", "SF"]
    num_nodes = [args.num_nodes]

    expected_edges_per_node = [args.expected_edges_per_node]

    for base_seed in range(start, N_seeds+start):
        for graph_type in graph_types:
            for N in num_nodes:
                for EPN in expected_edges_per_node:
                    E = int(EPN * N)
                    for sem_type, noise_type in sem_noise_pairs:

                        if isinstance(noise_type, float) and noise_type < 0.5:
                            noise_mult_factor_ = 10
                        else:
                            noise_mult_factor_ = noise_mult_factor

                            if expected_num_latent_confounders > 0:
                                name = "latent_" + get_name(graph_type, N, E, sem_type, noise_type, base_seed)
                            else:
                                name = get_name(graph_type, N, E, sem_type, noise_type, base_seed, is_tune=is_tune)
                            print(name)
                            (
                                X_train_test,
                                X_train,
                                X_test,
                                directed_matrix,
                                held_out_interventions,
                                true_posterior,
                                graph_args
                            ) = gen_dataset(
                                base_seed,
                                num_samples_train,
                                num_samples_test,
                                graph_type,
                                N,
                                E,
                                sem_type,
                                noise_type,
                                N_interventions=N_interventions,
                                generate_references=generate_references,
                                max_parent_depth=max_parent_depth,
                                use_quantile_references=use_quantile_references,
                                adj_matrix=None,
                                noise_mult_factor=noise_mult_factor,
                                discrete_dims_list=discrete_dims_list,
                                discrete_temperature=None,
                                expected_num_latent_confounders=expected_num_latent_confounders,
                            )

                            print(
                                X_train_test.shape,
                                X_train.shape,
                                X_test.shape,
                                directed_matrix.shape,
                            )

                            save_data(
                                dataset_folder,
                                name,
                                directed_matrix,
                                X_train_test,
                                X_train,
                                X_test,
                                held_out_interventions,
                                true_posterior,
                                graph_args,
                            )
                            