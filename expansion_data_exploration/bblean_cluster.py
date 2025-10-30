import bblean
import pandas as pd
import bblean.similarity as iSIM
import numpy as np

# This code is just a simple wrapper around code from one of the bblead example notebooks.
# The code I stole this from is here https://github.com/mqcomplab/bblean/blob/main/examples/bitbirch_best_practices.ipynb

def bblean_cluster(
    dataframe: pd.DataFrame,
    smiles_column: str = "SMILES",
    n_features: int = 2048,
    fp_kind: str = "ecfp4",
    n_samples_for_std: int = 50,
    std_multiplier: int = 4,
    branching_factor: int = 50,
    recluster_iterations: int = 5,
    verbose: bool = False,
) -> tuple[pd.DataFrame, bblean.BitBirch]:
    """
    Clusters molecules in a DataFrame based on their chemical fingerprints.

    Args:
        dataframe: DataFrame containing molecular data.
        smiles_column: Name of the column with SMILES strings.
        n_features: Number of features for the fingerprint generation.
        fp_kind: Kind of fingerprint to use (e.g., "ecfp4").
        n_samples_for_std: Number of samples for estimating similarity std dev.
        std_multiplier: Multiplier for std dev to determine the clustering threshold.
        branching_factor: Branching factor for the BitBirch algorithm.
        recluster_iterations: Number of iterations for the re-clustering step.
        verbose: If True, prints clustering progress and information.

    Returns:
        A tuple containing:
        - A list of cluster ids corresponding to rows in the input dataframe
        - The fitted BitBirch tree object.
    """
    df_clustered = dataframe.copy()

    # Generate fingerprints
    fps = bblean.fps_from_smiles(
        df_clustered[smiles_column],
        pack=True,
        n_features=n_features,
        kind=fp_kind
    )

    # Estimate an optimal clustering threshold
    average_sim = iSIM.jt_isim_packed(fps)
    representative_samples = iSIM.jt_stratified_sampling(
        fps, n_samples=n_samples_for_std
    )
    sim_matrix = iSIM.jt_sim_matrix_packed(fps[representative_samples])
    # Exclude diagonal elements (self-similarity)
    sim_matrix = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
    _, std = np.mean(sim_matrix), np.std(sim_matrix)
    optimal_threshold = average_sim + std_multiplier * std

    # Initialize and fit the clustering model
    bb_tree = bblean.BitBirch(
        branching_factor=branching_factor,
        threshold=optimal_threshold,
        merge_criterion="diameter"
    )
    bb_tree.fit(fps)

    if verbose:
        clusters = bb_tree.get_cluster_mol_ids()
        print("Number of singletons before reclustering:", sum(1 for c in clusters if len(c) == 1))

    # Refine clusters
    bb_tree.recluster_inplace(
        iterations=recluster_iterations,
        extra_threshold=std,
        shuffle=False,
        verbose=verbose
    )

    # Assign clusters to the DataFrame

    return bb_tree.get_assignments(), bb_tree
