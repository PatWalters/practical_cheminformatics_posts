import bblean
import pandas as pd
import bblean.similarity as iSIM
import numpy as np

# This code is just a simple wrapper around code from one of the bblean example notebooks.
# The code this is based on is here https://github.com/mqcomplab/bblean/blob/main/examples/bitbirch_best_practices.ipynb

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
) -> tuple[np.ndarray, bblean.BitBirch]:
    """
    Clusters molecules in a DataFrame based on their chemical fingerprints.

    This function uses BitBIRCH-Lean, a memory-efficient clustering algorithm
    for molecular similarity-based clustering. The clustering threshold is
    automatically determined based on the similarity distribution of the dataset.

    Args:
        dataframe: DataFrame containing molecular data with SMILES strings.
        smiles_column: Name of the column containing SMILES strings. Defaults to "SMILES".
        n_features: Number of features for the fingerprint generation. Defaults to 2048.
        fp_kind: Kind of fingerprint to use (e.g., "ecfp4"). Defaults to "ecfp4".
        n_samples_for_std: Number of samples for estimating similarity std dev. Defaults to 50.
        std_multiplier: Multiplier for std dev to determine the clustering threshold.
            Higher values create fewer, larger clusters. Defaults to 4.
        branching_factor: Branching factor for the BitBirch algorithm. Defaults to 50.
        recluster_iterations: Number of iterations for the re-clustering step. Defaults to 5.
        verbose: If True, prints clustering progress and information. Defaults to False.

    Returns:
        A tuple containing:
        - numpy.ndarray: Array of cluster IDs corresponding to rows in the input dataframe.
          Each element indicates which cluster the corresponding molecule belongs to.
        - bblean.BitBirch: The fitted BitBirch tree object that can be used for
          additional operations like predicting cluster assignments for new molecules.

    Example:
        >>> import pandas as pd
        >>> from bblean_cluster import bblean_cluster
        >>> df = pd.DataFrame({"SMILES": ["CCO", "CCCO", "CC"]})
        >>> cluster_ids, bb_tree = bblean_cluster(df)
        >>> df['cluster'] = cluster_ids
    """
    # Validate inputs
    if smiles_column not in dataframe.columns:
        raise ValueError(f"Column '{smiles_column}' not found in dataframe. Available columns: {list(dataframe.columns)}")
    
    if len(dataframe) == 0:
        raise ValueError("DataFrame is empty. Cannot cluster empty dataset.")
    
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

    # Return cluster assignments and the fitted BitBirch tree
    return bb_tree.get_assignments(), bb_tree
