# ExpansionRx Data Exploration

This repository contains an exploratory data analysis (EDA) notebook and utilities for analyzing the OpenADMET ExpansionRx Blind Challenge Dataset. The project uses [marimo](https://marimo.readthedocs.io/) for interactive data science notebooks and [BitBIRCH-Lean](https://github.com/mqcomplab/bblean) for efficient molecular clustering.

## Overview

The main notebook (`openadmet_expansion_exploration.py`) performs comprehensive EDA on the OpenADMET ExpansionRx dataset, including:

- Data loading and preprocessing
- Log transformation of assay values
- Chemical fingerprint generation
- Molecular clustering using BitBIRCH-Lean
- Principal Component Analysis (PCA) visualization
- Activity cliff identification using Structure Activity Landscape Index (SALI)
- Train/test set similarity analysis

## Files

- `openadmet_expansion_exploration.py` - Main marimo notebook for EDA
- `bblean_cluster.py` - Utility function for clustering molecules using BitBIRCH-Lean
- `openadmet_expansion_exploration.html` - Static HTML export of the notebook

## Requirements

The notebook uses inline dependency management (PEP 723) and will automatically install dependencies when run with marimo's `--sandbox` flag. Main dependencies include:

- marimo
- pandas
- numpy
- rdkit
- bblean (from GitHub)
- altair
- matplotlib
- seaborn
- scikit-learn
- useful-rdkit-utils
- datasets (HuggingFace)

## Usage

### Running the Marimo Notebook

1. Install marimo:
   ```bash
   pip install uv marimo
   ```

2. Run the notebook in sandbox mode (automatically installs dependencies):
   ```bash
   marimo edit openadmet_expansion_exploration.py --sandbox
   ```

3. Alternatively, install dependencies manually and run:
   ```bash
   marimo edit openadmet_expansion_exploration.py
   ```

### Using the Clustering Function

The `bblean_cluster` function can be imported and used independently:

```python
from bblean_cluster import bblean_cluster
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Cluster molecules
cluster_assignments, bb_tree = bblean_cluster(
    dataframe=df,
    smiles_column="SMILES",
    n_features=2048,
    fp_kind="ecfp4",
    verbose=True
)

# Add cluster assignments to dataframe
df['cluster'] = cluster_assignments
```

## Key Features

- **Interactive Visualizations**: The marimo notebook includes interactive Altair charts and widgets
- **Efficient Clustering**: Uses BitBIRCH-Lean for fast clustering of large molecular datasets
- **Activity Cliff Detection**: Identifies structure-activity relationships using SALI
- **Chemical Space Visualization**: PCA-based visualization of molecular diversity

## Credits

- BitBIRCH-Lean implementation by the [Miranda-Quintana research group](https://quintana.chem.ufl.edu/)
- Original BitBIRCH-Lean example code: [bitbirch_best_practices.ipynb](https://github.com/mqcomplab/bblean/blob/main/examples/bitbirch_best_practices.ipynb)

## License

See the main repository license.
