This code accompanies my blog post ["AI in Drug Discovery - Please Stop Finishing in the Bathtub"](https://patwalters.github.io/Please-Stop-Fishing/)

## Structure

*   `analyze_kinases.py`: Main script to fetch sequences and calculate similarities.
*   `01_dud-e_property_distributions.ipynb`: Analysis of property distributions.
*   `02_dude_kinase_models.ipynb`: Machine learning models for kinase targets.
*   `03_dude_tsne.ipynb`: t-SNE visualization of the chemical space.
*   `dud_kinase.txt`: List of kinase gene names used for analysis.
*   `dud-e.target_class.csv`: Target classification data.

## Installation

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Analyze Kinases

Run the `analyze_kinases.py` script to fetch sequences and calculate the similarity matrix:

```bash
python analyze_kinases.py
```

This will generate:
*   `kinase_sequences.csv`: Contains the retrieved UniProt sequences.
*   `kinase_similarities.csv`: A pairwise sequence similarity matrix (percent identity).

You can also specify input/output files (optional):

```bash
python analyze_kinases.py --input dud_kinase.txt --output-seq kinase_sequences.csv --output-sim kinase_similarities.csv
```

### Notebooks

Run the Jupyter notebooks to view the analysis and models:

```bash
jupyter notebook
```
