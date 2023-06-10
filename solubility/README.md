### Introduction
This is the code to accompany the Practical Cheminformatics post "Getting Real with Molecular Property Prediction".  The code in this repo uses a number of published models to evaluate aqueous solubiliity predictions on the a dataset published by [Fang and coworkers](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00160) from Biogen.  Each of the Jupyter notebooks below uses a different solubility prediction method.  All of the input data and intermediate files are provided so it should be possilble to run any of the notebooks independently. 

### Scripts
preprocess_data.ipynb - A bit of data preprocessing

ESOL.ipynb - evaluate a solubility dataset using [ESOL](https://pubs.acs.org/doi/10.1021/ci034243x)

GSE_solubility.ipynb - evaluate a solubility dataset using the [General Solubility Equation](https://pubs.acs.org/doi/10.1021/ci000338c) together with a melting point model recently published by [Zhu](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00308)

cross_validate_ML_model.ipynb - train and test a solubility model using a Biogen Solubility Dataset published by [Fang](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00160) 

evaluate_SFI.ipynb - evaluate a solubility dataset using the [Solubility Forecast Index](https://www.sciencedirect.com/science/article/pii/S1359644610001923?via%3Dihub)

literature_solubility_model.ipynb - build a solubility model using the AqSolDB dataset from the [Therapeutic Data Commons](https://tdcommons.ai/single_pred_tasks/adme/#solubility-aqsoldb)

generate_SIMPD_sets.ipynb - split into training and test sets using the [SIMPD alogrithm](https://github.com/rinikerlab/molecular_time_series) which approximates time splits

### Acknowledgments 
I borrowed a lot of code to put this together.  I'd like to thank the authors.

ga_lib_3.py came from [https://github.com/rinikerlab/molecular_time_series](https://github.com/rinikerlab/molecular_time_series).   
helper.py came from [https://github.com/chemprop/chemprop](https://github.com/chemprop/chemprop).   
The descriptor calculation code in GSE_solubility.ipynb was borrowed from [https://github.com/sutropub/OpenSOL](https://github.com/sutropub/OpenSOL).   
Routines for bootstrapping confidence intervals were adapted from [https://github.com/openforcefield/protein-ligand-benchmark-livecoms](https://github.com/openforcefield/protein-ligand-benchmark-livecoms).   



