preprocess_data.ipynb - A bit of data preprocessing

ESOL.ipynb - evaluate a solubility dataset using [ESOL](https://pubs.acs.org/doi/10.1021/ci034243x)

GSE_solubility.ipynb - evaluate a solubility dataset using the [General Solubility Equation](https://pubs.acs.org/doi/10.1021/ci000338c)

cross_validate_ML_model.ipynb - train and test a solubility model using a solubility dataset

evaluate_SFI.ipynb - evaluate a solubility dataset using the [Solubility Forecast Index](https://www.sciencedirect.com/science/article/pii/S1359644610001923?via%3Dihub)

literature_solubility_model.ipynb - build a solubility model using literature data from the Therapeutic Data Commons

generate_SIMPD_sets.ipynb - split into training and test sets using the [SIMPD alogrithm](https://github.com/rinikerlab/molecular_time_series) which approximates time splits

I borrowed a lot of code to put this together.  I'd like to thank the authors.
ga_lib_3.py came from [https://github.com/rinikerlab/molecular_time_series](https://github.com/rinikerlab/molecular_time_series)

helper.py came from [https://github.com/chemprop/chemprop](https://github.com/chemprop/chemprop)

The descriptor calculation code in GSE_solubility.ipynb was borrowed from [https://github.com/sutropub/OpenSOL](https://github.com/sutropub/OpenSOL)
