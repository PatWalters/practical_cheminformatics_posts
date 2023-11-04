### Comparing Classification Models - You're Probably Doing It Wrong

BSEP_classification_ChemProp_LightGBM.csv - combined results from LightGBM and ChemProp   
ML_Classification_Comparison.ipynb - main demo notebook for the blog post  

extra_notebooks/  
distribution_experiments.ipynb - some plots for the blog post  
friedman_example.ipynb - an explanation of Friedman's test  
plot_t-test_and_wilcoxon.ipynb - more plots for the blog post  

model_building/  
Setup  
generate_validation_data.ipynb - split data into training and test sets
helper.py - utility functions for scaffold splits

Build Models  
build_lightgbm_model.ipynb - build LightGBM models. This is pretty quick
run_chemprop_bsep.py - build ChemProp models. This takes ~12 hrs

Data Integration  
analyze_bsep_models.ipynb - integrate data from LightGBM and ChemProp

Datafiles    
MTL_2_input_BSEP_herg_BBB_PDK_HIV.csv - original data provided by authors  
lightgbm_classifciation_results.csv - per molecule results from LightGBM    
lgbm_result.csv - summary statistics for LightGBM    
BSEP_classification_preds.csv - ChemProp predictions    
BSEP_classification_ChemProp_LightGBM.csv - combined results from LightGBM and ChemProp   
