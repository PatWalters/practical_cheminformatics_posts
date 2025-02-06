### Machine Learning Model Comparison

This repo contains scripts and a Jupyter notebook for ML model comparison.  The scripts use 5x5 cross-validation as recommended in [Practically significant method comparison protocols for machine learning in small molecule drug discovery](https://chemrxiv.org/engage/chemrxiv/article-details/672a91bd7be152b1d01a926b) a set of guidelines written by members of the [Polaris](https://polarishub.io/) small molecule steering group. The notebook uses Tukey's Honestly Significant Difference (HSD) test to establish a statistically significant difference between methods. 

### Installation 
```shell
uv venv --python 3.11
source .venv/bin/activate
uv pip install rdkit useful_rdkit_utils scikit-learn chemprop tabpfn lightgbm jupyter statannotations statsmodels 
```

### Running Benchmarks
The script **run_benchmark.py** can be used to run benchmarks.  All that is necessary for the script is a wrapper class that wraps an ML model and supports a **validate** method. The wrapper class is instantiated with the name of the column to be predicted.  The validate method takes dataframes containing training and test sets as input and returns a list of predicted values for the test set. For examples of wrapper classes, see **chemprop_wrapper.py**, **lgbm_wrapper.py**, and **tabpfn_wrapper.py**. If the ML method requires a validation set, this can be created inside the wrapper by further splitting the training set. 

```python
df = pd.read_csv("myfile.csv")
train, test = train_test_split(df)
chemprop_wrapper = ChemPropWrapper("logS")
pred = chemprop_wrapper.validate(train, test)
```

The **cross_validate** method in **run_benchmark.py** comes from the [useful_rdkit_utils](https://github.com/PatWalters/useful_rdkit_utils).  The **cross_validate** function has four required arguments.

- **df** - a dataframe with a SMILES column  
- **model_list** - a list of tuples containing the model name and the wrapper class described above  
- **y_col** - the name of the column with the y value to be predicted  
- **group_list** - a list of group_names and group memberships (e.g. cluster ids), these can be calculated using the functions get_random_clusters, get_scaffold_clusters, get_butina_clusters, and get_umap_clusters in useful_rkdkit_utils.  

```python

y = "logS"
model_list = [("chemprop",ChemPropWrapper),("lgbm_morgan", LGBMMorganCountWrapper),("lgbm_prop",LGBMPropWrapper)]
group_list = [("random", uru.get_random_clusters),("butina", uru.get_butina_clusters)]
result_df = uru.cross_validate(df,model_list,y,group_list)
```

The notebook **analyze_crossval.ipynb** reads a file output by **run_benchmark.py**, makes several useful plots, and uses [Tukey's Honestly Signficant Difference (HSD) Test](https://en.wikipedia.org/wiki/Tukey%27s_range_test) to invalidate the null hypothesis that the means of the methods are the same. 
