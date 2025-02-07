#!/usr/bin/env python

import polaris as po
import useful_rdkit_utils as uru

from chemprop2_wrapper import ChemPropWrapper
from lgbm_wrapper import LGBMMorganCountWrapper
from tabpfn_wrapper import TabPFNWrapper


def main() -> None:
    """
    Load the dataset, preprocess the data, and perform cross-validation with different models and clustering methods.

    :return: None
    :rtype: None
    """
    ds = po.load_dataset("polaris/adme-fang-1")
    df = ds.table
    df.rename(columns={"smiles": "SMILES"}, inplace=True)
    y_list = [x for x in df.columns if x.startswith("LOG")]
    for y in y_list:
        df = df.dropna(subset=[y]).copy()
        model_list = [("chemprop", ChemPropWrapper), ("lgbm_morgan", LGBMMorganCountWrapper), ("tabpfn", TabPFNWrapper)]
        group_list = [("random", uru.get_random_clusters), ("butina", uru.get_butina_clusters)]
        result_df = uru.cross_validate(df, model_list, y, group_list)
        result_df.to_csv(f"{y}_results.csv", index=False)
    

if __name__ == "__main__":
    main()
    
