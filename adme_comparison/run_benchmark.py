#!/usr/bin/env python

import polaris as po
import useful_rdkit_utils as uru

from chemprop2_wrapper import ChemPropWrapper, ChemPropRDKitWrapper
from lgbm_wrapper import LGBMMorganCountWrapper, LGBMPropWrapper, LGBMOsmordredWrapper
from tabpfn_wrapper import TabPFNWrapper
from xgboost_wrapper import XGBOsmordredWrapper, XGBPropWrapper, XGBMorganCountWrapper

def main() -> None:
    """
    Load the dataset, preprocess the data, and perform cross-validation with different models and clustering methods.

    :return: None
    :rtype: None
    """
    ds = po.load_dataset("biogen/adme-fang-v1")
    ref_df = ds.table
    ref_df.rename(columns={"smiles": "SMILES"}, inplace=True)
    y_list = [x for x in ref_df.columns if x.startswith("LOG")]
    for y in y_list:
        df = ref_df.dropna(subset=[y]).copy()
        model_list = [("chemprop", ChemPropWrapper),
                      ("lgbm_morgan", LGBMMorganCountWrapper), ("lgbm_prop", LGBMPropWrapper),
                      ("lgbm_osm", LGBMOsmordredWrapper),
                      ("xgb_morgan", XGBMorganCountWrapper), ("xgb_prop", XGBPropWrapper),
                      ("xgb_osm", XGBOsmordredWrapper),
                      ("tabpfn", TabPFNWrapper)]
        group_list = [("random", uru.get_random_clusters), ("butina", uru.get_butina_clusters)]
        result_df = uru.cross_validate(df, model_list, y, group_list)
        result_df.to_csv(f"{y}_results.csv", index=False)
    

if __name__ == "__main__":
    main()
    
