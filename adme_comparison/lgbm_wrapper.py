#!/usr/bin/env python

from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import useful_rdkit_utils as uru
from calc_osmordred import calc_osmordred, find_finite_descriptors
import warnings


class LGBMOsmordredWrapper:
    def __init__(self, y_col):
        self.lgbm = LGBMRegressor()
        self.y_col = y_col
        self.desc_name = 'desc'
        self.good_cols = None

    def fit(self, train):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.lgbm.fit(np.stack(train[self.desc_name]),train[self.y_col])

    def predict(self, test):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pred = self.lgbm.predict(np.stack(np.stack(test[self.desc_name])))
        return pred

    def validate(self, train, test):
        all_df = pd.concat([train,test]).copy()
        all_df[self.desc_name] = all_df.SMILES.apply(calc_osmordred)
        self.good_cols = find_finite_descriptors(all_df,"desc")
        all_df[self.desc_name] = np.stack(all_df[self.desc_name].values)[:,self.good_cols].tolist()
        train = all_df.head(len(train))
        test = all_df.tail(len(test))
        self.fit(train)
        pred = self.predict(test)
        return pred


class LGBMPropWrapper:
    def __init__(self, y_col):
        self.lgbm = LGBMRegressor(verbose=-1)
        self.y_col = y_col
        self.rdkit_desc = uru.RDKitDescriptors(hide_progress=True)
        self.desc_name = 'desc'

    def fit(self, train):
        train[self.desc_name] = self.rdkit_desc.pandas_smiles(train.SMILES).values.tolist()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.lgbm.fit(np.stack(train[self.desc_name]),train[self.y_col])

    def predict(self, test):
        test[self.desc_name] = self.rdkit_desc.pandas_smiles(test.SMILES).values.tolist()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pred = self.lgbm.predict(np.stack(np.stack(test[self.desc_name])))
        return pred

    def validate(self, train, test):
        self.fit(train)
        pred = self.predict(test)
        return pred


class LGBMMorganCountWrapper:
    def __init__(self, y_col):
        self.lgbm = LGBMRegressor(verbose=-1)
        self.y_col = y_col
        self.fp_name = "fp"
        self.fg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def fit(self, train):
        train['mol'] = train.SMILES.apply(Chem.MolFromSmiles)
        train[self.fp_name] = train.mol.apply(self.fg.GetCountFingerprintAsNumPy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.lgbm.fit(np.stack(train.fp),train[self.y_col])

    def predict(self, test):
        test['mol'] = test.SMILES.apply(Chem.MolFromSmiles)
        test[self.fp_name] = test.mol.apply(self.fg.GetCountFingerprintAsNumPy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pred = self.lgbm.predict(np.stack(np.stack(test[self.fp_name])))
        return pred

    def validate(self, train, test):
        self.fit(train)
        pred = self.predict(test)
        return pred



def main():
    #df = pd.read_csv("https://raw.githubusercontent.com/PatWalters/datafiles/refs/heads/main/biogen_logS.csv")
    df = pd.read_csv("biogen_logS.csv")
    train, test = train_test_split(df)
    lgbm_wrapper = LGBMOsmordredWrapper("logS")
    pred = lgbm_wrapper.validate(train, test)
    print(pred)

if __name__ == "__main__":
    main()
