#!/usr/bin/env python

from tabpfn import TabPFNRegressor
from rdkit import Chem
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import useful_rdkit_utils as uru
import warnings

class TabPFNWrapper:
    def __init__(self, y_col):
        self.model = TabPFNRegressor()
        self.y_col = y_col
        self.rdkit_desc = uru.RDKitDescriptors(hide_progress=True)
        self.desc_name = 'desc'

    def fit(self, train):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            train[self.desc_name] = self.rdkit_desc.pandas_smiles(train.SMILES).values.tolist()
            self.model.fit(np.stack(train[self.desc_name]),train[self.y_col])

    def predict(self, test):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            test[self.desc_name] = self.rdkit_desc.pandas_smiles(test.SMILES).values.tolist()
            pred = self.model.predict(np.stack(np.stack(test[self.desc_name])))
        return pred

    def validate(self, train, test):
        self.fit(train)
        pred = self.predict(test)
        return pred

def main():
    df = pd.read_csv("https://raw.githubusercontent.com/PatWalters/datafiles/refs/heads/main/biogen_logS.csv")
    train, test = train_test_split(df)
    lgbm_wrapper = TabPFNWrapper("logS")
    pred = lgbm_wrapper.validate(train, test)
    print(pred)

if __name__ == "__main__":
    main()
