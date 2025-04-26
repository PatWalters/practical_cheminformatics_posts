#!/usr/bin/env python

from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import useful_rdkit_utils as uru
import warnings


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

