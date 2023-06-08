#!/usr/bin/env python

import sys
from rdkit import Chem
from rdkit.Chem import Descriptors
from logd_predictor import LogDPredictor

class SFIPredictor:
    def __init__(self, model_file_name=None, bins=None):
        self.bins = bins or [30, 200]
        self.logd_predictor = LogDPredictor(model_file_name)

    def predict(self,mol):
        logd = self.logd_predictor.predict(mol)
        num_aromatic = Descriptors.NumAromaticRings(mol)
        return logd + num_aromatic

    def predict_smiles(self,smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return self.predict(mol)
        else:
            return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{sys.argv[0]} infile.smi")
        sys.exit(0)

    sfi_predictor = SFIPredictor()
    suppl = Chem.SmilesMolSupplier(sys.argv[1],titleLine=False)
    for mol in suppl:
        pred_sfi = sfi_predictor.predict(mol)
        print(Chem.MolToSmiles(mol),mol.GetProp("_Name"),f"{pred_sfi:.2f}")
        
