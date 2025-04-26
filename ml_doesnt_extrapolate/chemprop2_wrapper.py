#!/usr/bin/env python

import itertools
import logging
import warnings

import numpy as np
import useful_rdkit_utils as uru
from chemprop import data, featurizers, nn, models
from lightning import pytorch as pl
from sklearn.model_selection import train_test_split
import pandas as pd


class ChemPropWrapper:
    def __init__(self, y_name):
        self.y_name = y_name

    def validate(self, train, test):
        pred = run_chemprop(train, test, self.y_name, num_epochs=20, accelerator='cpu')
        return pred


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def run_chemprop(train, test, y_col, num_epochs=20, accelerator="cpu"):
    # Generate the validation set
    train, val = train_test_split(train, test_size=len(test))
    # Convert data to MoleculeDatapoints
    train_pt = [data.MoleculeDatapoint.from_smi(smi, [y]) for smi, y in train[["SMILES", y_col]].values]
    val_pt = [data.MoleculeDatapoint.from_smi(smi, [y]) for smi, y in val[["SMILES", y_col]].values]
    test_pt = [data.MoleculeDatapoint.from_smi(smi, [y]) for smi, y in test[["SMILES", y_col]].values]
    # Instantiate the featurizer
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    # Create datasets and scalers
    train_dset = data.MoleculeDataset(train_pt, featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_pt, featurizer)
    val_dset.normalize_targets(scaler)

    test_dset = data.MoleculeDataset(test_pt, featurizer)
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    # Generate data loaders
    num_workers = 0
    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)
    # Create the FFNN
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        ffn_input_dim = mp.output_dim
        ffn = nn.RegressionFFN(input_dim=ffn_input_dim, output_transform=output_transform)
    # Create the MPNN
    batch_norm = True
    metric_list = [nn.metrics.RMSE()]
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
    # Train the model
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        with DisableLogger():
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                accelerator=accelerator,
                devices=1,
                max_epochs=num_epochs,  # number of epochs to train for
            )
            trainer.fit(mpnn, train_loader, val_loader)
            pred_tensor = trainer.predict(mpnn, test_loader)
    # Predict on the test set
    pred = np.array(list(itertools.chain(*pred_tensor))).flatten()
    return pred


