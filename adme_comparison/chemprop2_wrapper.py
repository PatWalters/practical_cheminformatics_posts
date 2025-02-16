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


class ChemPropRDKitWrapper:
    def __init__(self, y_name):
        self.y_name = y_name

    def validate(self, train, test):
        pred = run_chemprop_rdkit(train, test, self.y_name, num_epochs=50)
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


def calc_descriptors(smi_list):
    generator = uru.RDKitDescriptors(hide_progress=True)
    return generator.pandas_smiles(smi_list).values.tolist()


def run_chemprop_rdkit(train, test, y_col, num_epochs=20, accerlerator="cpu"):
    # generate the validation set
    train, val = train_test_split(train, test_size=len(test))
    # calculate the descriptors
    train['feat'] = calc_descriptors(train.SMILES.values)
    val['feat'] = calc_descriptors(val.SMILES.values)
    test['feat'] = calc_descriptors(test.SMILES.values)
    # convert the data to the format needed for chemprop
    cols = ["SMILES", y_col, 'feat']
    train_pt = [data.MoleculeDatapoint.from_smi(smi, [y], x_d=np.array(X_d)) for smi, y, X_d in train[cols].values]
    val_pt = [data.MoleculeDatapoint.from_smi(smi, [y], x_d=np.array(X_d)) for smi, y, X_d in val[cols].values]
    test_pt = [data.MoleculeDatapoint.from_smi(smi, [y], x_d=np.array(X_d)) for smi, y, X_d in test[cols].values]
    # create the featurizer
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    # create the datasets
    train_dset = data.MoleculeDataset(train_pt, featurizer)
    scaler = train_dset.normalize_targets()
    extra_datapoint_descriptors_scaler = train_dset.normalize_inputs("X_d")

    val_dset = data.MoleculeDataset(val_pt, featurizer)
    val_dset.normalize_targets(scaler)
    val_dset.normalize_inputs("X_d", extra_datapoint_descriptors_scaler)

    test_dset = data.MoleculeDataset(test_pt, featurizer)
    # create the dataloaders
    num_workers = 0
    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)
    # create the model
    num_extra_features = len(train.feat.values[0])
    mp = nn.BondMessagePassing()
    ffn_input_dim = mp.output_dim + num_extra_features
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        ffn = nn.RegressionFFN(input_dim=ffn_input_dim, output_transform=output_transform)

    batch_norm = True
    agg = nn.MeanAggregation()
    metric_list = [nn.metrics.RMSEMetric()]
    X_d_transform = nn.ScaleTransform.from_standard_scaler(extra_datapoint_descriptors_scaler)
    mpnn = models.MPNN(mp, agg, ffn, metrics=metric_list, X_d_transform=X_d_transform, batch_norm=batch_norm)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        with DisableLogger():
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                accelerator=accerlerator,
                devices=1,
                max_epochs=num_epochs,  # number of epochs to train for
            )
            trainer.fit(mpnn, train_loader, val_loader)
            pred_tensor = trainer.predict(mpnn, test_loader)
    pred = np.array(list(itertools.chain(*pred_tensor))).flatten()
    return pred


if __name__ == "__main__":
    df = pd.read_csv("https://raw.githubusercontent.com/PatWalters/datafiles/refs/heads/main/biogen_logS.csv")
    y_col = "logS"
    train, test = train_test_split(df)
    model = ChemPropWrapper(y_col)
    pred = model.validate(train, test)
    print(pred)
