#!/usr/bin/env python

import os
from tqdm.auto import tqdm
import chemprop

base_path = "data/BSEP"
max_folds = 1
for idx in tqdm(range(0,max_folds)):
    for split in ["random","scaffold"]:
        for task in ["ST","MT"]:
            if task == "ST":
                data_path = f'{base_path}{idx:03d}/{split}_train_{idx:03d}.csv'
                separate_val_path = f'{base_path}{idx:03d}/{split}_val_{idx:03d}.csv'
                separate_test_path = f'{base_path}{idx:03d}/{split}_test_{idx:03d}.csv'
            else:
                data_path = f'{base_path}{idx:03d}/{sbplit}_mt_train_{idx:03d}.csv'
                separate_val_path = f'{base_path}{idx:03d}/{split}_mt_val_{idx:03d}.csv'
                separate_test_path = f'{base_path}{idx:03d}/{split}_mt_test_{idx:03d}.csv'

            save_dir = f'{base_path}{idx:03d}/{split}_result_{task}'

            if os.path.exists(save_dir):
                os.rmdir(save_dir)

            arguments = [
                '--data_path', data_path,
                '--separate_test_path',separate_test_path,
                '--separate_val_path',separate_val_path,
                '--num_folds', '1',
                '--epochs','30',
                '--ensemble_size','10',
                '--extra_metrics','prc-auc',
                '--ignore_columns', 'Name',
                '--smiles_columns','SMILES',
                '--quiet',
                '--dataset_type', 'classification',
                '--save_dir', save_dir,
                '--save_preds'
            ]
            args = chemprop.args.TrainArgs().parse_args(arguments)
            mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
