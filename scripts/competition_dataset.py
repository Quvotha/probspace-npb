# -*- coding: utf-8 -*-
import os.path
from typing import Tuple

import numpy as np
import pandas as pd


def load() -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Load training and test set.

    Load and apply following transformation to competition dataset.
    1st, training set is known to have duplicate rows thus drop them.
    2nd, bool columns in training/test set are conveted into integer flag. 
    3rd, "test_data.csv" is known to have missing `pitcher` and `batter` thus they are to be interpolated by official external data.
    Finally, game information is merged into training/test set associated with `gameID`.

    Parameters
    ----------
    train_filepath, test_filepath, game_info_filepath: str
      Filepaths of competition dataset.

    Return
    ------
    dataset: Tuple[pd.DataFrame, pd.DataFrame]
        Dataframe of training/test set transformed.
    '''
    # Load
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'read_only')
    train = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    game_info = pd.read_csv(os.path.join(data_dir, 'game_info.csv'), parse_dates=['startDayTime'])
    official_external_data1 = pd.read_csv(os.path.join(data_dir, 'test_data_improvement.csv'))

    # Remove duplication
    key_columns = [c for c in train.columns if c != 'id']
    train = train.drop_duplicates(subset=key_columns)

    # Bool to int
    base_columns = ['b1', 'b2', 'b3']
    train[base_columns] = train[base_columns].copy() * 1
    test[base_columns] = test[base_columns].copy() * 1

    # Interpolate missing `batter` and `pitcher` in "test_data.csv"
    test[['pitcher', 'batter']] = official_external_data1[['pitcher', 'batter']]

    # Merge game information into training/test set
    game_info.drop(columns=['Unnamed: 0'], inplace=True)
    nrows_train, nrows_test = train.shape[0], test.shape[0]
    train = pd.merge(train, game_info, on='gameID', how='inner')
    test = pd.merge(test, game_info, on='gameID', how='inner')
    assert(train.shape[0] == nrows_train)
    assert(test.shape[0] == nrows_test)

    # Return dataset
    return (train, test)
