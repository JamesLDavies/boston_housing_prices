"""
Module Docstring
"""
import logging

from sklearn.datasets import load_boston

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _load_data():
    """
    Load data

    Args:
        None

    Returns:
         df (DataFrame): Input boston dataset
    """
    LOGGER.info("Loading input data")
    # Load data and convert to pandas dataframe
    boston_dataset = load_boston()
    df = pd.DataFrame(boston_dataset['data'], columns=boston_dataset['feature_names'])

    # Add target col
    df['target'] = boston_dataset['target']
    return df


def _preprocessing(df):
    """

    :param df:
    :return:
    """
    LOGGER.info("Preprocessing")
    null_df = df.isnull().sum()
    null_df = null_df[null_df > 0]

    if null_df.count() > 0:
        LOGGER.warning("There are NULLs present in the input data:")
        print(null_df)


def _data_exploration(df, target_col):
    """
    Calculate useful statistics
    :param df:
    :return:
    """
    LOGGER.info("Calculating statistics")
    target = df[target_col]
    print(f'Statistics for {target_col} column:')
    print(f'Min {target_col}: {np.amin(target)}')
    print(f'Max {target_col}: {np.amax(target)}')
    print(f'Mean {target_col}: {np.mean(target)}')
    print(f'Median {target_col}: {np.median(target)}')
    print(f'Standard deviation {target_col}: {np.std(target)}')


def run():
    """
    Main ENTRYPOINT function

    Args:
        None

    Returns:
        None
    """
    input_df = _load_data()
    _preprocessing(input_df)
    _data_exploration(input_df, 'target')

if __name__ == "__main__":
    LOGGER.info("Starting...")
    run()
    LOGGER.info("Finished")
