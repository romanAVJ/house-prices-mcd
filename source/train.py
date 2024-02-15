"""
Training Pipeline for Iowa Housing Dataset.

This script reads the clean housing dataset to train a machine learning model
to predict house prices using the CatBoost Regressor.

Author: ravj
Date: Mon Feb 12 19:49:11 2023
"""
# imports ####
import logging
import os
import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
from utils import evaluate_model


# parameters ####
# Logging configuration
logging.basicConfig(level=logging.INFO)

# Directory to save model
DIR_MODEL = '../models/vanilla_catboost'
os.makedirs(DIR_MODEL, exist_ok=True)

# cols for training
COLS_NUMERIC = [
    'CurrentAge', 'GrLivArea', 'LotArea', 'OverallCond', 'OverallQual'
]
COLS_CATEGORICAL = ['MSZoning']
COLS_TO_STAY = COLS_NUMERIC + COLS_CATEGORICAL + ['SalePrice']

# monotonic contraints
MONOTONE_CONSTRAINS = "GrLivArea:1,OverallQual:1,\
CurrentAge:-1,OverallCond:1,LotArea:1"


# functions ####


def read_data(file_path: str) -> pd.DataFrame:
    """
    Read the housing dataset from the given file path.

    Parameters
    ----------
    file_path : str
        Path to the dataset file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)


def split_data(df: pd.DataFrame, seed=42) -> (pd.DataFrame, pd.Series):
    """
    Split the housing dataset into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    seed: int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the features.
    pd.Series
        Series containing the target.
    """
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(['SalePrice'], axis=1), df['SalePrice'],
        test_size=0.1, random_state=seed
    )

    return X_train, X_test, y_train, y_test


def train_model(X: pd.DataFrame, y: pd.Series, cat_features=None,
                monotonic_constraints=None, random_seed=42,
                loss_function='RMSE', iterations=1000, learning_rate=0.01,
                verbose=True) -> cb.CatBoostRegressor:
    """
    Train a CatBoost Regressor model on the given features and target.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the features.
    y : pd.Series
        Series containing the target.
    cat_features: list, default=None
        List of categorical features.
    monotonic_constraints: str, default=None
        Monotonic constraints for the features.
    random_seed: int, default=42
        Random seed for reproducibility.
    loss_function: str, default='RMSE'
        Loss function to use.
    iterations: int, default=1000
        Number of iterations.
    learning_rate: float, default=0.01
        Learning rate.
    verbose: bool, default=True
        Whether to print training progress.

    Returns
    -------
    cb.CatBoostRegressor
        Trained CatBoost model.
    """
    model = cb.CatBoostRegressor(
        loss_function=loss_function,
        monotone_constraints=monotonic_constraints,
        iterations=iterations,
        learning_rate=learning_rate,
        random_seed=random_seed,
        verbose=verbose
    )
    model.fit(X, y, cat_features=cat_features)
    return model


# main ####
if __name__ == '__main__':
    # start training
    logging.info(f'{"="*10}TRAINING{"="*10}')

    # Load the data
    logging.info('Loading data...')
    df_houses = read_data('../data/clean/train.csv')

    # split data
    logging.info('Splitting data...')
    df_houses_subset = df_houses[COLS_TO_STAY].copy()
    X_train, X_test, y_train, y_test = split_data(df_houses_subset)

    # Train the model
    logging.info('Training model...')
    model = train_model(
        X_train, y_train,
        cat_features=COLS_CATEGORICAL,
        monotonic_constraints=MONOTONE_CONSTRAINS,
        verbose=False
        )

    # Evaluate the model
    logging.info('Evaluating model...')
    metrics_train = evaluate_model(
        model, X_train, y_train, do_plot=True, dir_model=DIR_MODEL
        )
    metrics_test = evaluate_model(
        model, X_test, y_test, do_plot=True, dir_model=DIR_MODEL
        )
    df_metrics = pd.concat([metrics_train, metrics_test], axis=1)
    df_metrics.columns = ['train', 'test']

    # Save the model and metrics
    logging.info('Saving model...')
    model.save_model(DIR_MODEL + '/model.cbm')
    df_metrics.to_csv(DIR_MODEL + '/metrics.csv')
    logging.info(f'Model and metrics saved to {DIR_MODEL}')
    logging.info('\nDone training!')
