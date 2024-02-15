"""
Training Pipeline for Iowa Housing Dataset.

This script reads the clean housing dataset to train a machine learning model
to predict house prices using the CatBoost Regressor.

Author: ravj
Date: Mon Feb 12 19:49:11 2023
"""
# imports ####
import os
import logging
import argparse
import pandas as pd
import numpy as np
import catboost as cb
from utils import evaluate_model


# parameters ####
# Logging configuration
logging.basicConfig(level=logging.INFO)

# model directory
DIR_MODEL = '../models/vanilla_catboost'
DIR_PREDICTIONS = '../data/predictions'

# cols for training
COLS_NUMERIC = [
    'CurrentAge', 'GrLivArea', 'LotArea', 'OverallCond', 'OverallQual'
]
COLS_CATEGORICAL = ['MSZoning']
COLS_TO_STAY = COLS_NUMERIC + COLS_CATEGORICAL

# functions ####


def read_csv_file(csv_file_path: str) -> None:
    """
    Read a CSV file and print its contents.

    Parameters:
        csv_file_path (str): The path to the CSV file.

    Returns:
        None
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
        logging.info(f"Successfully loaded. File shape: {df.shape}")
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print("An error occurred:", e)
    return df


def validate_cols(df: pd.DataFrame, cols: list) -> None:
    """
    Validate if the given columns are present in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to validate.
        cols (list): The list of columns to validate.

    Returns:
        None
    """
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")


# main ####
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Read a CSV file.")
    parser.add_argument("--csv_file", help="Path to the CSV file.")

    # parser for evaluate_model
    parser.add_argument("--evaluate", default=False, type=bool,
                        help="Whether to plot the predictions.")
    args = parser.parse_args()

    # read data
    csv_file_path = args.csv_file
    logging.info("Reading CSV file... ")
    df_batch = read_csv_file(csv_file_path)

    # validate columns
    validate_cols(df_batch, COLS_TO_STAY)

    # predict
    model = cb.CatBoostRegressor()
    model.load_model(f'{DIR_MODEL}/model.cbm')
    df_batch['SalePrice_pred'] = np.round(model.predict(
        df_batch[COLS_TO_STAY]
        ))

    # evaluate
    if args.evaluate:
        metrics = evaluate_model(
            model, df_batch[COLS_TO_STAY], df_batch['SalePrice'], do_plot=True
            )
        logging.info(f"Metrics: {metrics}")

    # save predictions
    logging.info("Saving predictions...")
    os.makedirs(DIR_PREDICTIONS, exist_ok=True)
    date_time = pd.to_datetime('now').strftime('%Y%m%d_%H%M%S')
    df_batch.to_csv(f'{DIR_PREDICTIONS}/predictions_{date_time}.csv')
    logging.info(f"Predictions saved to {DIR_PREDICTIONS}")
    logging.info("Done.")
