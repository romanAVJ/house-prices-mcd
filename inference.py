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
import source.utils as utils


# parameters ####
# Logging configuration
logging.basicConfig(level=logging.INFO)

# functions ####


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
    # parser ####
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Read a CSV file.")
    parser.add_argument("--csv_file",
                        help="Path to the CSV file.",
                        default='data/clean/house_data.csv')

    # parser for evaluate_model
    parser.add_argument("--evaluate", default=False, type=bool,
                        help="Whether to plot the predictions.")
    args = parser.parse_args()

    # main ####
    # Start training
    logging.info(f'{"="*10}INFERENCE{"="*10}')

    # Read config
    config = utils.get_config()

    # read data
    csv_file_path = args.csv_file
    logging.info("Reading CSV file... ")
    df_batch = utils.read_data(file_path=csv_file_path)

    # validate columns
    target_var = config['etl']['train']['target_variable']
    numeric_cols = config['model']['variables'].get('numerical', [])
    categorical_cols = config['model']['variables'].get('categorical', [])
    model_cols = numeric_cols + categorical_cols
    validate_cols(df_batch, model_cols)  # validation

    # predict
    model = cb.CatBoostRegressor()
    model_name = config['model']['name']
    dir_model = f"models/{model_name}"
    model.load_model(f'{dir_model}/model.cbm')
    y_name = config['etl']['train']['target_variable']

    df_batch[f'{y_name}_pred'] = np.round(model.predict(
        df_batch[model_cols]
        ))

    # evaluate
    if args.evaluate:
        metrics = utils.evaluate_model(
            model, df_batch[model_cols], df_batch[y_name], do_plot=True
            )
        logging.info(f"Metrics: {metrics}")

    # save predictions
    logging.info("Saving predictions...")
    dir_predictions = config['etl']['inference']['save_path']
    os.makedirs(dir_predictions, exist_ok=True)
    date_time = pd.to_datetime('now').strftime('%Y%m%d')
    df_batch.to_csv(f'{dir_predictions}/pred_{model_name}_{date_time}.csv')
    logging.info(f"Predictions saved to {dir_predictions}")
    logging.info("Done.")
