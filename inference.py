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
import pandas as pd
import numpy as np
import catboost as cb
import source.utils as utils


# parameters ####
datetime = pd.to_datetime('today').strftime('%Y%m%d-%H')

# Logging configuration
os.makedirs(f'logs/{datetime}', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    filename=f'logs/{datetime}/prep.log',
    filemode='a'
    )


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
            logging.error(f"Column '{col}' not found in DataFrame.\
                           Cannot proceed.")
            raise ValueError(f"Column '{col}' not found in DataFrame.")


def predict_model(dir_model: str, X: pd.DataFrame) -> pd.Series:
    """
    Predict using a CatBoost model.
    ---
    dir_model: str
        Directory where the model is saved.
    X: pd.DataFrame
        DataFrame containing the features.
    """
    # read model
    model = cb.CatBoostRegressor()

    try:
        model.load_model(f'{dir_model}model.cbm')
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e

    # predict
    return np.round(model.predict(X), 0)


# main ####
if __name__ == "__main__":
    # main ####
    # Start training
    logging.info(f'{"="*10}INFERENCE{"="*10}')

    # Read config
    config = utils.get_config()

    # Generate parser
    parser = utils.generate_parser(
        config['etl']['inference']['arguments'], name='Inference Arguments'
        )
    args = parser.parse_args()

    # read data
    logging.info("Reading CSV file... ")
    df_batch = utils.read_data(file_path=args.csv_file_path)

    # validate columns
    target_var = config['etl']['train']['target_variable']
    numeric_cols = config['model']['variables'].get('numerical', [])
    categorical_cols = config['model']['variables'].get('categorical', [])
    model_cols = numeric_cols + categorical_cols
    validate_cols(df_batch, model_cols)  # validation

    # predict
    logging.info("Predicting...")
    model = cb.CatBoostRegressor()
    dir_model = args.model_path
    model_name = dir_model.split('/')[-2]
    y_name = config['etl']['train']['target_variable']
    df_batch[f'{y_name}_pred'] = predict_model(dir_model, df_batch[model_cols])

    # evaluate
    if args.evaluate:
        metrics = utils.evaluate_model(
            model, df_batch[model_cols], df_batch[y_name], do_plot=True
            )
        logging.debug(f"Metrics: {metrics}")

    # save predictions
    logging.info("Saving predictions...")
    dir_predictions = "data/predictions"
    os.makedirs(dir_predictions, exist_ok=True)
    date_time = pd.to_datetime('now').strftime('%Y%m%d')
    file_name =\
        f"{dir_predictions}/{args.save_file}_{model_name}_{date_time}.csv"
    utils.save_file(df_batch, file_name)  # save predictions
    logging.info("Predictions saved")
    logging.info("Done.")
