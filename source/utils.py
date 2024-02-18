"""
Utils module for the project.

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
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


# parameters ####
# Logging configuration
logging.basicConfig(level=logging.DEBUG)

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
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        logging.error("Error: File not found.")
    except Exception as e:
        logging.error("An error occurred:", e)


def get_config() -> dict:
    """
    Get the configuration for the project.

    Returns
    -------
    dict
    """
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error("Error: File not found.")
    except Exception as e:
        logging.error("An error occurred:", e)


def generate_parser(
        dict_args: dict, name='Script arguments'
        ) -> argparse.ArgumentParser:
    """
    Generate an argument parser based on the given dictionary.

    Parameters
    ----------
    dict_args : dict
        Dictionary containing the arguments.
    name : str, default='Script arguments'
        Name of the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    # Parse command-line arguments
    try:
        parser = argparse.ArgumentParser(description=name)
        for key, value in dict_args.items():
            parser.add_argument(f"--{key}", **value)
        return parser
    except Exception as e:
        logging.error("An error occurred:", e)


def evaluate_model(model: cb.CatBoostRegressor, X: pd.DataFrame,
                   y: pd.Series, do_plot=False, dir_model='figures',
                   suffix=''
                   ) -> pd.Series:
    """
    Evaluate the given model on the given features and target.

    Parameters
    ----------
    model : cb.CatBoostRegressor
        Trained CatBoost model.
    X : pd.DataFrame
        DataFrame containing the features.
    y : pd.Series
        Series containing the target.
    do_plot: bool, default=False
        Whether to plot the predictions.
    dir_model: str, default='figures'
        Directory to save the plots.
    suffix: str, default=''
        Suffix to add to the plot file name.

    Returns
    -------
    pd.Series
    """
    try:
        # predict
        y_pred = model.predict(X)

        # metrics
        dict_metrics = {}
        dict_metrics['rmse'] = np.sqrt(np.mean((y - y_pred)**2))
        dict_metrics['mape'] = np.mean(np.abs((y - y_pred) / y)) * 100
        dict_metrics['r2'] = model.score(X, y)
        # TODO: add the suffix to the metrics keys

        # plot scatter
        if do_plot:
            # create directory
            os.makedirs(dir_model + '/figures', exist_ok=True)

            # scatter plot
            plt.figure(figsize=(8, 8))
            # add identity line
            data_to_plot = [[(y_pred).min(), (y_pred).max()],
                            [(y_pred).min(), (y_pred).max()]]
            plt.plot(data_to_plot, c='gray', linestyle='--')
            # get residuals
            sns.scatterplot(x=y, y=y_pred)
            plt.xlabel('observed')
            plt.ylabel('predicted')
            plt.title('observed vs predicted')
            # save plot
            os.makedirs(f'{dir_model}/figures', exist_ok=True)
            plt.savefig(
                f"{dir_model}/figures/observed_vs_predicted_{suffix}.png"
                )
            # close plot
            plt.close()

        return pd.Series(dict_metrics)
    except Exception as e:
        logging.error("An error occurred:", e)


def save_file(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the given DataFrame to the given file path.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    file_path : str
        Path to save the DataFrame.

    Returns
    -------
    None
    """
    try:
        # Save the DataFrame to the given file path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
    except Exception as e:
        logging.error("An error occurred:", e)
