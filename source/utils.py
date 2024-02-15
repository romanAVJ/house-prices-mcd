"""
Utils module for the project.

Author: ravj
Date: Mon Feb 12 19:49:11 2023
"""
# imports ####
import os
import pandas as pd
import numpy as np
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model: cb.CatBoostRegressor, X: pd.DataFrame,
                   y: pd.Series, do_plot=False, dir_model='figures'
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

    Returns
    -------
    pd.Series
    """
    # predict
    y_pred = model.predict(X)

    # metrics
    dict_metrics = {}
    dict_metrics['rmse'] = np.sqrt(np.mean((y - y_pred)**2))
    dict_metrics['mape'] = np.mean(np.abs((y - y_pred) / y)) * 100
    dict_metrics['r2'] = model.score(X, y)

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
        plt.savefig(dir_model + '/figures/observed_vs_predicted.jpg')
        # close plot
        plt.close()

    return pd.Series(dict_metrics)
