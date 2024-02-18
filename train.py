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
import yaml
import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
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


def split_data(
    df: pd.DataFrame, yobj='y', test_size=0.2, seed=42, stratify=None
     ) -> (pd.DataFrame, pd.Series):
    """
    Split the housing dataset into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    yobj: str
        Target variable name.
    test_size: float
        Proportion of the dataset to include in the test split.
    seed: int
        Random seed for reproducibility.
    stratify: array-like, default=None
        If not None, data is split in a stratified fashion.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the features.
    pd.Series
        Series containing the target.
    """
    if test_size < 0 or test_size > 1:
        logging.error('test_size must be between 0 and 1')
        logging.debug('Using default test_size=0.2')
        test_size = 0.2
    
    # split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop([yobj], axis=1), df[yobj],
            test_size=test_size, random_state=seed,
            stratify=stratify
        )

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f'Error splitting the data: {e}')
        return None, None, None, None


def train_model(
     X: pd.DataFrame, y: pd.Series,
     algorithm='CatBoost',
     cat_features=None,
     hyperparms={}
     ) -> cb.CatBoostRegressor:
    """
    Train a CatBoost Regressor model on the given features and target.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the features.
    y : pd.Series
    model_name: str, default='CatBoost'
        Name of the model to train. Only CatBoost is supported.
    cat_features: list, default=None
        List of categorical features.
    hyperparms: dict, default={}
        Hyperparameters for the model.

    Returns
    -------
    cb.CatBoostRegressor
        Trained CatBoost model.
    """
    # validation of algorithm
    if algorithm.lower() != 'catboost':
        logging.error('Only CatBoost is supported for now, using CatBoost.')

    # validation of hyperparameters
    if hyperparms is not None:
        if 'monotone_constraints' in hyperparms:
            # all hyperparms['monotone_constraints'] must be in X.columns
            if not all(
                [col in X.columns
                 for col in hyperparms['monotone_constraints']]
                 ):
                logging.error(
                    'The monotonic constraints must be in the features'
                    )
                logging.debug('Ignoring monotonic constraints')
                hyperparms.pop('monotone_constraints')

    # train model
    model = cb.CatBoostRegressor(**hyperparms)
    model.fit(X, y, cat_features=cat_features)
    return model


# main ####
if __name__ == '__main__':
    # start training
    logging.info(f'{"="*10}TRAINING{"="*10}')

    # Read config
    config = utils.get_config()

    # Generate parser
    parser = utils.generate_parser(
        config['etl']['train']['arguments'], name='Train arguments'
        )
    args = parser.parse_args()

    # Load the data
    logging.info('Loading data...')
    df_houses = utils.read_data(
        file_path=args.csv_file_path
        )

    # split data
    logging.info('Splitting data...')
    # getting y & X
    target_var = config['etl']['train']['target_variable']
    numeric_cols = config['model']['variables'].get('numerical', [])
    categorical_cols = config['model']['variables'].get('categorical', [])
    model_cols = numeric_cols + categorical_cols + [target_var]
    stratify_columns = config['etl']['train']['stratify']

    logging.debug(f"Model columns: {numeric_cols + categorical_cols}")
    # subset data
    df_houses_subset = df_houses[model_cols].copy()
    X_train, X_test, y_train, y_test = split_data(
        df_houses_subset,
        yobj=target_var,
        test_size=float(args.test_size),
        seed=config['etl']['train']['seed'],
        stratify=df_houses_subset[stratify_columns]
        )
    logging.debug(f"Train size: {X_train.shape[0]},\
                 Test size: {X_test.shape[0]}")

    # Train the model
    logging.info('Training model...')
    model = train_model(
        X_train, y_train,
        algorithm=config['model']['algorithm'],
        cat_features=categorical_cols,
        hyperparms=config['model']['hyperparams']
        )

    # Evaluate the model
    dir_model = f"models/{args.name_model}"
    if args.evaluate:
        logging.info('Evaluating model...')
        os.makedirs(dir_model, exist_ok=True)
        metrics_train = utils.evaluate_model(
            model, X_train, y_train, do_plot=True, dir_model=dir_model,
            suffix='_train'
            )
        metrics_test = utils.evaluate_model(
            model, X_test, y_test, do_plot=True, dir_model=dir_model,
            suffix='_test'
            )
        df_metrics = pd.concat([metrics_train, metrics_test], axis=1)
        # TODO: with the suffix as key, dont need to rename the columns
        df_metrics.columns = ['train', 'test']

        # Save the model and metrics
        logging.info('Saving model...')
        model.save_model(dir_model + '/model.cbm')
        df_metrics.to_csv(dir_model + '/metrics.csv')
        logging.info(f'Model and metrics saved to {dir_model}')

    # save hyperparameters
    logging.info('Saving hyperparameters...')
    with open(dir_model + '/hyperparams.yaml', 'w') as file:
        yaml.dump(config['model']['hyperparams'], file)

    logging.info('\nDone training!')
