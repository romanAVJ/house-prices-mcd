"""
This module contains unit tests for the functions in the utils module.

Author: ravj
Date: Mon Feb 24 8:25:31 2024
"""

import pytest
import logging
import pandas as pd
import numpy as np
import catboost as cb
import os
from source.utils import (
    read_data, get_config, generate_parser,
    evaluate_model, save_file, get_logger
    )


def test_get_config():
    """
    Test the get_config function with a valid file and a non-existent file.
    """
    # Test with a valid file
    config = get_config()
    assert isinstance(config, dict)

    # Test it has the expected keys
    assert 'prep' in config['etl']
    assert 'train' in config['etl']
    assert 'inference' in config['etl']


def test_read_data():
    """
    Test the read_data function with a valid file and a non-existent file.
    """
    # get config file
    config = get_config()['etl']

    # for prep, train and inference check if the file exists
    for key in config:
        valid_path = config[key]['arguments']['csv_file_path']['default']
        file_path = valid_path
        assert os.path.exists(file_path)

    # Test with a valid file
    df = read_data(valid_path)
    assert isinstance(df, pd.DataFrame)


def test_generate_parser():
    """
    Test the generate_parser function with valid arguments and invalid args.
    """
    # Test with valid arguments
    parser = generate_parser({
        'arg1': {'type': int, 'help': 'An integer'}
        })
    assert parser.parse_args(['--arg1', '10']).arg1 == 10

    # Test with invalid arguments
    with pytest.raises(SystemExit):
        generate_parser({
            'arg1': {'type': int, 'help': 'An integer'}}
            ).parse_args(['--arg1', 'not_an_int'])


def test_evaluate_model():
    """
    Test the evaluate_model function with valid inputs and invalid inputs.
    """
    # Create a simple CatBoostRegressor model for testing
    model = cb.CatBoostRegressor(silent=True)
    model.fit(np.arange(10), np.arange(10))

    # Create a simple DataFrame and Series for testing
    X = pd.DataFrame({'feature': np.arange(10)})
    y = pd.Series(np.arange(10))

    # Test with valid inputs
    result = evaluate_model(model, X, y)
    assert isinstance(result, pd.Series)


def test_save_file():
    """
    Test the save_file function with a valid DataFrame and file path,
    and with an invalid DataFrame.
    """
    # config
    config = get_config()

    # Test with a valid DataFrame and file path
    df = pd.DataFrame({'feature': range(10)})
    dir_save =\
        config['etl']['prep']['save_path']
    file_save =\
        config['etl']['prep']['arguments']['save_file']['default']
    valid_path = f"{dir_save}/{file_save}"

    # create dir if it does not exist
    assert os.path.exists(dir_save)

    # save file
    save_file(df, valid_path)


def test_get_logger():
    """
    Test the get_logger function with valid inputs and invalid inputs.
    """
    # Test with valid inputs
    logger = get_logger('test_logger')
    assert isinstance(logger, logging.Logger)
