"""
ETL Pipeline for Iowa Housing Dataset.

This script reads the raw housing dataset, transforms it, subsets the
relevant columns, and saves the cleaned data.

Author: ravj
Date: Sun Feb 11 18:09:01 2023
"""
# imports ####
import logging
import os
import pandas as pd
import numpy as np
from source import utils


def augment_data(df: pd.DataFrame, current_year=2010) -> pd.DataFrame:
    """
    Augment the housing dataset with additional features.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame to augment.
    current_year : int, default=2010
        Current year for age calculation.

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame.
    """
    # Calculate current age
    df['CurrentAge'] = current_year - np.maximum(df['YearBuilt'],
                                                 df['YearRemodAdd'])

    # Calculate porch area
    df['PorchArea'] = df['OpenPorchSF'] + df['EnclosedPorch'] +\
        df['3SsnPorch'] + df['ScreenPorch']

    # Calculate total baths
    df['TotalBaths'] = df['FullBath'] + df['HalfBath']

    # Classify house stories
    df['Story'] = np.select(
        [
            df['HouseStyle'].str.contains('1Story'),
            df['HouseStyle'].str.contains('1.5Fin'),
            df['HouseStyle'].str.contains('1.5Unf'),
            df['HouseStyle'].str.contains('2Story'),
            df['HouseStyle'].str.contains('2.5Fin'),
            df['HouseStyle'].str.contains('2.5Unf'),
            df['HouseStyle'].str.contains('SFoyer'),
            df['HouseStyle'].str.contains('SLvl')
        ],
        [
            'one',
            'one_and_half',
            'one_and_half',
            'two',
            'two_and_half',
            'two_and_half',
            'split',
            'split'
        ]
    )

    # Classify story finish
    df['StoryFinish'] = np.where(
        df['HouseStyle'].str.contains('Unf'), 'unfinished', 'finished'
        )

    return df


def subset_data(df: pd.DataFrame, subsets: dict, cols: str) -> pd.DataFrame:
    """
    Subset the DataFrame based on specific conditions and columns.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame to subset.
    subsets : dict
        Conditions to subset the DataFrame. The dictionary is of the form:
        {column_name: {
            operator: type_of_operator_as_str, value: value_to_compare},
        ...
        }
    cols : str
        Columns to keep after subsetting.

    Returns
    -------
    pd.DataFrame
        Subsetted DataFrame.
    """
    # Subset the DataFrame
    for col, subset in subsets.items():
        df = df.query(f"{col} {subset['operator']} {subset['value']}")

    # Keep only the specified columns
    df = df[cols].reset_index(drop=True)
    return df


# main ####
if __name__ == '__main__':
    # get logger
    logger = utils.get_logger('prep', level=logging.DEBUG)

    # Start ETL process
    logger.info(f'{"="*10}ETL{"="*10}')

    # Read config
    config = utils.get_config()

    # Generate parser
    parser = utils.generate_parser(
        config['etl']['prep']['arguments'], name='ETL arguments'
        )
    args = parser.parse_args()

    # Read raw data
    logger.info('Reading data...')
    df_houses_raw = utils.read_data(
        file_path=args.csv_file_path
        )
    logger.debug(f"Data read with {df_houses_raw.shape[0]} rows and\
    {df_houses_raw.shape[1]} columns")

    # Augment data
    logger.info('Transforming data...')
    df_houses = augment_data(
        df_houses_raw,
        current_year=config['etl']['prep']['current_year']
        )

    # Subset data
    if args.subset:
        logger.info('Subsetting data...')
        df_houses = subset_data(
            df_houses,
            subsets=config['etl']['prep']['filters'],
            cols=config['etl']['prep']['variables']
            )
        percent_less = 1 - df_houses.shape[0] / df_houses_raw.shape[0]
        percent_less_rounded = np.round(percent_less * 100, 2)
        logger.debug(f"Data trimmed to {percent_less_rounded}% less rows")

    # Save cleaned data
    logger.info('Saving data...')
    dir_save = config['etl']['prep']['save_path']
    os.makedirs(dir_save, exist_ok=True)
    file_name = f"{dir_save}{args.save_file}.csv"
    utils.save_file(df_houses, file_name)  # save the file
    logger.info(f"Data saved to {file_name}")
    logger.info('\nDone ETL!')
