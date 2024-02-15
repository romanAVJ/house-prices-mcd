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

# parameters ####
# Current year for age calculation
CURR_YEAR = 2010

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Columns to keep in the final dataset
COLS_TO_STAY = [
    'Id', 'CurrentAge', 'LotArea', 'PorchArea',
    'TotalBaths', 'OverallQual', 'OverallCond', 'GrLivArea',
    'SaleType', 'SaleCondition', 'MSZoning', 'Story',
    'StoryFinish', 'SalePrice', 'TotRmsAbvGrd'
]

# Directory to save cleaned data
DIR_SAVE = '../data/clean/'

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


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment the housing dataset with additional features.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame to augment.

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame.
    """
    # Calculate current age
    df['CurrentAge'] = CURR_YEAR - np.maximum(df['YearBuilt'],
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


def subset_data(df: pd.DataFrame, cols: str) -> pd.DataFrame:
    """
    Subset the DataFrame based on specific conditions and columns.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame to subset.
    cols : str
        Columns to keep after subsetting.

    Returns
    -------
    pd.DataFrame
        Subsetted DataFrame.
    """
    df = (
        df
        # Filter out properties with 0 bedrooms or 0 bathrooms
        .query('TotRmsAbvGrd > 0 and TotalBaths > 0')
        # Filter out properties with lot area greater than 200,000 sqft
        .query('LotArea < 200000')
        # Filter out properties with porch area greater than 1,000 sqft
        .query('PorchArea < 1000')
        # Subset by sale type
        .query('SaleType in ["WD", "New"]')
        # Subset by sale condition
        .query('SaleCondition in ["Normal", "Partial"]')
        # Subset by MSZoning
        .query('MSZoning in ["RL", "RM", "RP", "RH"]')
    )
    # Keep only the specified columns
    df = df[cols].reset_index(drop=True)
    return df


# main ####
if __name__ == '__main__':
    # Start ETL process
    logging.info(f'{"="*10}ETL{"="*10}')

    # Read raw data
    logging.info('Reading data...')
    df_houses_raw = read_data('../data/raw/train.csv')
    logging.info(f"Data read with {df_houses_raw.shape[0]} rows and\
    {df_houses_raw.shape[1]} columns")

    # Augment data
    logging.info('Transforming data...')
    df_houses = augment_data(df_houses_raw)

    # Subset data
    df_houses = subset_data(df_houses, cols=COLS_TO_STAY)
    percent_less = 1 - df_houses.shape[0] / df_houses_raw.shape[0]
    percent_less_rounded = np.round(percent_less * 100, 2)
    logging.info(f"Data trimmed to {percent_less_rounded}% less rows")

    # Save cleaned data
    logging.info('Saving data...')
    os.makedirs(DIR_SAVE, exist_ok=True)
    df_houses.to_csv(DIR_SAVE + 'train.csv', index=False)
    logging.info(f"Data saved to {DIR_SAVE} train.csv")
    logging.info('\nDone ETL!')
