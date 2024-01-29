# house-prices-mcd :house_with_garden:
Homework for product architecture. The repo creates an interface where the user can value a house from the Ames Housing dataset. This data is only for simulation purposes. The idea is to generate a ML product that can predict the price of a house based on the following features:

`Variables` 
1. `Lot Area` in square feet
2. `Built Area` in square feet
3. `Age of the house` in years (the last time it was built or renovated)
4. `Overall Condition`: rates the overall material and finish of the housefrom from 1 to 10
5. `Overall Quality`: rates the overall condition of the house from 1 to 10
6. `MSZoning`: The general zoning classification (only residential houses):
    - RL: Residential Low Density
    - RM: Residential Medium Density
    - RH: Residential High Density


# How to use it :question: :computer:
## Notebooks :notebook:

The notebooks are in the `notebooks` folder. The notebooks should be run in the following order:
1. `eda.ipynb`: This notebook contains the exploratory data analysis of the dataset. It also contains the feature engineering and the feature selection.
2. `model.ipynb`: This notebook contains the model training and the model evaluation.
3. `model_no_log.ipynb`: This notebook contains the model training and the model evaluation without the log transformation of the target variable.

## Demo (source) :rocket:

The demo is in the `source` folder. The demo is a notebook called `demo.ipynb`. The demo does the following:
1. Predict the price of a house based on the input of the user.
2. Explain the prediction of the model using SHAP values in a waterfall plot.

The payload of the model is the following:
```python
data = {
    'CurrentAge': 20, # Current age of the house
    'GrLivArea': 1000, # Above grade (ground) living area square feet
    'LotArea': 1000, # Lot size in square feet
    'OverallCond': 5, # Overall condition rating, from 1 to 10
    'OverallQual': 5, # Overall material and finish quality, from 1 to 10
    'MSZoning': 'RL' # Zoning classification
}
```

# Caveats :warning:
The MVP only works for:

- Houses
- Normal Sale Condition or Partial Sale Condition which means it was a presale. (not foreclosure, not short sale, etc.)
- Residential houses (no commercial, no industrial, etc.)
- Warranty Convencional or New Home (no VA, no FHA, etc.)

# Requirements :clipboard:
The requirements are in the `requirements.txt` file. To install them, run the following command in your terminal:
```bash
pip install -r requirements.txt
```

The python version used is `3.9`.