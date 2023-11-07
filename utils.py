import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from constants import INPUT_DATA_COLUMNS, OUTPUT_DATA_COLUMNS, DATA_PATH, MODELS_DICT, PARAMS_GRID_SEARCH, \
    PARAMS_BAYESIAN_SEARCH, DATA_COLUMNS_TO_DELETE, CONSTITUENT_CHEMICAL_ELEMENTS


def load_data():
    df = pd.read_excel(
        io=DATA_PATH,
        sheet_name='Dane',
        skiprows=1,
        nrows=520,
        usecols=INPUT_DATA_COLUMNS + OUTPUT_DATA_COLUMNS,
    )
    return df


def calculate_log_seconds(minutes):
    seconds = minutes * 60
    return np.log10(seconds)


def calculate_Fe_content(X: DataFrame):
    return 100 - X[CONSTITUENT_CHEMICAL_ELEMENTS].sum(axis=1)


def process_data(data, target_column, augmentation):
    data = data.dropna(subset=[target_column])
    X = data[INPUT_DATA_COLUMNS]
    y = data[target_column]
    if augmentation:
        X['Fe'] = calculate_Fe_content(X)
        X = X.drop(DATA_COLUMNS_TO_DELETE, axis=1)
        X['logAustTime'] = X['aust_czas'].apply(lambda x: calculate_log_seconds(x))
        X['logAusfTime'] = X['ausf_czas'].apply(lambda x: calculate_log_seconds(x))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    return X_train, X_test, y_train, y_test


def get_models_dict(model_names, hyperparameters_tuning_method):
    if hyperparameters_tuning_method == 'gs':
        search_space = PARAMS_GRID_SEARCH
    elif hyperparameters_tuning_method == 'bs':
        search_space = PARAMS_BAYESIAN_SEARCH
    else:
        search_space = {}
    models_dict = {
        model_name: (model, search_space.get(model_name))
        for model_name, model in MODELS_DICT.items()
        if model_name in model_names
    }
    return models_dict
