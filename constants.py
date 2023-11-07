from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor, XGBRFRegressor

CONSTITUENT_CHEMICAL_ELEMENTS = [
    'C', 'Si', 'Mn', 'Mg', 'Cu', 'Ni', 'Mo',
    'S', 'P', 'V', 'Cr', 'Ti', 'Sn', 'Al'
]
INPUT_DATA_COLUMNS = [
    'C', 'Si', 'Mn', 'Mg', 'Cu', 'Ni', 'Mo',
    'S', 'P', 'V', 'Cr', 'Ti', 'Sn', 'Al',
    'aust_temp', 'aust_czas',
    'ausf_temp', 'ausf_czas',
    'thickness'
]
DATA_COLUMNS_TO_DELETE = [
    'S', 'P', 'V', 'Cr', 'Ti', 'Sn', 'Al'
]
OUTPUT_DATA_COLUMNS = ['Rm', 'Rp02', 'A5', 'HB', 'K']
DATA_PATH = 'data/Dane_zeliwa.xlsx'
RESULTS_PATH = Path('results')
RESULTS_PATH.mkdir(exist_ok=True)

MODELS_DICT = {
    'lin': LinearRegression(),
    'dt': DecisionTreeRegressor(random_state=7),
    'etr': ExtraTreeRegressor(random_state=7),
    'rf': RandomForestRegressor(random_state=7, verbose=0),
    'efr': ExtraTreesRegressor(random_state=7, verbose=0),
    'svr': SVR(verbose=0),
    'knn': KNeighborsRegressor(),
    'ab': AdaBoostRegressor(random_state=7),
    'gb': GradientBoostingRegressor(random_state=7),
    'xgb': XGBRegressor(random_state=7),
    'xgbf': XGBRFRegressor(random_state=7)
}

PARAMS_BAYESIAN_SEARCH = {
    'gb': [
        Integer(10, 500, name='n_estimators'),
        Categorical([None, 2, 3, 4, 5, 10, 15, 20], name='max_depth'),
        Integer(2, 20, name='min_samples_split'),
        Integer(1, 20, name='min_samples_leaf'),
        Categorical(['sqrt', 'log2', None, 1, 3, 5, 10], name='max_features'),
        Real(0.01, 0.7, name='learning_rate')
    ]
}

PARAMS_GRID_SEARCH = {
    'lin': {
        'model__fit_intercept': [True, False],
    },
    'dt': {
        'model__max_depth': [None, 2, 3, 5, 10, 15],
        'model__min_samples_split': [2, 5, 15],
        'model__min_samples_leaf': [1, 2, 4, 8],
        'model__max_features': ['sqrt', 'log2', None, 1]
    },
    'etr': {
        'model__max_depth': [None, 2, 3, 5, 10, 15],
        'model__min_samples_split': [2, 5, 15],
        'model__min_samples_leaf': [1, 2, 4, 8],
        'model__max_features': ['sqrt', 'log2', None, 1]
    },
    'rf': {
        'model__n_estimators': [10, 100, 200, 500],
        'model__max_depth': [None, 2, 3, 4, 5, 10, 15],
        'model__min_samples_split': [2, 5, 15],
        'model__min_samples_leaf': [1, 2, 4, 8],
        'model__max_features': ['sqrt', 'log2', None, 1]
    },
    'efr': {
        'model__n_estimators': [10, 100, 200, 500],
        'model__max_depth': [None, 2, 3, 4, 5, 10, 15],
        'model__min_samples_split': [2, 5, 15],
        'model__min_samples_leaf': [1, 2, 4, 8],
        'model__max_features': ['sqrt', 'log2', None, 1]
    },
    'svr': {
        'model__C': [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80],
        'model__kernel': ['linear', 'rbf', 'poly'],
        'model__degree': [1, 2, 3, 4],
    },
    'knn': {
        'model__n_neighbors': [3, 5, 7, 10],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2],
    },
    'ab': {
        'model__n_estimators': [50, 100, 200, 500],
        'model__learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4]

    },
    'gb': {
        'model__n_estimators': [50, 100, 200, 500],
        'model__max_depth': [None, 2, 5, 15],
        'model__min_samples_split': [2, 5, 15],
        'model__min_samples_leaf': [1, 2, 4, 8],
        'model__max_features': ['sqrt', 'log2', None, 1],
        'model__learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4]
    },
    'xgb': {
        'model__n_estimators': [50, 100, 200, 500],
        'model__max_depth': [None, 2, 5, 15],
        'model__learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4]
    },
    'xgbf': {
        'model__n_estimators': [50, 100, 200, 500],
        'model__max_depth': [None, 2, 5, 15],
        'model__colsample_bynode': [0.0, 0.25, 0.5, 0.75, 1.0],
        'model__colsample_bytree': [0.0, 0.25, 0.5, 0.75, 1.0],

    }
}
