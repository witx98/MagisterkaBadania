import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from tqdm import tqdm

from configs import experiment_configs
from constants import OUTPUT_DATA_COLUMNS, RESULTS_PATH
from utils import load_data, process_data, get_models_dict


class Experiments:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models_dict = None
        self.search_results_df = None
        self.target_column = None
        self.search_results_dfs = []

    def run_experiment(self):
        for self.target_column in OUTPUT_DATA_COLUMNS:
            logging.info(f"Running experiment {self.config['experiment_id']} for target: {self.target_column}")

            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test
            ) = process_data(
                data=data,
                target_column=self.target_column,
                augmentation=config['augmentation']
            )
            self.models_dict = get_models_dict(
                model_names=config['model_names'],
                hyperparameters_tuning_method=config['hyperparameters_tuning_method']
            )
            self._train_models()
        self._save_results()
        logging.info(f"Saved results")

    def testing_final_model(self):
        # TODO
        pass

    def _save_results(self):
        results_file_name = f"{config['experiment_id']}.csv"
        results_df = pd.concat(self.search_results_dfs).reset_index(drop=True)
        results_df.to_csv(RESULTS_PATH / results_file_name)

    def _train_models(self):
        if self.config['hyperparameters_tuning_method'] == 'bs':
            results = []
            assert len(config['model_names']) == 1, "Podaj tylko  jeden model na raz"
            model_name = config['model_names'][0]

            model_name, (model, param_space) = list(self.models_dict.items())[0]
            # Define the objective function for optimization (RMSE and R2)
            def objective(params):
                if model_name == 'gb':
                    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, learning_rate = params

                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        learning_rate=learning_rate
                    )

                rmse = -np.mean(
                    cross_val_score(model,
                                    self.X_train,
                                    self.y_train,
                                    cv=5,
                                    scoring='neg_root_mean_squared_error'))

                y_pred = cross_val_predict(model, self.X_train, self.y_train, cv=5)
                r2 = r2_score(self.y_train, y_pred)
                if model_name == 'gb':
                    results.append(
                        [n_estimators, max_depth, min_samples_split,
                         min_samples_leaf, max_features, learning_rate,
                         rmse, r2])

                return rmse

            _ = gp_minimize(objective, param_space,
                            n_calls=50, random_state=0, n_jobs=-1)
            if model_name == 'gb':
                search_results_df = pd.DataFrame(
                    results,
                    columns=['n_estimators', 'max_depth', 'min_samples_split',
                             'min_samples_leaf', 'max_features', 'learning_rate',
                             'RMSE', 'R2']
                )
                search_results_df.insert(0, 'target', self.target_column)
                self.search_results_dfs.append(search_results_df)

        if self.config['hyperparameters_tuning_method'] == 'gs':
            grid_search_results = []
            pbar_inner = tqdm(self.models_dict.items())
            for model_name, (model, param_grid) in pbar_inner:
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                grid_search = GridSearchCV(
                    pipe,
                    param_grid,
                    cv=3,
                    scoring=['neg_root_mean_squared_error', 'r2'],
                    refit='neg_root_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(self.X_train, self.y_train)
                cv_results_df = pd.DataFrame(grid_search.cv_results_)[[
                    'params',
                    'mean_test_neg_root_mean_squared_error',
                    'std_test_neg_root_mean_squared_error',
                    'mean_test_r2',
                    'std_test_r2',
                    'mean_fit_time'
                ]]
                cv_results_df['model_name'] = model_name
                grid_search_results.append(
                    cv_results_df
                )
            search_results_df = (
                pd.concat(grid_search_results)
                .sort_values(
                    'mean_test_neg_root_mean_squared_error',
                    ascending=False
                ).reset_index(drop=True)
            )
            search_results_df.insert(0, 'target', self.target_column)
            self.search_results_dfs.append(search_results_df)


if __name__ == "__main__":
    data = load_data()
    pbar = tqdm(experiment_configs)
    for config in pbar:
        experiments = Experiments(config=config, data=data)
        experiments.run_experiment()
