import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from tqdm import tqdm

from constants import OUTPUT_DATA_COLUMNS
from utils import process_data, load_data

models_dict = {
    "bs_hb": GradientBoostingRegressor(
        n_estimators=276,
        max_depth=20,
        min_samples_split=6,
        min_samples_leaf=14,
        max_features='log2',
        learning_rate=0.2523920613427682,
    ),
    "bs_rm": GradientBoostingRegressor(
        n_estimators=263,
        max_depth=2,
        min_samples_split=6,
        min_samples_leaf=1,
        max_features='sqrt',
        learning_rate=0.19746561089365053,
    ),
    "bs_rp02": GradientBoostingRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=15,
        min_samples_leaf=16,
        max_features=5,
        learning_rate=0.41487003552155716,
    ),
    "bs_a5": GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=3,
        max_features=10,
        learning_rate=0.0643048881887624,
    ),
    "bs_k": GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        min_samples_split=16,
        min_samples_leaf=17,
        max_features=10,
        learning_rate=0.06280486937006903,
    ),
    "gs_a5": GradientBoostingRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=15,
        min_samples_leaf=2,
        max_features='sqrt',
        learning_rate=0.01,
    ),
    "gs_k": ExtraTreesRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
    ),
}


# meta_model = LinearRegression()
# stacking_dict = {
#     'stack': StackingRegressor(estimators= [
#         ('hb', models_dict['Best_HB']),
#         ('a5', models_dict['Best_A5']),
#         ('rm', models_dict['Best_Rm']),
#     ], final_estimator=meta_model)
# }

def run_final_test(models_dict, file_name):
    data = load_data()

    results = []
    for target in OUTPUT_DATA_COLUMNS:
        (X_train, X_test, y_train, y_test) = process_data(
            data=data,
            target_column=target,
            augmentation=False
        )
        pbar = tqdm(models_dict.items())
        for model_name, model in pbar:
            rmse_cv_train = -np.mean(
                cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error'))
            y_cv_pred = cross_val_predict(model, X_train, y_train, cv=5)
            r2_cv_train = r2_score(y_train, y_cv_pred)

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            rmse_test = mean_squared_error(y_true=y_test, y_pred=y_test_pred, squared=False)
            rmse_train = mean_squared_error(y_true=y_train, y_pred=y_train_pred, squared=False)

            r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)
            r2_train = r2_score(y_true=y_train, y_pred=y_train_pred)

            results.append({
                "model_name": model_name,
                "target": target,
                "rmse_test": rmse_test,
                "rmse_train": rmse_train,
                "rmse_cv_train": rmse_cv_train,
                "r2_test": r2_test,
                "r2_train": r2_train,
                "r2_cv_train": r2_cv_train,
            })

    pd.DataFrame(results).to_csv("results/" + file_name + ".csv")


if __name__ == "__main__":
    # run_final_test(MODELS_DICT, 'basic_results_RMSE')
    run_final_test(models_dict, 'evaluation_results')
    # run_final_test(stacking_dict, 'final_results_Stack')
