experiment_configs = [
    # {
    #     'experiment_id': '1_0',
    #     'experiment_name': 'Test Models Grid Search',
    #     'experiment_description': 'Basic Test with Grid Search hyperparameters tuning without data augmentation.',
    #     'augmentation': False,
    #     'model_names': ['lin', 'dt', 'etr', 'rf','efr', 'svr', 'knn', 'ab', 'gb', 'xgb', 'xgbf'],
    #     'hyperparameters_tuning_method': 'gs',
    # },
    {
        'experiment_id': '1_1_0_x',
        'experiment_name': 'Test Models Bayessian Search',
        'experiment_description': 'Bayessian Search test',
        'augmentation': False,
        'model_names': ['gb'],
        'hyperparameters_tuning_method': 'bs',
    },

    # {
    #     'experiment_id': '2_0_final',
    #     'experiment_name': 'Test Models Grid Search + Augmentation (parameters reduction)',
    #     'experiment_description': 'Test with Grid Search hyperparameters tuning with parameters reduction',
    #     'augmentation': True,
    #     'model_names': ['lin', 'dt', 'etr', 'rf', 'efr', 'svr', 'knn', 'ab', 'gb', 'xgb', 'xgbf'],
    #     'hyperparameters_tuning_method': 'gs',
    # },
    # {
    #     'experiment_id': '3_0_final',
    #     'experiment_name': 'Test Models Grid Search + Augmentation(parameters reduction, added Fe parameter)',
    #     'experiment_description': 'Test with Grid Search hyperparameters tuning with parameters reduction and addition of Fe parameter',
    #     'augmentation': True,
    #     'model_names': ['lin', 'dt', 'etr', 'rf', 'efr', 'svr', 'knn', 'ab', 'gb', 'xgb', 'xgbf'],
    #     'hyperparameters_tuning_method': 'gs',
    # },
    # {
    #     'experiment_id': '4_0_final',
    #     'experiment_name': 'Test Models Grid Search + Augmentation(parameters reduction, added Fe parameter, log10 time)',
    #     'experiment_description': 'Test with Grid Search hyperparameters tuning with parameters reduction and log10 time',
    #     'augmentation': True,
    #     'model_names': ['lin', 'dt', 'etr', 'rf', 'efr', 'svr', 'knn', 'ab', 'gb', 'xgb', 'xgbf'],
    #     'hyperparameters_tuning_method': 'gs',
    # },
]
