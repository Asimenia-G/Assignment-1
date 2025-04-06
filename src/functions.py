import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from copy import deepcopy

def load_data_from_csv(filename, select_best_columns=False):
    df = pd.read_csv(filename)

    if select_best_columns:
        selected_columns = ['Host age', 'Eubacterium sulci', 'Desulfonispora thiosulfatigenes',
                            'Sex', 'Sporobacter termitidis', 'Clostridium clariflavum',
                            'Alistipes putredinis', 'Ruminococcus champanellensis',
                            'Clostridium symbiosum', 'Ruminiclostridium thermocellum']
        X = df[selected_columns].values
    else:
        X = df.drop(columns=['BMI']).values

    y = df['BMI'].values

    return X, y


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def negative_rmse_scorer(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))


def perform_parameter_tuning(estimator, param_grid, X, y, cv_splits, seed=42, verbose=4):
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid,
                               cv=KFold(n_splits=cv_splits, shuffle=True, random_state=seed),
                               scoring=make_scorer(negative_rmse_scorer),
                               verbose=verbose)
    grid_search.fit(X, y)

    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    return best_estimator, best_score, best_params


def train_baseline(estimator, X, y, cv=10):
    scores    = cross_val_score(estimator, X, y, cv=cv, scoring=make_scorer(negative_rmse_scorer))
    mean_rmse = -np.mean(scores)
    std_rmse  = np.std(scores)

    estimator.fit(X, y)
    return estimator, mean_rmse, std_rmse


def train_with_feature_selection(estimator, X, y, k, cv=10):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)

    scores = cross_val_score(estimator, X_selected, y, cv=cv, scoring=make_scorer(negative_rmse_scorer))
    mean_rmse = -np.mean(scores)
    std_rmse = np.std(scores)

    estimator.fit(X_selected, y)

    return estimator, selector, mean_rmse, std_rmse



def perform_feature_selection(X, y, model, model_name, random_state=42, cv=10):

    print(f"Performing feature selection using F-test based on baseline.")
    feature_dimensions = [2, 3, 5, 8, 10, 15, 20, 50]
    print(f"Testing {feature_dimensions} as possible dimensions")

    min_RMSE      = np.infty
    best_model    = None
    best_selector = None

    for k in feature_dimensions:
        model, selector, mean_rmse, std_rmse = train_with_feature_selection(model, X, y, k=k)
        if mean_rmse < min_RMSE:
            min_RMSE      = mean_rmse
            best_model    = deepcopy(model)
            best_selector = selector
            s = '(updating best model)'
        else:
            s = ''
        print(f"k={k:02d} | ElasticNet RMSE: {mean_rmse:.3f}±{std_rmse:.3f} " + s)

    return best_model, best_selector, min_RMSE




def get_bootstrap_splits(X_val, y_val, n_bootstrap=100, random_state=42):
    np.random.seed(random_state)
    n_samples = len(y_val)

    X_bootstrap = []
    y_bootstrap = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap.append(X_val[indices])
        y_bootstrap.append(y_val[indices])

    return X_bootstrap, y_bootstrap


def evaluate_with_bootstrap(X_val, y_val, estimator, n_bootstrap=100, random_state=42):
    X_bootstrap, y_bootstrap = get_bootstrap_splits(X_val, y_val, n_bootstrap, random_state)

    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for X_boot, y_boot in zip(X_bootstrap, y_bootstrap):
        y_pred = estimator.predict(X_boot)

        rmse_scores.append(np.sqrt(mean_squared_error(y_boot, y_pred)))
        mae_scores.append(mean_absolute_error(y_boot, y_pred))
        r2_scores.append(r2_score(y_boot, y_pred))

    return rmse_scores, mae_scores, r2_scores


