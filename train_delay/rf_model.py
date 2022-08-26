from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import json
import pickle


def train_random_forest(
    train_set_rf_x, train_set_rf_y, val_set_rf_x, val_set_rf_y, use_features=None, save_path="test", **kwargs
):
    # TUNING
    best_regr, best_performance = None, np.inf
    for max_depth in [2, 5, 10, 20, 30]:
        regr = RandomForestRegressor(max_depth=max_depth, random_state=0)
        # fit
        regr.fit(train_set_rf_x, train_set_rf_y)
        pred = regr.predict(val_set_rf_x)
        mse = mean_squared_error(pred, val_set_rf_y)
        print(f"Performance on val set with max depth {max_depth}: {mse}")
        if mse < best_performance:
            best_performance = mse
            best_regr = regr
    regr = best_regr

    if use_features is not None:
        print("Most important features:")
        print(np.array(use_features)[np.argsort(regr.feature_importances_)][::-1])
        with open(os.path.join("trained_models", save_path, "rf_feature_importances.json"), "w") as outfile:
            json.dump(np.array(use_features)[np.argsort(regr.feature_importances_)][::-1].tolist(), outfile)

    with open(os.path.join("trained_models", save_path, "random_forest.p"), "wb") as outfile:
        pickle.dump(regr, outfile)


def test_random_forest(load_model, val_set_rf_x, **kwargs):
    # load trained model
    with open(os.path.join("trained_models", load_model, "random_forest.p"), "rb") as infile:
        regr = pickle.load(infile)

    # get prediction per tree
    per_tree_pred = [tree.predict(val_set_rf_x) for tree in regr.estimators_]
    rf_unc = np.std(per_tree_pred, axis=0)
    rf_pred = np.mean(per_tree_pred, axis=0)

    return rf_pred, rf_unc
