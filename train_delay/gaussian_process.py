from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import json
import pickle


def train_gaussian_process(train_set_rf_x, train_set_rf_y, val_set_rf_x, val_set_rf_y, save_path="test", **kwargs):
    # TUNING
    best_regr, best_performance = None, np.inf
    for kernel in [None, DotProduct() + WhiteKernel()]:
        regr = GaussianProcessRegressor(kernel=kernel)
        print("GP is initialized")
        # fit
        regr.fit(train_set_rf_x, train_set_rf_y)
        print("GP is fitted")
        pred = regr.predict(val_set_rf_x)
        mse = mean_squared_error(pred, val_set_rf_y)
        print(f"Performance on val set with kernel: {mse}")
        if mse < best_performance:
            best_performance = mse
            best_regr = regr
    with open(os.path.join("trained_models", save_path, "gaussian_process.p"), "wb") as outfile:
        pickle.dump(best_regr, outfile)


def test_gaussian_process(load_model, val_set_rf_x, **kwargs):
    # load trained model
    with open(os.path.join("trained_models", load_model, "gaussian_process.p"), "rb") as infile:
        regr = pickle.load(infile)

    # get prediction per tree
    pred, unc = regr.predict(val_set_rf_x, return_std=True)

    return pred, unc
