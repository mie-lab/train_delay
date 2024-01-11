# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from GPy.models import SparseGPRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import json
import pickle
import time


def train_gaussian_process(
    train_set_rf_x, train_set_rf_y, val_set_rf_x, val_set_rf_y, save_path="test", nr_induced=500, **kwargs
):
    # TUNING
    # best_regr, best_performance = None, np.inf
    # for kernel in [DotProduct() + WhiteKernel()]:
    # sklearn version
    # regr = GaussianProcessRegressor(kernel=kernel)
    # regr.fit(train_set_rf_x, train_set_rf_y)
    # print("GP is fitted")
    # pred = regr.predict(val_set_rf_x)
    # print(f"Performance on val set with kernel: {mse}")
    # if mse < best_performance:
    #     best_performance = mse
    #     best_regr = regr

    # Sparse GPy
    train_set_rf_y = train_set_rf_y.reshape((-1, 1))
    rand_inds = np.random.permutation(len(train_set_rf_x))[:nr_induced]
    Z = train_set_rf_x[rand_inds]

    tic = time.time()
    m = SparseGPRegression(train_set_rf_x, train_set_rf_y, Z=Z)
    m.likelihood.variance = 0.05
    print("Time fit", time.time() - tic)
    print("Likelihood", m.log_likelihood())

    # optimize parameters
    # tic = time.time()
    # m.randomize()
    # m.Z.unconstrain()
    # m.optimize("bfgs")
    # print("Time for optimizing parameters")

    # predict
    pred, std = m.predict(val_set_rf_x)
    print("mean and std of pred and std", np.mean(pred), np.std(pred), np.mean(std), np.std(std))

    # error
    mse = mean_squared_error(pred, val_set_rf_y)
    print("MSE val", mse)

    with open(os.path.join(save_path, "gaussian_process.p"), "wb") as outfile:
        pickle.dump(m, outfile)


def test_gaussian_process(load_model, val_set_rf_x, **kwargs):
    # load trained model
    with open(os.path.join(load_model, "gaussian_process.p"), "rb") as infile:
        regr = pickle.load(infile)

    # get prediction per tree
    pred, unc = regr.predict(val_set_rf_x)

    return np.squeeze(pred), np.squeeze(unc)
