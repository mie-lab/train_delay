import os
import pickle
import numpy as np
from ngboost import NGBRegressor
from ngboost import learners  # default_tree_learner
from ngboost.distns import Normal, LogNormal
from ngboost import scores  # MLE
from scipy.stats import lognorm
from config import OUTLIER_CUTOFF


def train_ngboost(train_set_rf_x, train_set_rf_y, val_set_rf_x, val_set_rf_y, save_path="test", **kwargs):
    ngb = NGBRegressor(
        Base=learners.default_tree_learner,
        Dist=Normal,
        Score=scores.LogScore,
        natural_gradient=True,
        verbose=True,
    )
    ngb.fit(train_set_rf_x, train_set_rf_y)

    # compute mse
    ngb_mean_pred = ngb.predict(val_set_rf_x)
    print("MSE ngboost:", np.mean((ngb_mean_pred - val_set_rf_y) ** 2))

    with open(os.path.join(save_path, "ngb.p"), "wb") as outfile:
        pickle.dump(ngb, outfile)


def train_ngboost_lognormal(
    train_set_rf_x, train_set_rf_y, val_set_rf_x, val_set_rf_y, save_path="test", dist=Normal, **kwargs
):
    ngb = NGBRegressor(Dist=LogNormal, verbose=True)
    ngb.fit(train_set_rf_x, train_set_rf_y + OUTLIER_CUTOFF)
    # compute mse
    ngb_mean_pred = ngb.predict(val_set_rf_x) - OUTLIER_CUTOFF
    print("MSE ngboost:", np.mean((ngb_mean_pred - val_set_rf_y) ** 2))
    with open(os.path.join(save_path, "ngb_lognormal.p"), "wb") as outfile:
        pickle.dump(ngb, outfile)


def test_ngb_lognormal(load_model, val_set_rf_x, return_params=False, **kwargs):
    # load trained model
    with open(os.path.join(load_model), "rb") as infile:
        ngb = pickle.load(infile)
    # predicted distribution parameters
    ngb_dist_pred = ngb.pred_dist(val_set_rf_x)
    if return_params:
        return ngb_dist_pred.params["s"], ngb_dist_pred.params["scale"]
    else:
        ngb_mean_pred = ngb.predict(val_set_rf_x) - OUTLIER_CUTOFF
        ngb_unc = get_unc_lognormal(ngb_dist_pred.params["s"], ngb_dist_pred.params["scale"])
        return ngb_mean_pred, ngb_unc


def test_ngboost(load_model, val_set_rf_x, **kwargs):
    # load trained model
    with open(os.path.join(load_model), "rb") as infile:
        ngb = pickle.load(infile)
    ngb_mean_pred = ngb.predict(val_set_rf_x)
    # predicted distribution parameters
    ngb_dist_pred = np.array(ngb.pred_dist(val_set_rf_x).params["scale"])
    return ngb_mean_pred, ngb_dist_pred


def get_unc_lognormal(s, scale):
    return lognorm.ppf(0.7, s, scale=scale) - lognorm.ppf(0.3, s, scale=scale)
