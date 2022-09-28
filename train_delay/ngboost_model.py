import os
import pickle
import numpy as np
import ngboost
from ngboost import learners  # default_tree_learner
from ngboost import distns  # Normal
from ngboost import scores  # MLE


def train_ngboost(train_set_rf_x, train_set_rf_y, val_set_rf_x, val_set_rf_y, save_path="test", **kwargs):
    ngb = ngboost.NGBoost(
        Base=learners.default_tree_learner,
        Dist=distns.Normal,
        Score=scores.LogScore,
        natural_gradient=True,
        verbose=True,
    )
    ngb.fit(train_set_rf_x, train_set_rf_y)

    # compute mse
    ngb_mean_pred = ngb.predict(val_set_rf_x)
    print("MSE ngboost:", np.mean((ngb_mean_pred - val_set_rf_y) ** 2))

    with open(os.path.join("trained_models", save_path, "ngb.p"), "wb") as outfile:
        pickle.dump(ngb, outfile)


def test_ngboost(load_model, val_set_rf_x, **kwargs):
    # load trained model
    with open(os.path.join("trained_models", load_model, "ngb.p"), "rb") as infile:
        ngb = pickle.load(infile)
    ngb_mean_pred = ngb.predict(val_set_rf_x)
    # predicted distribution parameters
    ngb_dist_pred = np.array(ngb.pred_dist(val_set_rf_x).params["scale"])
    return ngb_mean_pred, ngb_dist_pred
