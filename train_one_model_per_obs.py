import os
import pandas as pd
import argparse
import warnings
import numpy as np

from run import get_features, split_train_test, get_train_val_test
from train_delay.mlp_model import train_test_time_dropout, train_aleatoric, train_unc_nn
from train_delay.rf_model import train_random_forest, rf_overfit
from train_delay.gaussian_process import train_gaussian_process
from train_delay.ngboost_model import train_ngboost, train_ngboost_lognormal

MODEL_FUNC_TRAIN = {
    "nn_dropout": train_test_time_dropout,
    "nn_aleatoric": train_aleatoric,
    "random_forest": train_random_forest,
    "gaussian_process": train_gaussian_process,
    "nn": train_unc_nn,
    "rf_overfit": rf_overfit,
    "ngb": train_ngboost,
    "ngb_lognormal": train_ngboost_lognormal,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_path", type=str, default=os.path.join("data", "data_enriched.csv"))
    parser.add_argument(
        "-m",
        "--method",
        default="random_forest",
        type=str,
        help="one of random_forest, gaussian_process, ngb, nn, nn_aleatoric or nn_dropout",
    )
    parser.add_argument("-e", "--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("-o", "--out_dir", default="test", type=str, help="Where to save model")
    parser.add_argument("--model_dir", default="trained_models", type=str, help="Path to save models")
    parser.add_argument("-v", "--version", default=2, type=int, help="version of feature set")
    args = parser.parse_args()

    epochs = args.epochs
    model_func = MODEL_FUNC_TRAIN[args.method]
    if os.path.exists(os.path.join(args.model_dir, args.out_dir)):
        warnings.warn("Careful: model dir already exists")
    os.makedirs(os.path.join(args.model_dir, args.out_dir), exist_ok=True)

    data = pd.read_csv(args.inp_path)
    data.index.name = "id"

    for obs in np.arange(1, 130, 10):
        print("---------- OBS", obs, "---------------------")
        # print("changed part")
        # print(len(data))
        data_obs = data[data["obs_count"] == obs]
        # print(len(data))

        # split into train, val and test
        train_set, val_set, test_set = split_train_test(data_obs)

        use_features = get_features(data.columns, version=args.version)

        # remove features that are not relevant when only training on one observation
        use_features = [f for f in use_features if f not in ["feat_obs_count", "feat_time_since_stop"]]

        (train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y) = get_train_val_test(
            train_set, val_set, test_set, use_features, training=True
        )
        print("DATA SHAPES", train_set_nn_x.shape, train_set_nn_y.shape, val_set_nn_x.shape, val_set_nn_y.shape)

        model_func(
            train_set_nn_x,
            train_set_nn_y,
            val_set_nn_x,
            val_set_nn_y,
            epochs=epochs,
            dropout_rate=0.5,
            save_path=os.path.join(args.out_dir, f"nn_{obs}"),
            use_features=use_features,
            learning_rate=0.0005,
        )
