import os
import pandas as pd
import argparse
import warnings
import time

from feature_selection import select_features
from run import split_train_test, get_train_val_test
from train_delay.mlp_model import train_test_time_dropout, train_aleatoric, train_unc_nn
from train_delay.rf_model import train_random_forest, rf_overfit

# from train_delay.gaussian_process import train_gaussian_process
from train_delay.ngboost_model import train_ngboost, train_ngboost_lognormal

MODEL_FUNC_TRAIN = {
    "nn_dropout": train_test_time_dropout,
    "nn_aleatoric": train_aleatoric,
    "random_forest": train_random_forest,
    # "gaussian_process": train_gaussian_process,
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
    parser.add_argument("-o", "--out_dir", default="test", type=str, help="Name of folder to save models")
    parser.add_argument("--model_dir", default="trained_models", type=str, help="Path to save model")
    parser.add_argument("-v", "--version", default="allfeatures", type=str, help="version of feature set")
    parser.add_argument("--first_layer_size", default=128, type=int)
    parser.add_argument("--second_layer_size", default=128, type=int)
    parser.add_argument("--nr_layers", default=2, type=int)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    args = parser.parse_args()

    epochs = args.epochs
    model_func = MODEL_FUNC_TRAIN[args.method]
    os.makedirs(args.model_dir, exist_ok=True)

    out_folder = os.path.join(args.model_dir, f"{args.out_dir}_{args.version}")
    if os.path.exists(out_folder):
        warnings.warn("Careful: model dir already exists")
    os.makedirs(out_folder, exist_ok=True)

    data = pd.read_csv(args.inp_path)
    data.index.name = "id"
    # print("changed part")
    # print(len(data))
    # data = data[(data["obs_count"] >= 100) & (data["obs_count"] < 120)]
    # print(len(data))

    # split into train, val and test
    train_set, val_set, test_set = split_train_test(data)

    use_features = select_features(data.columns, version=args.version)

    (train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y) = get_train_val_test(
        train_set, val_set, test_set, use_features, training=True
    )
    print("DATA SHAPES", train_set_nn_x.shape, train_set_nn_y.shape, val_set_nn_x.shape, val_set_nn_y.shape)
    tic = time.time()
    model_func(
        train_set_nn_x,
        train_set_nn_y,
        val_set_nn_x,
        val_set_nn_y,
        save_path=out_folder,
        use_features=use_features,
        **vars(args),
    )
    print("Runtime", time.time() - tic)
