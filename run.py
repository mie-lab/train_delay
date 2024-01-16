from pyexpat import model
import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

from train_delay.baselines import simple_current_delay_bl, simple_median_bl, simple_mean_bl, overall_avg
from train_delay.metrics import (
    get_metrics,
    calibrate_pi,
    get_intervals,
    calibrate_likely,
    add_likely,
    likelihood_lognormal,
)
from train_delay.mlp_model import test_test_time_dropout, test_aleatoric, test_unc_nn
from train_delay.ngboost_model import test_ngboost, test_ngb_lognormal
from train_delay.rf_model import test_random_forest
from feature_selection import select_features

# from train_delay.gaussian_process import test_gaussian_process

MODEL_FUNC_TEST = {
    "nn_dropout": test_test_time_dropout,
    "nn_aleatoric": test_aleatoric,
    "random_forest": test_random_forest,
    # "gaussian_process": test_gaussian_process,
    "nn": test_unc_nn,
    "ngb": test_ngboost,
    "ngb_lognormal": test_ngb_lognormal,
    "simple_median": simple_median_bl,
    "simple_mean": simple_mean_bl,
    "simple_avg": overall_avg,
    "simple_current_delay": simple_current_delay_bl,
}

SAVE_MODELS = ["nn", "ngb", "simple_median", "random_forest", "ngb_lognormal", "simple_current_delay"]


def split_train_test(data, ratio=0.8, save_path=None):
    if "use" in data.columns:  # case 1: train test split already exists
        print("Split already exists in dataset")
        out = []
        for split_name in ["train", "val", "test"]:
            out.append(data[data["use"] == split_name])
    else:
        # split into train and test by day
        all_days = data["day"].unique()
        cutoff = round(len(all_days) * ratio)
        test_val_cutoff = (len(all_days) - cutoff) // 2 + cutoff
        train_days, val_days, test_days = (
            all_days[:cutoff],
            all_days[cutoff:test_val_cutoff],
            all_days[test_val_cutoff:],
        )
        print("cuttoff after train:", cutoff, "cutoff after val", test_val_cutoff, "total nr days", len(all_days))
        out = []
        for days, split_name in zip([train_days, val_days, test_days], ["train", "val", "test"]):
            out.append(data[data["day"].isin(days)])
            if save_path is not None:
                data.loc[data["day"].isin(days), "split"] = split_name
        if save_path is not None:
            data.to_csv(save_path, index=False)
    return tuple(out)


def plot_by_obs_count(df_unc, col="obs_count", save_path="outputs/by_obs_count"):
    plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)
    ax.plot(df_unc.groupby(col).agg({"MSE": "mean"}), label="error", c="blue")
    ax.set_ylabel("MSE", c="blue")
    ax2 = ax.twinx()
    ax2.plot(df_unc.groupby(col).agg({"unc": "mean"}), label="uncertainty", c="orange")
    ax2.set_ylabel("Uncertainty", c="orange")
    ax2.set_xlabel(col + " (normalized)")
    plt.title(f"Error and uncertainty by {col}")
    plt.savefig(save_path + f"{col}.png")


def get_train_val_test(train_set, val_set, test_set, use_features, training=False):
    # retrict to these features plus the predicted variable
    prev_len = len(train_set)
    train_set = train_set[use_features + ["final_delay"]].dropna()
    assert prev_len - len(train_set) == 0, "there are still NaNs"
    print("Dropping nans:", prev_len - len(train_set))
    val_set = val_set[use_features + ["final_delay"]].dropna()
    test_set = test_set[use_features + ["final_delay"]].dropna()
    # divide into x and y
    train_set_x, train_set_y = train_set[use_features], train_set["final_delay"]
    val_set_x, val_set_y = val_set[use_features], val_set["final_delay"]
    test_set_x, test_set_y = test_set[use_features], test_set["final_delay"]

    # normalize train set
    train_set_nn_x = np.array(train_set_x).astype(np.float64)
    train_mean, train_std = np.mean(train_set_nn_x, axis=0), np.std(train_set_nn_x, axis=0)
    train_set_nn_x = (train_set_nn_x - train_mean) / train_std
    train_set_nn_y = np.array(train_set_y.values).astype(float)
    # normalize val set
    val_set_nn_x = np.array(val_set_x).astype(np.float64)
    val_set_nn_x = (val_set_nn_x - train_mean) / train_std
    val_set_nn_y = np.array(val_set_y.values).astype(float)
    # normalize test set
    test_set_nn_x = np.array(test_set_x).astype(np.float64)
    test_set_nn_x = (test_set_nn_x - train_mean) / train_std
    test_set_nn_y = np.array(test_set_y.values).astype(float)
    print(
        "train x and y",
        train_set_nn_x.shape,
        train_set_nn_y.shape,
        "val x and y",
        val_set_nn_x.shape,
        val_set_nn_y.shape,
        "test x and y",
        test_set_nn_x.shape,
        test_set_nn_y.shape,
    )
    if training:
        return train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y
    else:
        # returning also train and val index again because restricted to not nans
        return (
            train_set.index,
            val_set.index,
            test_set.index,
            train_set_nn_x,
            train_set_nn_y,
            val_set_nn_x,
            val_set_nn_y,
            test_set_nn_x,
            test_set_nn_y,
        )


def extract_param_kwargs_from_name(model_to_test):
    if "-" not in model_to_test:
        return {"dropout_rate": 0.5}
    # replace minuses
    model_to_test = model_to_test.replace("1e-05", "0.00001").replace("1e-06", "0.000001").split("-")
    params_kwargs = {
        "dropout_rate": float(model_to_test[-1]),
        "learning_rate": float(model_to_test[-2]),
        "nr_layers": int(model_to_test[-3]),
        "first_layer_size": int(model_to_test[-5]),
        "second_layer_size": int(model_to_test[-4]),
    }
    print("PARAMS", params_kwargs)
    return params_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_path", type=str, default=os.path.join("data", "data_enriched.csv"))
    parser.add_argument("-m", "--model_dir", default="test_allfeatures", type=str, help="name of model directory")
    parser.add_argument("--model_path", default="trained_models", type=str)
    parser.add_argument("-o", "--out_path", default="outputs", type=str)
    parser.add_argument("-v", "--version", default="all_features", type=str, help="version of feature set")
    parser.add_argument("-a", "--pi_alpha", default=0.1, type=float, help="alpha for PI width calibration")
    parser.add_argument("-p", "--plot", action="store_true", help="plot?")
    args = parser.parse_args()

    data = pd.read_csv(args.inp_path)
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, args.model_dir), exist_ok=True)

    assert args.version in args.model_dir
    # # to remove outliers:
    # data = data[(data["final_delay"] > -5) & (data["final_delay"] < 15)]

    if "up_submission" in args.model_dir:
        print("ATTENTION: reducing to up")
        data = data[data["DIR"] == "up"]
    elif "down_submission" in args.model_dir:
        print("ATTENTION: reducing to down")
        data = data[data["DIR"] == "down"]

    # split into train, val and test
    train_set, val_set, test_set = split_train_test(data)  # , save_path="data/data_enriched.csv")

    # select suitable features for ML models
    use_features = select_features(data.columns, version=args.version)

    # preprocess and dropn the ones with NaN features
    (
        train_set_index,
        val_set_index,
        test_set_index,
        train_set_nn_x,
        train_set_nn_y,
        val_set_nn_x,
        val_set_nn_y,
        test_set_nn_x,
        test_set_nn_y,
    ) = get_train_val_test(train_set, val_set, test_set, use_features)

    # For simple baselines, use the raw train and test data
    train_set = train_set.loc[train_set_index]  # restrict to the ones that are also used for the other models
    val_set = val_set.loc[val_set_index]
    test_set = test_set.loc[test_set_index]
    # print(len(train_set), len(test_set))
    # res_dict = run_simple_baselines(train_set, test_set)
    # print("Intermediate results of baselines:")
    # print(pd.DataFrame(res_dict).swapaxes(1, 0).sort_values("MSE"))

    basic_df = test_set[
        [
            "train_id_daily",
            "train_id",
            "delay_dep",
            "DIR",
            "day",
            "obs_count",
            "stops",
            "cat",
            "remaining_runtime",
            "distanceKM_to_final",
        ]
    ].copy()

    # get baseline uncertainties (std of final delay per train ID)
    _, unc_bl = simple_mean_bl(train_set, test_set)
    _, unc_bl_val = simple_mean_bl(train_set, val_set)

    print("DATA SHAPES", train_set_nn_x.shape, train_set_nn_y.shape, test_set_nn_x.shape, test_set_nn_y.shape)
    res_dict = {}

    # Test all models in the folder
    model_weights = os.path.join(args.model_path, args.model_dir)
    models_to_test_list = [
        m_file for m_file in os.listdir(model_weights) if m_file[0] != "." and m_file[-3:] != "png"
    ] + ["simple_median", "simple_mean", "simple_avg"]
    for model_to_test in models_to_test_list:
        # if os.path.exists(os.path.join(args.out_path, args.model_dir, model_to_test.split("p")[0])):
        #     print("already there", model_to_test)
        #     continue
        # ngp.p -> ngp;  nn-128-64 -> nn
        model_type = model_to_test.split("-")[0].split(".")[0]
        # check whether pretrained model exists
        trained_model_exists = os.path.exists(os.path.join(model_weights, model_to_test)) or os.path.exists(
            os.path.join(model_weights, model_to_test + ".p")
        )
        if not trained_model_exists and "simple" not in model_to_test:
            print(f"Skipping {model_to_test} because no pretrained model available.")
            continue

        print("-------------- ", model_to_test, "--------------")
        # get the correct test function
        model_func = MODEL_FUNC_TEST[model_type]

        param_kwargs = extract_param_kwargs_from_name(model_to_test)

        if "simple" in model_type:
            pred, unc = model_func(train_set, test_set)
        else:
            pred, unc = model_func(
                os.path.join(model_weights, model_to_test), test_set_nn_x, return_params=False, **param_kwargs
            )
            # # for aleatoric vs epistmic:
            # pred, unc, _ = model_func(
            #     os.path.join(model_weights, model_to_test), test_set_nn_x, return_params=False, **param_kwargs
            # )

        # add to the other metrics
        for model_type_name, unc_est in zip([model_type, model_type + "_unc_bl"], [unc, unc_bl]):
            if "simple" in model_type and "bl" in model_type_name:
                # only do the bl evaluation for the non-simple models (otherwise same result)
                continue
            # if "bl" in model_type_name:
            #     continue

            # fill the table with our basic information
            temp_df = basic_df.copy()
            temp_df["pred"] = pred
            temp_df["unc"] = unc_est
            temp_df["final_delay"] = test_set_nn_y

            # Calibration --> run on val set
            if "simple" in model_type:
                pred_val, unc_val = model_func(train_set, val_set)
            else:
                pred_val, unc_val = model_func(os.path.join(model_weights, model_to_test), val_set_nn_x, **param_kwargs)
                # # for aleatoric vs epistmic:
                # pred_val, unc_val, _ = model_func(
                #     os.path.join(model_weights, model_to_test), val_set_nn_x, **param_kwargs
                # )

            # use bl uncertainty if required
            if "unc_bl" in model_type_name:
                unc_val = unc_bl_val
            # get interval bounds
            quantiles = calibrate_pi(val_set_nn_y, pred_val, unc_val, alpha=args.pi_alpha)
            print("Quantiles", quantiles)
            intervals = get_intervals(temp_df["pred"].values, temp_df["unc"].values, quantiles)
            temp_df["interval_low_bound"] = intervals[:, 0] * 60
            temp_df["interval_high_bound"] = intervals[:, 1] * 60

            # Likelihood:
            if "lognormal" in model_type_name:
                s, scale = test_ngb_lognormal(
                    os.path.join(model_weights, model_to_test), test_set_nn_x, return_params=True
                )
                temp_df = likelihood_lognormal(temp_df, s, scale)
            else:
                best_factor = calibrate_likely(val_set_nn_y, pred_val, unc_val)
                print("Best factor for likelihood", best_factor)
                temp_df = add_likely(temp_df, factor=best_factor)

            # get metrics and save in final dictionary
            cleaned_name = model_to_test.replace(".p", "")  # remove .p  # for aleatoric vs epistmic + "_epistemic"
            save_csv_path = (
                os.path.join(args.out_path, args.model_dir, cleaned_name) if model_type_name in SAVE_MODELS else None
            )
            if args.pi_alpha != 0.1:
                converted_to_coverage = int((1 - args.pi_alpha) * 100)
                save_csv_path += f"_coverage{converted_to_coverage}"
            res_dict[cleaned_name] = get_metrics(temp_df, save_path=save_csv_path)
            print("metrics", res_dict[cleaned_name])

    # result_table = pd.DataFrame(res_dict).swapaxes(1, 0).sort_values(["mean_pi_width"]).round(3)
    # print(result_table)
    # result_table.to_csv(os.path.join(args.out_path, args.model_dir, "results_summary.csv"))
