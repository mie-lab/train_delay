from pyexpat import model
import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

from train_delay.baselines import simple_current_delay_bl, simple_median_bl, simple_mean_bl, overall_avg
from train_delay.metrics import (
    calibrate_pi,
    get_intervals,
    calibrate_likely,
    add_likely,
    likelihood_lognormal,
)
from train_delay.mlp_model import test_test_time_dropout, test_aleatoric, test_unc_nn
from train_delay.ngboost_model import test_ngboost, test_ngb_lognormal
from train_delay.rf_model import test_random_forest
from train_delay.gaussian_process import test_gaussian_process
from run import split_train_test, get_train_val_test, get_features

MODEL_FUNC_TEST = {
    "nn_dropout": test_test_time_dropout,
    "nn_aleatoric": test_aleatoric,
    "random_forest": test_random_forest,
    "gaussian_process": test_gaussian_process,
    "nn": test_unc_nn,
    "ngb": test_ngboost,
    "ngb_lognormal": test_ngb_lognormal,
    "simple_median": simple_median_bl,
    "simple_mean": simple_mean_bl,
    "simple_avg": overall_avg,
    # "simple_current_delay": simple_current_delay_bl, # not working currently --> do in evaluation
}

SAVE_MODELS = ["nn", "ngb", "simple_median", "random_forest", "ngb_lognormal"]


def get_metrics(temp_df, save_path=None):
    # fill table and convert to seconds
    temp_df["final_delay"] = temp_df["final_delay"] * 60
    temp_df["pred"] = temp_df["pred"] * 60
    temp_df["unc"] = temp_df["unc"] * 60
    temp_df["MAE"] = np.abs(temp_df["pred"] - temp_df["final_delay"])
    temp_df["MAE_min"] = temp_df["MAE"] / 60
    temp_df["MSE"] = (temp_df["pred"] - temp_df["final_delay"]) ** 2
    temp_df["MSE_min"] = (temp_df["pred"] / 60 - temp_df["final_delay"] / 60) ** 2
    temp_df["pi_width"] = temp_df["interval_high_bound"] - temp_df["interval_low_bound"]

    model_res_dict = {}

    save_df = temp_df.drop(["normed_obs_count", "normed_time_to_end_plan"], axis=1, errors="ignore")
    save_path_obs = int(save_path.split("_")[-1])
    save_df = save_df[save_df["obs_count"] == save_path_obs]
    save_df.to_csv(save_path + "_res.csv", index=False)
    return model_res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_path", type=str, default=os.path.join("data", "data_enriched.csv"))
    parser.add_argument("-m", "--model_dir", default="test", type=str, help="name of model directory")
    parser.add_argument("-v", "--version", default=2, type=int, help="version of feature set")
    parser.add_argument("-p", "--plot", action="store_true", help="plot?")
    args = parser.parse_args()

    data = pd.read_csv(args.inp_path)
    os.makedirs(os.path.join("outputs", args.model_dir), exist_ok=True)

    # split into train, val and test
    train_set, val_set, test_set = split_train_test(data)  # , save_path="data/data_enriched.csv")

    # select suitable features for ML models
    use_features = get_features(data.columns, version=args.version)

    # remove features that are not relevant when only training on one observation
    use_features = [f for f in use_features if f not in ["feat_obs_count", "feat_time_since_stop"]]

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
        ["train_id_daily", "train_id", "day", "DIR", "cat", "stops", "remaining_runtime", "obs_count", "delay_dep",]
    ].copy()
    # ca 10000 - 60000 --> 60 bins
    basic_df["time_to_end_plan"] = test_set["feat_time_to_end_plan"].values
    basic_df["normed_time_to_end_plan"] = (test_set["feat_time_to_end_plan"].values / 100).astype(int)

    # get baseline uncertainties (std of final delay per train ID)
    _, unc_bl = simple_mean_bl(train_set, test_set)
    _, unc_bl_val = simple_mean_bl(train_set, val_set)

    print("DATA SHAPES", train_set_nn_x.shape, train_set_nn_y.shape, test_set_nn_x.shape, test_set_nn_y.shape)
    res_dict = {}

    # Test models
    model_weights_base = args.model_dir

    # for model_type in [
    #     # "simple_current_delay",
    #     "simple_median",
    #     "simple_mean",
    #     "simple_avg",
    #     "ngb",
    #     "ngb_lognormal",
    #     "nn",
    #     "random_forest",
    #     "nn_aleatoric",
    #     "nn_dropout",
    # ]:
    for model_name in os.listdir(os.path.join("trained_models", args.model_dir)):
        if model_name[-3:] == "png" or model_name[0] == ".":
            continue
        model_type = "nn"
        # check whether pretrained model exists
        trained_model_exists = os.path.exists(
            os.path.join("trained_models", args.model_dir, model_name)
        ) or os.path.exists(os.path.join("trained_models", args.model_dir, model_type + ".p"))
        if not trained_model_exists and "simple" not in model_type:
            print(f"Skipping {model_type} because no pretrained model available.")
            continue
        # get the correct test function
        model_func = MODEL_FUNC_TEST[model_type]

        model_weights = os.path.join(model_weights_base, model_name)

        print("-------------- ", model_name, "--------------")
        if "simple" in model_type:
            pred, unc = model_func(train_set, test_set)
        else:
            pred, unc = model_func(model_weights, test_set_nn_x, dropout_rate=0.5, return_params=False)

        # add to the other metrics
        # for model_type_name, unc_est in zip([model_type, model_type + "_unc_bl"], [unc, unc_bl]):
        if True:
            model_type_name = model_type
            unc_est = unc
            if "simple" in model_type and "bl" in model_type_name:
                # only do the bl evaluation for the non-simple models (otherwise same result)
                continue

            # fill the table with our basic information
            temp_df = basic_df.copy()
            temp_df["pred"] = pred
            temp_df["unc"] = unc_est
            temp_df["final_delay"] = test_set_nn_y

            # Calibration --> run on val set
            if "simple" in model_type:
                pred_val, unc_val = model_func(train_set, val_set)
            else:
                pred_val, unc_val = model_func(model_weights, val_set_nn_x, dropout_rate=0.5)
            # use bl uncertainty if required
            if "unc_bl" in model_type_name:
                unc_val = unc_bl_val
            # get interval bounds
            quantiles = calibrate_pi(val_set_nn_y, pred_val, unc_val)
            print("Quantiles", quantiles)
            intervals = get_intervals(temp_df["pred"].values, temp_df["unc"].values, quantiles)
            temp_df["interval_low_bound"] = intervals[:, 0] * 60
            temp_df["interval_high_bound"] = intervals[:, 1] * 60

            # Likelihood:
            if "lognormal" in model_type_name:
                s, scale = test_ngb_lognormal(model_weights, test_set_nn_x, return_params=True)
                temp_df = likelihood_lognormal(temp_df, s, scale)
            else:
                best_factor = calibrate_likely(val_set_nn_y, pred_val, unc_val)
                print("Best factor for likelihood", best_factor)
                temp_df = add_likely(temp_df, factor=best_factor)

            # get metrics and save in final dictionary
            # save_csv_path = (
            #     os.path.join("outputs", args.model_dir, model_type_name) if model_type_name in SAVE_MODELS else None
            # )
            save_csv_path = os.path.join("outputs", args.model_dir, model_name)
            res_dict[model_type_name] = get_metrics(temp_df, save_path=save_csv_path)
            print("metrics", res_dict[model_type_name])

    result_table = pd.DataFrame(res_dict).swapaxes(1, 0).sort_values(["mean_pi_width"]).round(3)
    print(result_table)
    result_table.to_csv(os.path.join("outputs", args.model_dir, "results_summary.csv"))

