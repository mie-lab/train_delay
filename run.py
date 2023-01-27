from pyexpat import model
import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

from train_delay.baselines import run_simple_baselines, simple_avg_bl, simple_median_bl, simple_mean_bl, overall_avg
from train_delay.metrics import (
    get_metrics,
    calibrate_pi,
    add_nll_metric,
    get_intervals,
    add_metrics_in_sec,
    calibrate_likely,
    add_likely,
)
from train_delay.mlp_model import test_test_time_dropout, test_aleatoric, test_unc_nn
from train_delay.ngboost_model import test_ngboost
from train_delay.rf_model import test_random_forest
from train_delay.gaussian_process import test_gaussian_process

MODEL_FUNC_TEST = {
    "nn_dropout": test_test_time_dropout,
    "nn_aleatoric": test_aleatoric,
    "random_forest": test_random_forest,
    "gaussian_process": test_gaussian_process,
    "nn": test_unc_nn,
    "ngb": test_ngboost,
    "simple_median": simple_median_bl,
    "simple_mean": simple_mean_bl,
    "simple_avg": overall_avg,
}

SAVE_MODELS = ["nn", "ngb", "simple_median", "random_forest"]


def split_train_test(data, ratio=0.8, save_path=None):
    # split into train and test by day
    all_days = data["day"].unique()
    cutoff = round(len(all_days) * ratio)
    test_val_cutoff = (len(all_days) - cutoff) // 2 + cutoff
    train_days, val_days, test_days = all_days[:cutoff], all_days[cutoff:test_val_cutoff], all_days[test_val_cutoff:]
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


def get_features(columns, version=2):
    if version == 1:
        use_features = [
            "delay_dep",
            "feat_obs_count",
            "feat_time_to_end_real",
            "feat_time_to_end_plan",
            "feat_weather_tavg",
            "feat_weather_tmin",
            "feat_weather_tmax",
            "feat_weather_prcp",
            # "feat_weather_wdir",
            "feat_weather_wspd",
            "feat_weather_wpgt",
            "feat_weather_pres",  # weather features
            # 'feat_weather_snow' 'feat_weather_tsun'
            "feat_trip_final_arr_plan_hour",
            "feat_trip_final_arr_plan_day",
            "feat_trip_final_arr_plan_hour_sin",
            "feat_trip_final_arr_plan_hour_cos",
            "feat_trip_final_arr_plan_day_sin",
            "feat_trip_final_arr_plan_day_cos",  # time features
        ]
    # train_id_feats = [col for col in cols if col.startswith("train_id_SBB")]
    elif version == 2:
        # All features
        use_features = ["delay_dep"] + [
            feat
            for feat in columns
            if feat.startswith("feat")
            # and not feat.startswith("feat_weather")
            # note: weather is not contained anymore in any case
        ]
    elif version == 3:
        print("Using features that are comparable features to Markov chain")
        feats_necessary = ["delay_dep", "feat_obs_count", "feat_time_to_end_plan", "feat_avg_prev_delay"]
        hist_delay = [feat for feat in columns if feat.startswith("feat_delay_day")]
        hist_final_delay = [feat for feat in columns if feat.startswith("feat_final_delay-day")]
        use_features = feats_necessary + hist_delay + hist_final_delay
        print(use_features)
    else:
        raise NotImplementedError
    return use_features


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

    basic_df = pd.DataFrame()
    # add basic information
    basic_df["train_id"] = test_set["train_id_daily"].values
    basic_df["day"] = test_set["day"].values
    # get baseline uncertainties (std of final delay per train ID)
    _, unc_bl = simple_mean_bl(train_set, test_set)

    print("DATA SHAPES", train_set_nn_x.shape, train_set_nn_y.shape, test_set_nn_x.shape, test_set_nn_y.shape)
    res_dict = {}

    # Test models
    model_weights = args.model_dir
    for model_type in [
        "simple_median",
        "simple_mean",
        "simple_avg",
        "ngb",
        "nn",
        "random_forest",
        "nn_aleatoric",
        "nn_dropout",
    ]:
        # check whether pretrained model exists
        trained_model_exists = os.path.exists(
            os.path.join("trained_models", args.model_dir, model_type)
        ) or os.path.exists(os.path.join("trained_models", args.model_dir, model_type + ".p"))
        if not trained_model_exists and "simple" not in model_type:
            print(f"Skipping {model_type} because no pretrained model available.")
            continue
        # get the correct test function
        model_func = MODEL_FUNC_TEST[model_type]

        print("-------------- ", model_type, "--------------")
        if "simple" in model_type:
            pred, unc = model_func(train_set, test_set)
        else:
            pred, unc = model_func(model_weights, test_set_nn_x, dropout_rate=0.5)

        # put results in one df
        res_df = basic_df.copy()
        # add results
        res_df["pred"] = pred
        res_df["final_delay"] = test_set_nn_y
        res_df["unc"] = unc
        # use 100 bins of relative obs point
        res_df["obs_count"] = test_set["obs_count"].values
        res_df["obs_point_id"] = test_set["obs_point_id"].values
        res_df["normed_obs_count"] = (test_set["feat_obs_count"].values * 100).astype(int)
        # ca 10000 - 60000 --> 60 bins
        res_df["time_to_end_plan"] = test_set["feat_time_to_end_plan"].values
        res_df["normed_time_to_end_plan"] = (test_set["feat_time_to_end_plan"].values / 100).astype(int)
        # add metrics
        res_df["MSE"] = (res_df["final_delay"] - res_df["pred"]) ** 2
        res_df["MAE"] = (res_df["final_delay"] - res_df["pred"]).abs()

        # add to the other metrics
        for model_type_name, unc_est in zip([model_type, model_type + "_unc_bl"], [unc, unc_bl]):
            if "simple" in model_type and "bl" in model_type_name:
                # only do the bl evaluation for the non-simple models (otherwise same result)
                continue
            # COPY FOR SPECIFIC UNC
            temp_df = res_df.copy()
            temp_df["unc"] = unc_est
            # add uncertainty specific stuff
            # calibration --> run on val set
            if "simple" in model_type:
                pred_val, unc_val = model_func(train_set, val_set)
            else:
                pred_val, unc_val = model_func(model_weights, val_set_nn_x, dropout_rate=0.5)
            # get interval bounds
            quantiles = calibrate_pi(val_set_nn_y, pred_val, unc_val)
            print("Quantiles", quantiles)
            intervals = get_intervals(temp_df["pred"].values, temp_df["unc"].values, quantiles)
            temp_df["interval_low_bound"] = intervals[:, 0]
            temp_df["interval_high_bound"] = intervals[:, 1]
            temp_df["covered"] = (temp_df["final_delay"] >= temp_df["interval_low_bound"]) & (
                temp_df["final_delay"] <= temp_df["interval_high_bound"]
            )
            # add nll metric
            # temp_df = add_nll_metric(temp_df)
            best_factor = calibrate_likely(val_set_nn_y, pred_val, unc_val)
            print("Best factor for likelihood", best_factor)
            temp_df = add_likely(temp_df, factor=best_factor)
            # add metrics in seconds:
            temp_df = add_metrics_in_sec(temp_df)

            # get metrics and save in final dictionary
            save_csv_path = (
                os.path.join("outputs", args.model_dir, model_type_name) if model_type_name in SAVE_MODELS else None
            )
            res_dict[model_type_name] = get_metrics(temp_df, save_path=save_csv_path)
            print("metrics", res_dict[model_type_name])

        # plot results by obs count
        if args.plot:
            save_plot_obscount = os.path.join("outputs", args.model_dir, model_type)
            plot_by_obs_count(res_df, col="normed_obs_count", save_path=save_plot_obscount)
            # plot by time to end
            plot_by_obs_count(res_df, col="normed_time_to_end_plan", save_path=save_plot_obscount)

    result_table = pd.DataFrame(res_dict).swapaxes(1, 0).sort_values(["MSE", "mean_pi_width"]).round(3)
    print(result_table)
    result_table.to_csv(os.path.join("outputs", args.model_dir, "results_summary.csv"))

