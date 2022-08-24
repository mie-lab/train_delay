import numpy as np
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

from train_delay.baselines import run_simple_baselines
from train_delay.metrics import get_metrics
from train_delay.mlp_model import fit_mlp_aleatoric, fit_mlp_test_time_dropout
from train_delay.rf_model import fit_rf_model

MODEL_FUNC_DICT = {"nn_dropout": fit_mlp_test_time_dropout, "nn_aleatoric": fit_mlp_aleatoric, "rf": fit_rf_model}


def split_train_test(data, ratio=0.8):
    all_days = data["day"].unique()
    cutoff = round(len(all_days) * ratio)
    test_val_cutoff = (len(all_days) - cutoff) // 2 + cutoff
    train_days, val_days, test_days = all_days[:cutoff], all_days[cutoff:test_val_cutoff], all_days[test_val_cutoff:]
    print("cuttoff after train:", cutoff, "cutoff after val", test_val_cutoff, "total nr days", len(all_days))
    out = []
    for days in [train_days, val_days, test_days]:
        out.append(data[data["day"].isin(days)])
    return tuple(out)


def plot_by_obs_count(pred, unc, gt, obs_count, save_path="outputs/by_obs_count"):
    df_unc = pd.DataFrame()
    df_unc["pred_mean"] = pred
    df_unc["unc"] = unc
    df_unc["obs_count"] = obs_count
    df_unc["ground_truth"] = gt
    df_unc["mse"] = ((df_unc["pred_mean"] - df_unc["ground_truth"]).values) ** 2
    df_unc.to_csv(save_path + ".csv")

    plt.figure(figsize=(8, 5))
    ax = plt.subplot(111)
    ax.plot(df_unc.groupby("obs_count").agg({"mse": "mean"}), label="error")
    ax2 = ax.twinx()
    ax2.plot(df_unc.groupby("obs_count").agg({"unc": "mean"}), label="uncertainty", c="orange")
    plt.title("Uncertainty by observation count")
    plt.legend()
    plt.savefig(save_path + ".png")
    # plt.show()


def get_train_val_test(train_set, val_set, test_set, use_features, training=False):
    # retrict to these features plus the predicted variable
    train_set = train_set[use_features + ["final_delay"]].dropna()
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
        return train_set.index, test_set.index, train_set_nn_x, train_set_nn_y, test_set_nn_x, test_set_nn_y


def get_features(columns, version=2):
    if version == 1:
        use_features = [
            "delay_dep",
            "obs_count",
            "time_to_end_real",
            "time_to_end_plan",
            "feat_weather_tavg",
            "feat_weather_tmin",
            "feat_weather_tmax",
            "feat_weather_prcp",
            "feat_weather_wdir",
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
        use_features = ["delay_dep", "obs_count"] + [feat for feat in columns if feat.startswith("feat")]
    else:
        raise NotImplementedError
    return use_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_path", type=str, default=os.path.join("data", "data_enriched.csv"))
    parser.add_argument("-m", "--model_dir", default="best_models", type=str, help="name of model directory")
    args = parser.parse_args()

    data = pd.read_csv(args.inp_path)
    data.index.name = "id"

    # split into train, val and test
    train_set, val_set, test_set = split_train_test(data)

    # select suitable features for ML models
    use_features = get_features(data.columns, version=2)

    # preprocess and dropn the ones with NaN features
    (
        train_set_index,
        test_set_index,
        train_set_nn_x,
        train_set_nn_y,
        test_set_nn_x,
        test_set_nn_y,
    ) = get_train_val_test(train_set, val_set, test_set, use_features)

    # run baselines
    print(len(train_set), len(test_set))
    train_set = train_set.loc[train_set_index]  # restrict to the ones that are also used for the other models
    test_set = test_set.loc[test_set_index]
    print(len(train_set), len(test_set))
    res_dict = run_simple_baselines(train_set, test_set)
    print("Intermediate results of baselines:")
    print(pd.DataFrame(res_dict).swapaxes(1, 0).sort_values("MSE"))

    print("DATA SHAPES", train_set_nn_x.shape, train_set_nn_y.shape, test_set_nn_x.shape, test_set_nn_y.shape)

    # Train MLP with uncertainty
    model_weights = args.model_dir
    # model_weights = None
    for model_type in ["rf", "nn_aleatoric", "nn_dropout"]:
        model_func = MODEL_FUNC_DICT[model_type]

        print("-------------- ", model_type, "--------------")
        pred, unc = model_func(
            train_set_nn_x, train_set_nn_y, test_set_nn_x, test_set_nn_y, dropout_rate=0.5, load_model=model_weights
        )

        # plot
        plot_by_obs_count(
            pred, unc, test_set_nn_y, test_set["obs_count"].values, save_path=os.path.join("outputs", model_type),
        )

        # add to the other metrics
        temp_df = pd.DataFrame(pred, columns=["pred"])
        temp_df["final_delay"] = test_set_nn_y
        temp_df["unc"] = unc
        metrics_nn_torch = get_metrics(temp_df[["pred", "final_delay", "unc"]], model_type)
        # save in general dictionary
        print("metrics", metrics_nn_torch[model_type])
        res_dict[model_type] = metrics_nn_torch[model_type]

    result_table = pd.DataFrame(res_dict).swapaxes(1, 0).sort_values("MSE")
    print(result_table)
    result_table.to_csv(os.path.join("outputs", "results_summary.csv"))

