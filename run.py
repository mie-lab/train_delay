import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd

from train_delay.baselines import run_simple_baselines
from train_delay.metrics import get_metrics
from train_delay.mlp_model import fit_mlp_aleatoric, fit_mlp_test_time_dropout
from train_delay.rf_model import fit_rf_model


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


def select_features(cols):
    train_id_feats = [col for col in cols if col.startswith("train_id_SBB")]
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
        # train ID
    ]  # + train_id_feats
    return use_features


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


if __name__ == "__main__":

    data = pd.read_csv(os.path.join("data", "data_enriched.csv"))
    data.index.name = "id"

    # split into train, val and test
    train_set, val_set, test_set = split_train_test(data)

    # run baselines
    res_dict = run_simple_baselines(train_set, val_set)
    print("Intermediate results of baselines:")
    print(pd.DataFrame(res_dict).swapaxes(1, 0).sort_values("MSE"))

    # make suitable data for ML models
    use_features = select_features(data.columns)
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

    print("DATA SHAPES", train_set_nn_x.shape, train_set_nn_y.shape, val_set_nn_x.shape, val_set_nn_y.shape)

    # Train MLP with uncertainty
    model_weights = ["trained_models/random_forest.p", "trained_models/aleatoric_nn", "trained_models/dropout_nn"]
    # model_weights = [None, None, None]
    for model_type, model_weights in zip(["rf", "nn_aleatoric", "nn_dropout"], model_weights):
        print("-------------- ", model_type, "--------------")
        if model_type == "rf":
            pred, unc = fit_rf_model(
                train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y, load_model=model_weights
            )
        elif model_type == "nn_aleatoric":
            pred, unc = fit_mlp_aleatoric(
                train_set_nn_x, train_set_nn_y, val_set_nn_x, val_set_nn_y, epochs=1, load_model=model_weights
            )
        elif model_type == "nn_dropout":
            pred, unc = fit_mlp_test_time_dropout(
                train_set_nn_x,
                train_set_nn_y,
                val_set_nn_x,
                val_set_nn_y,
                epochs=5,
                dropout_rate=0.5,
                load_model=model_weights,
            )
        else:
            raise NotImplementedError

        # plot
        plot_by_obs_count(
            pred, unc, val_set_nn_y, val_set_x["obs_count"].values, save_path=os.path.join("outputs", model_type),
        )

        # add to the other metrics
        temp_df = pd.DataFrame(pred, columns=["pred"])
        temp_df["final_delay"] = val_set_nn_y
        temp_df["unc"] = unc
        metrics_nn_torch = get_metrics(temp_df[["pred", "final_delay", "unc"]], model_type)
        # save in general dictionary
        print("metrics", metrics_nn_torch[model_type])
        res_dict[model_type] = metrics_nn_torch[model_type]

    result_table = pd.DataFrame(res_dict).swapaxes(1, 0).sort_values("MSE")
    print(result_table)
    result_table.to_csv(os.path.join("outputs", "results_summary.csv"))

