import numpy as np
import pandas as pd
from .metrics import get_metrics
from config import OUTLIER_CUTOFF


def run_simple_baselines(train_set, val_set):
    # random predictions
    pred_random_model = val_set[["train_id_daily", "final_delay"]].copy()
    pred_random_model["pred"] = np.random.rand(len(val_set)) * 2 - 1
    pred_random_model["unc"] = np.random.rand(len(val_set))
    metrics_pred_random_model = get_metrics(pred_random_model, "random")

    # average per train ID
    pred_simple_avg = simple_avg_bl(train_set, val_set)
    metrics_pred_simple_avg = get_metrics(pred_simple_avg, "simple avg")

    # median per train ID
    pred_simple_median = simple_avg_bl(train_set, val_set, agg_func="median")
    metrics_pred_simple_median = get_metrics(pred_simple_median, "simple median")

    # overall average
    pred_overall_avg = val_set[["train_id_daily", "final_delay"]].copy()
    pred_overall_avg["pred"] = train_set["final_delay"].mean()
    pred_overall_avg["unc"] = train_set["final_delay"].std()
    metrics_pred_overall_avg = get_metrics(pred_overall_avg, "overall avg")

    # # KNN
    # pred_knn_bl = knn_weighted(train_set, val_set)
    # metrics_pred_knn_bl = get_metrics(pred_knn_bl[["pred", "final_delay"]], "knn")

    # prepare output
    collection_of_results = [
        metrics_pred_random_model,
        # metrics_pred_knn_bl,
        metrics_pred_simple_avg,
        metrics_pred_simple_median,
        metrics_pred_overall_avg,
    ]
    res_dict = {}
    for res in collection_of_results:
        keys = list(res.keys())
        assert len(keys) == 1
        res_dict[keys[0]] = res[keys[0]]
    return res_dict


def simple_avg_bl(train_data, test_data, agg_func="mean"):
    output_pred_df = test_data[["train_id_daily", "final_delay"]].copy()
    for train_id, id_testset in test_data.groupby("train_id_daily"):
        id_trainset = train_data[train_data["train_id_daily"] == train_id]
        if agg_func == "mean":
            avg_delay = id_trainset["final_delay"].mean()
        elif agg_func == "median":
            avg_delay = id_trainset["final_delay"].median()
        simple_unc = id_trainset["final_delay"].std()
        if pd.isna(avg_delay):
            avg_delay = 0  # TODO: train id does not appear in train_set, only in test
            simple_unc = 1  # use 1 as an expected uncertainty for unseen train IDs

        output_pred_df.loc[output_pred_df["train_id_daily"] == train_id, "pred"] = avg_delay
        output_pred_df.loc[output_pred_df["train_id_daily"] == train_id, "unc"] = simple_unc
    return output_pred_df


def simple_median_bl(train_data, test_data, **kwargs):
    output_pred_df = simple_avg_bl(train_data, test_data, agg_func="median")
    return output_pred_df["pred"].values, output_pred_df["unc"].values


def simple_current_delay_bl(train_data, test_data):
    output_for_unc = simple_avg_bl(train_data, test_data, agg_func="median")
    return (
        np.clip(test_data["delay_dep"].values, -OUTLIER_CUTOFF * 60, OUTLIER_CUTOFF * 60) / 60,
        output_for_unc["unc"].values,
    )


def simple_mean_bl(train_data, test_data, **kwargs):
    output_pred_df = simple_avg_bl(train_data, test_data, agg_func="mean")
    return output_pred_df["pred"].values, output_pred_df["unc"].values


def overall_avg(train_data, test_data, **kwargs):
    pred = np.zeros(len(test_data)) + train_data["final_delay"].mean()
    unc = np.zeros(len(test_data)) + train_data["final_delay"].std()
    return pred, unc


def knn_weighted(train_data, test_data):
    overall_avg = train_data["final_delay"].mean()
    list_of_historic_delays = train_data.groupby(["train_id_daily", "obs_point_id"]).agg(
        {"final_delay": tuple, "delay_dep": tuple}
    )  # or obs_count
    test_w_historic = test_data.merge(
        list_of_historic_delays,
        left_on=["train_id_daily", "obs_point_id"],
        right_index=True,
        suffixes=("", "_historic"),
    )

    def get_weighted_nn(row):
        historic_delay_list = row["delay_dep_historic"]
        weights = np.array([abs(row["delay_dep"] - hd) for hd in historic_delay_list])
        return np.sum((weights / np.sum(weights)) * np.array(list(row["final_delay_historic"])))

    test_w_historic["pred"] = test_w_historic.apply(get_weighted_nn, axis=1)

    # TODO: can also make a NN baseline of all data
    # here: only trains with the same id are okay
    return test_w_historic
