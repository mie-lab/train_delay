import numpy as np
import sklearn.metrics as me
from scipy.stats import pearsonr, spearmanr, norm


def get_metrics(pred_and_unc, save_path=None):
    assert len(pred_and_unc.dropna()) == len(pred_and_unc), "Error: NaNs in predictions"
    # print(name, "NaNs in predictions (during get_metrics):", 1 - len(pred_and_unc.dropna()) / len(pred_and_unc))
    pred = pred_and_unc["pred"].values
    gt = pred_and_unc["final_delay"].values
    # init res dict
    res_dict_model = {"MSE": me.mean_squared_error(pred, gt), "MAE": me.mean_absolute_error(pred, gt)}
    # -------- UNCERTAINTY metrics --------
    # compute rmse for correlation metrics
    rmse = np.sqrt((pred - gt) ** 2)
    unc = pred_and_unc["unc"].values
    print("Correlation mse & unc: ", pearsonr(unc, rmse ** 2)[0])
    print("Correlation rmse & unc: ", pearsonr(unc, rmse)[0])
    print("Spearman: ", spearmanr(unc, rmse)[0])
    # negative log likelihood:
    res_dict_model["nll"] = np.mean(pred_and_unc["nll"])
    # store mean uncertainty to check the nll stuff
    res_dict_model["mean_unc"] = np.mean(unc)
    # correlations
    res_dict_model["spearman_r"] = spearmanr(unc, rmse)[0]
    res_dict_model["pearsonr"] = pearsonr(unc, rmse)[0]
    # PI
    # devide into val and test for prediction intervals
    res_dict_model["coverage"] = coverage(pred_and_unc)
    res_dict_model["mean_pi_width"] = mean_pi_width(pred_and_unc)
    if save_path is not None:
        pred_and_unc.to_csv(save_path + "_res.csv", index=False)
    return res_dict_model


def add_nll_metric(res_df):
    nll = []
    for _, row in res_df.iterrows():
        #     print(type(row["pred_mean"]))
        prob = norm.pdf(row["final_delay"], row["pred"], row["unc"])
        nll.append(prob)
    nll = -1 * np.log(np.array(nll))
    res_df["nll"] = nll
    return res_df


def calibrate_pi(gt, pred, unc, alpha: float = 0.1):
    n = len(pred)
    quant = np.ceil((1 - alpha) * (n + 1)) / n
    # floor because of zero division problems
    unc_floored = np.clip(unc, a_min=1e-4, a_max=None)
    return np.quantile(np.abs(gt - pred) / unc_floored, quant, axis=0)


def get_intervals(pred, unc, quantiles):
    return np.stack(((pred - unc * quantiles), (pred + unc * quantiles)), axis=1)


def coverage(res_df):
    greater_lower_bound = res_df["final_delay"] >= res_df["interval_low_bound"]
    smaller_upper_bound = res_df["final_delay"] <= res_df["interval_high_bound"]
    coverage = np.sum(greater_lower_bound & smaller_upper_bound) / len(res_df)
    return coverage


def mean_pi_width(res_df):
    return np.mean(res_df["interval_high_bound"] - res_df["interval_low_bound"])
