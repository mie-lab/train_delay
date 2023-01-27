import numpy as np
import sklearn.metrics as me
from scipy.stats import pearsonr, spearmanr, norm


def get_metrics(pred_and_unc, save_path=None):
    assert len(pred_and_unc.dropna()) == len(pred_and_unc), "Error: NaNs in predictions"
    # print(name, "NaNs in predictions (during get_metrics):", 1 - len(pred_and_unc.dropna()) / len(pred_and_unc))
    pred = pred_and_unc["pred"].values
    gt = pred_and_unc["final_delay"].values
    # init res dict
    res_dict_model = {"MSE": pred_and_unc["MSE"].mean(), "MAE": pred_and_unc["MAE"].mean()}
    # -------- UNCERTAINTY metrics --------
    # compute rmse for correlation metrics
    rmse = np.sqrt((pred - gt) ** 2)
    unc = pred_and_unc["unc"].values
    print("Correlation mse & unc: ", pearsonr(unc, rmse ** 2)[0])
    print("Correlation rmse & unc: ", pearsonr(unc, rmse)[0])
    print("Spearman: ", spearmanr(unc, rmse)[0])
    # negative log likelihood:
    res_dict_model["Likelihood_30"] = np.mean(pred_and_unc["Likely_30"])
    # store mean uncertainty to check the nll stuff
    res_dict_model["mean_unc"] = np.mean(unc) * 60
    # correlations
    res_dict_model["spearman_r"] = spearmanr(unc, rmse)[0]
    res_dict_model["pearsonr"] = pearsonr(unc, rmse)[0]
    # PI
    # devide into val and test for prediction intervals
    res_dict_model["coverage"] = coverage(pred_and_unc)
    res_dict_model["mean_pi_width"] = pred_and_unc["pi_width"].mean()
    if save_path is not None:
        pred_and_unc.to_csv(save_path + "_res.csv", index=False)
    return res_dict_model


def add_nll_metric(res_df):
    nll = []
    for _, row in res_df.iterrows():
        #     print(type(row["pred_mean"]))
        prob = norm.pdf(row["final_delay"], row["pred"], row["unc"])
        # Area under the PDF with +- 30s around ground truth
        nll.append(prob)
    res_df["my_likelihood"] = nll
    nll = -1 * np.log(np.array(nll))
    res_df["nll"] = nll
    return res_df


def add_likely(res_df, factor=1, radius=[0.25, 0.5, 0.75]):
    for r in radius:
        res_df["Likely_" + str(int(60 * r))] = norm.cdf(
            res_df["final_delay"] + r, res_df["pred"], res_df["unc"] * factor
        ) - norm.cdf(res_df["final_delay"] - r, res_df["pred"], res_df["unc"] * factor)
    return res_df


def calibrate_likely(val_gt, val_pred, val_unc, radius=0.25):
    best_factor, best_median = 1, 0
    for factor in np.arange(0.1, 2.1, 0.1):
        val_unc_scaled = val_unc * factor
        #     row["unc_scaled"] = row["unc"]**2
        likelyhood = norm.cdf(val_gt + radius, val_pred, val_unc_scaled) - norm.cdf(
            val_gt - radius, val_pred, val_unc_scaled
        )
        if np.median(likelyhood) > best_median:
            best_factor = factor
            best_median = np.median(likelyhood)
    return best_factor


def add_metrics_in_sec(res_df):
    # MAE
    res_df["MAE_min"] = res_df["MAE"].copy()
    res_df["MAE"] = res_df["MAE"] * 60
    # pi width
    res_df["pi_width"] = (res_df["interval_high_bound"] - res_df["interval_low_bound"]) * 60
    # MSE
    res_df["MSE_min"] = res_df["MSE"].copy()
    res_df["MSE"] = (res_df["final_delay"] * 60 - res_df["pred"] * 60) ** 2
    return res_df


def calibrate_pi(gt, pred, unc, alpha: float = 0.1):
    n = len(pred)
    quant = np.ceil((1 - alpha) * (n + 1)) / n
    # floor because of zero division problems
    unc_floored = np.clip(unc, a_min=1e-4, a_max=None)
    # q is one scalar value
    q = np.quantile(np.abs(gt - pred) / unc_floored, quant, axis=0)
    return q


def get_intervals(pred, unc, q):
    return np.stack(((pred - unc * q), (pred + unc * q)), axis=1)


def coverage(res_df):
    greater_lower_bound = res_df["final_delay"] >= res_df["interval_low_bound"]
    smaller_upper_bound = res_df["final_delay"] <= res_df["interval_high_bound"]
    coverage = np.sum(greater_lower_bound & smaller_upper_bound) / len(res_df)
    return coverage


def mean_pi_width(res_df):
    return np.mean(res_df["interval_high_bound"] - res_df["interval_low_bound"])
