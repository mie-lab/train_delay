import numpy as np
import sklearn.metrics as me
from scipy.stats import pearsonr, spearmanr, norm, lognorm
from config import OUTLIER_CUTOFF


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
    model_res_dict["coverage"] = coverage(temp_df)

    # add the average values for these columns
    cols_to_agg = temp_df[["unc", "pi_width", "MSE", "MAE", "Likely_30", "Likely_45", "Likely_15"]].rename(
        columns={"pi_width": "mean_pi_width", "unc": "mean_unc"}
    )
    model_res_dict.update(cols_to_agg.mean().to_dict())

    model_res_dict["spearman_r"] = spearmanr(temp_df["unc"], np.sqrt(temp_df["MSE"]))[0]
    model_res_dict["pearsonr"] = pearsonr(temp_df["unc"], np.sqrt(temp_df["MSE"]))[0]

    if save_path is not None:
        temp_df.drop(
            ["obs_point_id", "normed_obs_count", "time_to_end_plan", "normed_time_to_end_plan"], axis=1, errors="ignore"
        ).to_csv(save_path + "_res.csv", index=False)
    return model_res_dict


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


def likelihood_lognormal(temp_df, s, scale, radius=[0.25, 0.5, 0.75]):
    for r in radius:
        temp_df["Likely_" + str(int(60 * r))] = lognorm.cdf(
            temp_df["final_delay"] + OUTLIER_CUTOFF + r, s, scale=scale
        ) - lognorm.cdf(temp_df["final_delay"] + OUTLIER_CUTOFF - r, s, scale=scale)
    return temp_df
