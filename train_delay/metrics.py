import numpy as np
import sklearn.metrics as me
from scipy.stats import pearsonr, spearmanr, norm


def get_metrics(pred_and_unc, name="model"):
    print(name, "NaNs in predictions (during get_metrics):", 1 - len(pred_and_unc.dropna()) / len(pred_and_unc))
    pred_and_unc = pred_and_unc.dropna()
    pred = pred_and_unc["pred"].values
    gt = pred_and_unc["final_delay"].values
    res_dict = {name: {"MSE": me.mean_squared_error(pred, gt), "MAE": me.mean_absolute_error(pred, gt)}}
    # -------- UNCERTAINTY metrics --------
    if "unc" in pred_and_unc.columns:
        # compute rmse for correlation metrics
        rmse = np.sqrt((pred - gt) ** 2)
        unc = pred_and_unc["unc"].values
        print("Correlation mse & unc: ", pearsonr(unc, rmse ** 2)[0])
        print("Correlation rmse & unc: ", pearsonr(unc, rmse)[0])
        print("Spearman: ", spearmanr(unc, rmse)[0])
        # negative log likelihood:
        nll = []
        for _, row in pred_and_unc.iterrows():
            #     print(type(row["pred_mean"]))
            prob = norm.pdf(row["final_delay"], row["pred"], row["unc"])
            nll.append(prob)
        nll = -1 * np.log(np.array(nll))
        res_dict[name]["nll"] = np.mean(nll)
        # store mean uncertainty to check the nll stuff
        res_dict[name]["mean_unc"] = np.mean(unc)
        # correlations
        res_dict[name]["spearman_r"] = spearmanr(unc, rmse)[0]
        res_dict[name]["pearsonr"] = pearsonr(unc, rmse)[0]
        # PI
        # devide into val and test for prediction intervals
        # TODO: should take actual val set or train set instead
        interval_val_set = pred_and_unc[: len(pred_and_unc) // 2]
        interval_test_set = pred_and_unc[len(pred_and_unc) // 2 :]
        quantiles = calibrate_pi(
            interval_val_set["final_delay"].values, interval_val_set["pred"].values, interval_val_set["unc"].values
        )
        intervals = get_intervals(interval_test_set["pred"].values, interval_test_set["unc"].values, quantiles)
        res_dict[name]["coverage"] = coverage(intervals, interval_test_set["final_delay"].values)
        res_dict[name]["mean_pi_width"] = mean_pi_width(intervals)
    return res_dict


def calibrate_pi(gt, pred, unc, alpha: float = 0.1):
    n = len(pred)
    quant = np.ceil((1 - alpha) * (n + 1)) / n
    # floor because of zero division problems
    unc_floored = np.clip(unc, a_min=1e-4, a_max=None)
    return np.quantile(np.abs(gt - pred) / unc_floored, quant, axis=0)


def get_intervals(pred, unc, quantiles):
    return np.stack(((pred - unc * quantiles), (pred + unc * quantiles)), axis=1)


def coverage(intervals, gt):
    greater_lower_bound = gt >= intervals[:, 0]
    smaller_upper_bound = gt <= intervals[:, 1]
    coverage = np.sum(greater_lower_bound & smaller_upper_bound) / len(gt)
    return coverage


def mean_pi_width(intervals):
    return np.mean(intervals[:, 1] - intervals[:, 0])
