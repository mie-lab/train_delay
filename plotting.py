import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr
import seaborn as sns
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 20})


metric_mapping = {
    "Likely_45": "Likelihood (+/- 45 sec)",
    "Likely_15": "Likelihood (+/- 15 sec)",
    "Likely_30": "Likelihood (+/- 30 sec)",
    "pi_width": "Prediction interval width",
    "MAE": "MAE",
    "MSE": "MSE",
    "unc": "Estimated uncertainty",
}

model_mapping = {
    "nn": "Neural network",
    "ngb": "NGBoost (Normal)",
    "ngb_lognormal": "NGBoost (Lognormal)",
    "simple_median": "Simple median",
    "random_forest": "Random Forest",
    "mc_multi": "MC (multi)",
    "mc_2stepE": "MC (2-step, E)",
    "mc_2stepP": "MC (2-step, P)",
    "MC_2": "MC (2-step)",
    "simple_current_delay": "Current delay",
}
col_mapping = {
    "DIR": "Direction (0: down, 1: up)",
    "cat": "Train type (0: slow, 2: fast)",
    "delay_dep": "Current delay [min]",
    "final_delay": "Final delay [min]",
    "remaining_runtime": "Remaining runtime [min]",
    "distanceKM_to_final": "Remaining distance [km]",
    "daytime": "Daytime (0: before 12, 1: after 12)",
    "const": "Intercept",
}

metric_agg_dict = {metric: "mean" for metric in ["MAE", "Likely_30", "Likely_45", "Likely_15", "pi_width", "unc"]}
total_dist = 95.156198


def func_simple_powerlaw(x, beta=0.01, offset=0.5, c=1):
    # exponential decay function
    return (1 - offset) * np.exp(-x * beta) + offset  # new version without c!!


def mc_to_myformat(mc_out):
    """Function to align the output formats"""
    for l in ["15", "30", "45"]:
        try:
            mc_out["Likely_" + l] = mc_out["Likely" + l]
        except KeyError:
            mc_out["Likely_" + l] = mc_out["likely_" + l]
    try:
        mc_out["final_delay"] = mc_out["FINAL_DELAY"]
        mc_out["pred"] = mc_out["Expected_value"]
    except:
        pass
    #     mc_out["train_id"] = mc_out["train_id_daily"]
    mc_out["obs_count"] = mc_out["Forecast_at_obsCount"]
    #     mc_out["pi_width"] = 1 # TODO
    mc_out["time_to_end_plan"] = mc_out["remaining_runtime"] * 60
    return mc_out


def double_grouping_horizon(res):
    """Helper function: group by horizon"""
    grouped_by_day = res.groupby(["train_id", "horizon"]).agg(metric_agg_dict).reset_index()

    vals = grouped_by_day.groupby("horizon").agg(metric_agg_dict).reset_index()

    vals["num"] = vals["horizon"].apply(lambda x: int(x.split("(")[1].split(",")[1][:-1]))
    vals_filtered = vals[vals["num"] <= 70]
    return vals_filtered


def plot_aleatoric_vs_epistemic(metric="unc"):
    path = "2023_2_v4 - final"
    nn_a = pd.read_csv(f"outputs/{path}/nn_aleatoric_res.csv")
    nn_e = pd.read_csv(f"outputs/{path}/nn_epistemic_res.csv")
    nn = pd.read_csv(f"outputs/{path}/nn_res.csv")

    model_res_files = [nn, nn_a, nn_e]
    name = ["aleatoric + epistemic", "aleatoric", "epistemic"]

    model_res_files_horizon = []
    for i, res in enumerate(model_res_files):
        #     res = res[res["cat"] == "fast"]
        if "remaining_runtime" in res.columns:
            res["horizon"] = res["remaining_runtime"] // 5 * 5
            res = res[res["horizon"] < 75]
            # filter out higher values
            res["horizon"] = res["horizon"].map({c: f"H({int(c)},{int(c+5)}]" for c in res["horizon"].unique()})

        if "train_id" not in res.columns:
            res.rename({"train_ID": "train_id"}, axis=1, inplace=True)

        vals_filtered = double_grouping_horizon(res)
        model_res_files_horizon.append(vals_filtered)

    plt.figure(figsize=(7, 6))
    sum_df = pd.DataFrame()
    for m_name, m_file in zip(name, model_res_files_horizon):
        # need to transform it
        vals = m_file.groupby("horizon")[metric].mean()
        vals = pd.concat([vals.loc[["H(0,5]", "H(5,10]"]], vals[1:].drop("H(5,10]")])
        #     plt.plot(vals, label=m_name)
        sum_df[m_name] = vals

    plt.plot(np.arange(len(sum_df)), sum_df["aleatoric + epistemic"], c="blue")
    plt.plot(np.arange(len(sum_df)), sum_df["epistemic"], c="green")
    plt.fill_between(
        np.arange(len(sum_df)), np.zeros(len(sum_df)), sum_df["epistemic"], color="mediumseagreen", label="epistemic"
    )
    plt.fill_between(
        np.arange(len(sum_df)),
        sum_df["epistemic"],
        sum_df["aleatoric + epistemic"],
        color="cornflowerblue",
        label="aleatoric",
    )

    plt.ylim(0, 70)
    plt.xlim(0, len(sum_df) - 5)
    plt.xticks(np.arange(len(sum_df)), sum_df.index, rotation=90)
    plt.xlabel("Prediction horizon [min]")
    plt.ylabel(metric_mapping.get(metric, metric) + " [sec]")
    plt.legend(title="Uncertainty type", loc="center right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "aleatoric_epistemic.pdf"))
    plt.show()


def metric_summary_table(model_res_files):
    """Make a table with all models as rows and all metrics as columns"""
    mc_2step_index = 4
    mc_multi_index = 5
    res_summary = pd.read_csv(f"outputs/{path}/results_summary.csv")
    res_summary.set_index("Unnamed: 0", inplace=True)
    res_summary.index.name = "Model"
    res_summary = res_summary.loc[["nn", "ngb", "ngb_lognormal", "simple_median", "simple_current_delay"]]
    mapping_here = {"pi_width": "mean_pi_width"}
    series = model_res_files[mc_2step_index][["MSE", "MAE", "Likely_30", "pi_width"]].mean().rename(mapping_here)
    res_summary.loc["MC - 2 step"] = series.to_dict()
    series = model_res_files[mc_multi_index][["MSE", "MAE", "Likely_30", "pi_width"]].mean().rename(mapping_here)
    res_summary.loc["MC - multi"] = series.to_dict()

    final_res_table = (
        res_summary.sort_values("MAE")[["MSE", "MAE", "Likely_30", "mean_pi_width"]]
        .rename(columns={"mean_pi_width": "PI width"}, index=model_mapping)
        .round(3)
    )
    print(final_res_table.to_latex())


def evaluate_one_model_per_observation():
    nn_base = pd.read_csv("outputs/2023_2_v4/nn_res.csv")
    res_path = "outputs/2023_2_one_model_per_obs/"
    all_res = {}
    for f in os.listdir(res_path):
        if f[-3:] != "csv":
            continue
        res_new = pd.read_csv(os.path.join(res_path, f))
        obs_trained_on = int(f.split("_")[1])
        nn_base_obs = nn_base[nn_base["obs_count"] == obs_trained_on]
        assert len(nn_base_obs) == len(res_new)
        all_res[(obs_trained_on, "specific")] = (
            res_new[["Likely_15", "Likely_30", "Likely_45", "MAE", "MSE", "pi_width"]].mean().to_dict()
        )
        all_res[(obs_trained_on, "general")] = (
            nn_base_obs[["Likely_15", "Likely_30", "Likely_45", "MAE", "MSE", "pi_width"]].mean().to_dict()
        )
    #     print("specific:", res_new["MAE"].mean(), ", general:", nn_base_obs["MAE"].mean())
    df_comparison = (
        pd.DataFrame(all_res)
        .swapaxes(1, 0)
        .sort_index()
        .reset_index()
        .rename(columns={"level_0": "obs_count", "level_1": "model"})
    )
    df_comparison["model_name"] = df_comparison["model"].map(
        {"general": "Trained on all observations", "specific": "Trained on this observation"}
    )
    plt.figure(figsize=(10, 5))
    metric = "MAE"
    sns.barplot(data=df_comparison, x="obs_count", y=metric, hue="model_name")
    plt.legend(loc="lower left", framealpha=1)
    plt.xlabel("Observation count")
    plt.ylabel(metric_mapping[metric])
    plt.savefig(os.path.join(fig_path, "general_vs_specific.pdf"))
    # plt.show()


def group_all_by_horizon(model_res_files):
    """Group and aggregate horizon into new dataframes"""
    model_res_files_horizon = []
    for i, res in enumerate(model_res_files):
        if "remaining_runtime" in res.columns:
            res["horizon"] = res["remaining_runtime"] // 5 * 5
            # filter out higher values
            res["horizon"] = res["horizon"].map({c: f"H({int(c)},{int(c+5)}]" for c in res["horizon"].unique()})

        if "train_id" not in res.columns:
            res.rename({"train_ID": "train_id"}, axis=1, inplace=True)

        #     print(res["train_id"].nunique())

        vals_filtered = double_grouping_horizon(res)

        model_res_files_horizon.append(vals_filtered)
    return model_res_files_horizon


def exponential_decay_plot(model_res_files_horizon):
    metric = "Likely_45"
    cols = {
        m: c
        for c, m in zip(
            ["#999999", "#133337", "#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#9F4800", "#F14CC1", "#FFC400"],
            [
                "Current Delay",
                "Historical Average",
                "NN",
                "NGB-N",
                "NGB-LN",
                "MC-2stepE",
                "MC-2stepP",
                "MC-multi",
                "BN",
            ],
        )
    }
    params_df = pd.DataFrame()
    plt.figure(figsize=(8.5, 7))
    for i, vals_in in enumerate(model_res_files_horizon):
        vals = vals_in.copy()
        #     vals["num"] = vals["horizon"].apply(lambda x: int(x.split("(")[1].split(",")[0]))
        vals.sort_values("num", inplace=True)
        #     print(vals["num"], vals[metric])
        vals_x = [0] + list(vals["num"])
        vals_y = [1] + list(vals[metric])
        params, _ = curve_fit(func_simple_powerlaw, list(vals["num"]), vals[metric], bounds=(0, 1))  # maxfev=3000,
        model_name = name[i]
        params_df[model_name] = params

        plt.plot(vals_x, vals_y, label=model_mapping.get(model_name, model_name), c=cols[model_name])
        x = np.arange(0, 100)
        plt.plot(
            x,
            func_simple_powerlaw(x, beta=params[0], offset=params[1], c=params[2]),
            c=cols[model_name],
            linestyle="--",
        )  # beta=0.01, offset=0.25))

        plt.xlim(0, 70)

    custom_lines = [
        Line2D([0], [0], color="black", linestyle="--", lw=2),
    ]
    legend1 = plt.legend(custom_lines, ["Exponential decay fit"], loc="lower left")  # , bbox_to_anchor=(0.4, 0.35))
    plt.gca().add_artist(legend1)

    plt.legend(title="Model", ncol=2, fontsize=17)  # bbox_to_anchor=(1,1)
    plt.xlabel("Prediction horizon [min]")
    plt.xticks(vals["num"], vals["horizon"], rotation=90, fontsize=18, c="dimgrey")
    plt.ylabel("LoR [%] - 90 seconds")  # metric_mapping[metric])
    plt.ylim(0.2, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "exp_decay.pdf"))
    # plt.show()

    return params_df


def exp_decay_parameters_all_metrics(model_res_files_horizon, model_res_files):
    # for the parameter table, we need to find the normalizing factor for MAE and PI Width
    alltogether = pd.concat(model_res_files)
    mae_max = np.quantile(alltogether["MAE"].values, 0.9)
    pi_width_max = np.quantile(alltogether["pi_width"].values, 0.9)

    params_df_all = []

    for metric in ["MAE", "Likely_45", "pi_width"]:
        params_df = pd.DataFrame(columns=["Beta", "Offset (c)", "Factor (a)"])

        for i, vals_in in enumerate(model_res_files_horizon):
            vals = vals_in.copy()
            vals.sort_values("num", inplace=True)

            vals = vals[vals["num"] <= 75]

            # Scale to 0, 1 if MAE or PI width
            if metric == "MAE":
                vals[metric] = vals[metric] / mae_max
            elif metric == "pi_width":
                vals[metric] = vals[metric] / pi_width_max

            # FIT
            params, _ = curve_fit(
                func_simple_powerlaw,
                list(vals["num"]),
                vals[metric],
                bounds=np.array([[0, 0, -1], [1, 1, 1]]),  # maxfev=3000,
            )
            model_name = name[i]
            params_df.loc[model_name] = params
        params_df.columns = pd.MultiIndex.from_arrays([[metric for _ in range(3)], params_df.columns])
        params_df_all.append(params_df)
    params_df_all = pd.concat(params_df_all, axis=1)
    print(params_df_all.to_latex(float_format="%.2f"))


def plot_distance_vs_runtime(nn_res_filtered, metric="Likely_30", direction="down", ymin=0, ymax=1):
    # Compute where the stops are
    data = pd.read_csv("data/data_2023.csv")  # .drop(["Unnamed: 0"], axis=1)
    stops = data[(data["obs_type"] == "stop") & (data["DIR"] == "down")]
    check = stops.groupby(["cat", "obs_short"]).agg({"distanceKM_to_final": "mean", "cat": "count"})
    check = check[check["cat"] > 1000]
    final_stops = check.drop("cat", axis=1).reset_index().drop_duplicates(subset="obs_short")
    final_stops["col"] = final_stops["cat"].map({"fast": "dimgrey", "slow": "lightgrey"})

    # group by distance
    grouped = (
        nn_res_filtered.groupby(["Remaining distance", "cat"])
        .agg({"remaining_runtime": ["mean", "std"], metric: "mean"})
        .reset_index()
    )
    grouped.columns = ["Remaining distance", "Train speed", "Runtime", "Runtime std", "LoR"]

    fast_runtime = grouped.loc[grouped["Train speed"] == "fast"]
    slow_runtime = grouped.loc[grouped["Train speed"] == "slow"]

    mapping = []
    for s in slow_runtime["Runtime"]:
        prev_f = 0
        for i, row in fast_runtime.iterrows():
            f = row["Runtime"]
            if f > s:
                percentage = (s - prev_f) / (f - prev_f)
                interpolated_value = prev_lor + percentage * (row["LoR"] - prev_lor)
                mapping.append([row["Remaining distance"] - 5 * (1 - percentage), interpolated_value])
                break
            prev_f = f
            prev_lor = row["LoR"]

    plt.figure(figsize=(6, 5))

    # plot stops
    for i, row in final_stops.iterrows():
        if direction == "down":
            d = row["distanceKM_to_final"]
        else:
            d = total_dist - row["distanceKM_to_final"]
        plt.plot([d, d], [ymin, ymax], c=row["col"], alpha=0.5)

    custom_lines = [
        Line2D([0], [0], color="black", linestyle="--", lw=2),
    ]
    legend1 = plt.legend(
        custom_lines, ["Mapping of remaining runtime"], loc="lower left", fontsize=17, framealpha=1
    )  # bbox_to_anchor=(0.3, 0.43)
    plt.gca().add_artist(legend1)

    # plot LoR
    sns.lineplot(grouped, hue="Train speed", x="Remaining distance", y="LoR")

    # plot mapping between runtime and distance
    for i in range(len(mapping)):
        plt.plot([i * 5, mapping[i][0]], [slow_runtime.iloc[i]["LoR"], mapping[i][1]], c="black", linestyle="--")

    plt.ylim(ymin, ymax)
    plt.ylabel("LoR [%] - 60 seconds")
    plt.tight_layout()
    plt.savefig(f"figures/distance_time_mapping_{direction}.pdf")
    # plt.show()


def compare_up_down_fast_slow_exp_decay(data, res_orig, mae_max, pi_width_max):
    res_orig = res_orig.merge(
        data[["train_id", "obs_count", "dep_real"]],
        left_on=["train_id", "obs_count"],
        right_on=["train_id", "obs_count"],
        how="left",
    )
    res_orig["dep_real"] = pd.to_datetime(res_orig["dep_real"])
    res_orig["horizon"] = res_orig["remaining_runtime"] // 5 * 5
    # filter out higher values
    res_orig["horizon"] = res_orig["horizon"].map({c: f"H({int(c)},{int(c+5)}]" for c in res_orig["horizon"].unique()})
    res_orig["daytime"] = (res_orig["dep_real"].dt.hour < 12).map({True: "Morning", False: "Afternoon"})

    params_df_all = []
    for metric in ["MAE", "Likely_45", "pi_width"]:
        params_df = pd.DataFrame(columns=["Beta", "Offset (c)", "Factor (a)"])

        for filter_col, filter_val in [
            ("DIR", "up"),
            ("DIR", "down"),
            ("cat", "fast"),
            ("cat", "slow"),
            ("daytime", "Morning"),
            ("daytime", "Afternoon"),
        ]:
            res = res_orig[res_orig[filter_col] == filter_val]
            print(len(res))

            vals = double_grouping_horizon(res)
            vals.sort_values("num", inplace=True)

            # Scale to 0, 1 if MAE or PI width
            if metric == "MAE":
                vals[metric] = (
                    vals[metric] / mae_max
                )  # vals[metric].max() # ((vals[metric] * -1) + vals[metric].max()) / vals[metric].max()
            elif metric == "pi_width":
                vals[metric] = vals[metric] / pi_width_max

            vals_x = [0] + list(vals["num"])
            vals_y = [1] + list(vals[metric])
            # FIT
            params, _ = curve_fit(
                func_simple_powerlaw, vals_x, vals_y, bounds=np.array([[0, 0, -1], [1, 1, 1]])  # maxfev=3000,
            )
            params_df.loc[filter_val] = params
        params_df.columns = pd.MultiIndex.from_arrays([[metric for _ in range(3)], params_df.columns])
        params_df_all.append(params_df)

    params_df_all = pd.concat(params_df_all, axis=1)
    # print(params_df_all.to_latex(float_format="%.2f"))
    params_df_all["index"] = ["Direction", "Direction", "Speed", "Speed", "Time", "Time"]
    params_df_all = params_df_all.reset_index().set_index(["index", "level_0"])

    new_version = params_df_all["Likely_45"].drop("Factor (a)", axis=1).T
    print(new_version.to_latex(float_format="%.2f"))


def regression_analysis(res_orig):
    import statsmodels.api as sm

    # map variables
    res_transformed = res_orig.copy()

    res_transformed["DIR"] = res_transformed["DIR"].map({"down": 0, "up": 1})
    res_transformed["cat"] = res_transformed["cat"].map({"slow": 0, "fast": 1})
    # res_transformed["stops"] = res_transformed["delay_dep"] / 60 # in min
    res_transformed["delay_dep"] = res_transformed["delay_dep"] / 60  # in min
    res_transformed["final_delay"] = res_transformed["final_delay"] / 60  # in min
    res_transformed["remaining_runtime"] = res_transformed[
        "remaining_runtime"
    ]  # / res_transformed["remaining_runtime"].max()
    res_transformed["distanceKM_to_final"] = res_transformed[
        "distanceKM_to_final"
    ]  # / res_transformed["distanceKM_to_final"].max()
    res_transformed["daytime"] = res_transformed["daytime"].map({"Morning": 0, "Afternoon": 1})

    # define inputs and outputs
    inp = res_transformed[
        ["DIR", "daytime", "cat", "distanceKM_to_final", "final_delay"]
    ].copy()  # , "obs_count", "stops",
    outp = res_transformed["Likely_45"]
    inp = sm.add_constant(inp, prepend=False)
    mod = sm.OLS(outp, inp)
    res = mod.fit()
    print(res.summary())

    # print table
    results_summary = res.summary2().tables[1]
    results_summary.index.name = "Variable"
    results_summary.reset_index(inplace=True)
    results_summary["Variable"] = results_summary["Variable"].map(col_mapping)
    # pd.Series(results_summary["Variable"])
    print(results_summary.to_latex(float_format="%.3f", index=False))


if __name__ == "__main__":
    fig_path = "figures"
    in_path = "outputs"

    # LOAD ALL RESULTS
    path = "2023_2_v4 - final"
    nn_out = pd.read_csv(f"outputs/{path}/nn_res.csv")
    ngb_out = pd.read_csv(f"outputs/{path}/ngb_res.csv")
    simple_median_out = pd.read_csv(f"outputs/{path}/simple_median_res.csv")
    # random_forest_out = pd.read_csv(f"outputs/{path}/random_forest_res.csv")
    ngb_lognormal = pd.read_csv(f"outputs/{path}/ngb_lognormal_res.csv")
    mc_2stepE = mc_to_myformat(pd.read_csv(f"outputs/{path}/mc-2stepE_res.csv", delimiter=";", decimal=","))
    mc_2stepP = mc_to_myformat(pd.read_csv(f"outputs/{path}/mc-2stepP_res.csv", delimiter=";", decimal=","))
    mc_multi = mc_to_myformat(pd.read_csv(f"outputs/{path}/mc-multi_res.csv", delimiter=";", decimal=","))
    bn_out = pd.read_csv(f"outputs/{path}/bn_res.csv", delimiter=";", decimal=",").rename(
        {"likely_15": "Likely_15", "likely_30": "Likely_30", "likely_45": "Likely_45", "mae": "MAE"}, axis=1
    )
    # current_delay_out = pd.read_csv(f"outputs/{path}/simple_current_delay_res.csv")

    # List of all results
    model_res_files = [ngb_out, ngb_lognormal, nn_out, mc_2stepE, mc_2stepP, mc_multi, bn_out]
    name = ["NGB-N", "NGB-LN", "NN", "MC-2stepE", "MC-2stepP", "MC-multi", "BN"]

    # group all by horizon
    model_res_files_horizon = group_all_by_horizon(model_res_files)

    # make exponential decay plot
    params_df = exponential_decay_plot(model_res_files_horizon)

    # table with parameters for the LoR exponential decay
    params_df.index = ["Beta", "Offset (c)", "Factor (a)"]
    params_df = params_df.swapaxes(1, 0)
    params_df = params_df.drop("Factor (a)", axis=1).T
    print(params_df.to_latex(float_format="%.2f"))

    # plot fast vs slow
    nn_res = model_res_files[2]
    nn_res["Remaining distance"] = nn_res["distanceKM_to_final"] // 5 * 5
    # distance-vs-time plot --> up
    nn_res_filtered = nn_res[nn_res["DIR"] == "up"]
    metric = "Likely_30"
    plot_distance_vs_runtime(nn_res_filtered, direction="up")
    # distance-vs-time plot --> down
    nn_res_filtered = nn_res[nn_res["DIR"] == "down"]
    plot_distance_vs_runtime(nn_res_filtered, direction="down")

    # regression analysis
    regression_analysis(nn_res)
