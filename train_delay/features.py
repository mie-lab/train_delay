from xml.sax.handler import feature_external_ges
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import geopandas as gpd
from datetime import timedelta
from scipy.stats import pearsonr

from meteostat import Daily
from datetime import timedelta
from meteostat import Point as MeteoPoint

WEATHER_OUT_PATH = "data/weather_data_2.csv"


class Features:
    def __init__(self, path) -> None:
        self.data = pd.read_csv(path).drop(["Unnamed: 0"], axis=1, errors="ignore")

        # correct NaNs
        # fill in nans in delay_dep und dep_real at last observation
        nans = pd.isna(self.data["delay_dep"])
        self.data.loc[nans, "delay_dep"] = self.data.loc[nans, "final_delay"]
        self.data.loc[self.data["dep_real"].isna(), "dep_real"] = self.data.loc[
            self.data["dep_real"].isna(), "arr_real"
        ]
        self.data.loc[self.data["dep_plan"].isna(), "dep_plan"] = self.data.loc[
            self.data["dep_plan"].isna(), "arr_plan"
        ]
        # replace remaining Nans in dep_real with the row before
        self.data.sort_values(["train_id", "obs_count"], inplace=True)
        self.data["last_dep_time"] = self.data["dep_real"].shift(1)
        self.data["last_train_id"] = self.data["train_id"].shift(1)
        fill_cond = (pd.isna(self.data["dep_real"])) & (self.data["train_id"] == self.data["train_id"])
        self.data.loc[fill_cond, "dep_real"] = self.data.loc[fill_cond, "last_dep_time"]
        self.data.drop(["last_dep_time", "last_train_id"], axis=1, inplace=True)
        print("still NaN:", self.data["dep_real"].isna().sum())

    def add_timevars(self) -> None:
        """convert to datetimes and compute features on times"""

        # to datetime
        for timevar in [
            "arr_plan",
            "arr_real",
            "dep_plan",
            "dep_real",
            "trip_first_dep_plan",
            "trip_final_arr_plan",
            "trip_final_arr_real",
        ]:
            self.data[timevar] = pd.to_datetime(self.data[timevar])

        # create difference
        self.data["feat_time_to_end_real"] = (
            self.data["trip_final_arr_plan"] - self.data["dep_real"]
        ).dt.total_seconds()
        self.data["feat_time_to_end_plan"] = (
            self.data["trip_final_arr_plan"] - self.data["dep_plan"]
        ).dt.total_seconds()

    def scale_final_delay(self, outlier_cutoff=5):
        """If final delay is given in seconds, convert it to minutes. Then, remove all above the cutoff"""
        if any(self.data["final_delay"] > 60 * 5):
            # scale the data if they are provided in seconds
            self.data["final_delay"] = self.data["final_delay"] / 60

    def add_basic_features(self):
        # fill nans for feature
        self.fill_with_mean_per_train("supplement_time")
        self.fill_with_mean_per_train("buffer_time")
        self.data["feat_log_buffer_time"] = np.log(self.data["feat_buffer_time"])
        self.data.drop("feat_buffer_time", inplace=True, axis=1)
        # scale delay
        self.data["feat_delay_dep"] = self.data["delay_dep"] / 60
        self.data["distanceKM_to_final"] = self.data["distanceKM_to_final"]
        self.data["feat_fast_slow"] = self.data["cat"].map({"slow": 0, "fast": 1})
        self.data["feat_direction"] = self.data["DIR"].map({"up": 0, "down": 1})

    def fill_with_mean_per_train(self, variable):
        # get mean per train ID
        mean_per_train = pd.DataFrame(self.data.groupby("train_id")[variable].mean()).rename(
            columns={variable: "mean_val"}
        )
        # fill NaNs with the overall mean
        mean_per_train.fillna(mean_per_train.mean(), inplace=True)
        # add to data
        self.data = self.data.merge(mean_per_train, how="left", left_on="train_id", right_index=True)
        # transform to feature
        self.data["feat_" + variable] = self.data[variable]
        # fill with mean
        self.data.loc[pd.isna(self.data["feat_" + variable]), "feat_" + variable] = self.data.loc[
            pd.isna(self.data["feat_" + variable]), "mean_val"
        ]
        self.data.drop("mean_val", axis=1, inplace=True)

    def transform_obs_count(self):
        # get maximum observation per train ID
        max_obs_per_train = pd.DataFrame(self.data.groupby("train_id")["obs_count"].max()).rename(
            columns={"obs_count": "feat_obs_count"}
        )
        # merge with data
        self.data = self.data.merge(max_obs_per_train, how="left", left_on="train_id", right_index=True)
        # feat obs is divided by the maximum
        self.data["feat_obs_count"] = self.data["obs_count"] / self.data["feat_obs_count"]

    def add_weather(self, weather_path="data/weather_data"):
        weather_data = pd.read_csv(weather_path + ".csv")
        weather_data["time"] = pd.to_datetime(weather_data["time"])
        weather_data.set_index(["station", "time"], inplace=True)
        with open(weather_path + ".json", "r") as infile:
            obs_point_station_mapping = json.load(infile)

        # map closest station
        self.data["closest_station"] = self.data["obs_point_id"].astype(str).map(obs_point_station_mapping).astype(int)

        # get columns that have no NaNs (the others have basically just Nans)
        nr_nans = pd.isna(weather_data).sum()
        cols_no_nans = list(nr_nans[nr_nans < 1].index)
        useful_weather_data = weather_data[cols_no_nans]
        useful_weather_data.columns = "feat_weather_" + useful_weather_data.columns
        # round departure to hour
        self.data["dep_real_hour"] = self.data["dep_real"].dt.round("H")
        # merge
        self.data = self.data.merge(
            useful_weather_data, left_on=["closest_station", "dep_real_hour"], right_index=True, how="left"
        ).drop(["closest_station", "dep_real_hour"], axis=1)

    def num_stops_feature(self):
        self.data["feat_stops"] = self.data["stops"] / self.data["stops"].max()

    def time_features(self, col_name):
        # add basic features
        self.data[f"feat_{col_name}_hour"] = self.data[col_name].apply(lambda x: x.hour if not pd.isna(x) else pd.NA)
        self.data[f"feat_{col_name}_day"] = self.data[col_name].apply(lambda x: x.dayofweek if ~pd.isna(x) else pd.NA)

        for time_range, period in zip(["hour", "day"], [24, 7]):
            to_sin = lambda x: np.sin(x / period * 2 * np.pi) if not pd.isna(x) else pd.NA
            to_cos = lambda x: np.cos(x / period * 2 * np.pi) if not pd.isna(x) else pd.NA
            self.data[f"feat_{col_name}_{time_range}_sin"] = self.data[f"feat_{col_name}_{time_range}"].apply(to_sin)
            self.data[f"feat_{col_name}_{time_range}_cos"] = self.data[f"feat_{col_name}_{time_range}"].apply(to_cos)

    def delay_at_preceding_obs(self, order=3):
        """Delay at the previous <order> observations (of the same train on the same day)"""
        self.data.sort_values(["day", "train_id", "obs_count"], inplace=True)
        for i in range(1, order + 1):
            self.data[f"prev_train_id_{i}"] = self.data["train_id"].shift(i)
            self.data[f"feat_delay_obs-{i}"] = self.data["delay_dep"].shift(i)
            self.data.loc[(self.data[f"prev_train_id_{i}"] != self.data["train_id"]), f"feat_delay_obs-{i}"] = 0
        self.data.drop([f"prev_train_id_{i}" for i in range(1, order + 1)], axis=1, inplace=True)

    def historic_delay_at_obs(self, order=3):
        """Delay at this observation in the last <order> days"""
        # sort self.data
        self.data.sort_values(["train_id_daily", "obs_count", "day"], inplace=True)
        for i in range(1, order + 1):
            # get delay on previous day at this observation
            self.data[f"feat_delay_day-{i}"] = self.data["delay_dep"].shift(i)
            # remove wrong ones (where it was shifted across train IDs or across observations)
            different_obs = self.data["obs_count"] != self.data["obs_count"].shift(i)
            different_id = self.data["train_id_daily"] != self.data["train_id_daily"].shift(i)
            self.data.loc[(different_obs | different_id), f"feat_delay_day-{i}"] = 0

    def avg_historic_final_delay(self, order=3):
        """Final delay of this train in the last <order> days, and average delay of this train in the past"""
        # sort by day
        self.data.sort_values("day", inplace=True)
        # get final delay per train per day
        delay_per_day_per_train = self.data.groupby(["train_id_daily", "day"]).agg({"final_delay": "mean"})
        delay_per_day_per_train["counter"] = 1  # helper variable
        # make cumsum to get rolling average (must be shifted by 1 because we want the historic without the current)
        delay_per_day_per_train["delay_cumsum"] = (
            delay_per_day_per_train.reset_index()
            .groupby("train_id_daily")["final_delay"]
            .cumsum()
            .shift(1)
            .fillna(0)
            .values
        )
        delay_per_day_per_train["nr_samples_per_day"] = (
            delay_per_day_per_train.reset_index().groupby("train_id_daily")["counter"].cumsum().values - 1
        )
        delay_per_day_per_train["feat_avg_prev_delay"] = (
            delay_per_day_per_train["delay_cumsum"] / delay_per_day_per_train["nr_samples_per_day"]
        )
        # fill nans
        delay_per_day_per_train.loc[delay_per_day_per_train["nr_samples_per_day"] == 0, "feat_avg_prev_delay"] = 0

        # Secondly, get the previous final delays of the past x days
        data_for_day_shift = delay_per_day_per_train.reset_index()
        for i in range(1, order + 1):
            data_for_day_shift[f"feat_final_delay-day-{i}"] = data_for_day_shift["final_delay"].shift(i)
            different_id = data_for_day_shift["train_id_daily"] != data_for_day_shift["train_id_daily"].shift(i)
            data_for_day_shift.loc[different_id, f"feat_final_delay-day-{i}"] = 0

        # merge into original dataframe by day * train id
        relevant_columns = ["train_id_daily", "day", "feat_avg_prev_delay"] + [
            f"feat_final_delay-day-{i}" for i in range(1, order + 1)
        ]
        self.data = pd.merge(
            self.data,
            data_for_day_shift[relevant_columns],
            how="left",
            left_on=["train_id_daily", "day"],
            right_on=["train_id_daily", "day"],
        )

    def general_delay_on_day(self):
        """Final delay of all trains on this day (so far)"""
        # average delay of all trains on this day so far
        data_temp = self.data.copy()
        # isnan
        data_temp.loc[data_temp.groupby("train_id")["obs_count"].transform("idxmax"), "is_last"] = True
        data_temp["is_last"].fillna(False, inplace=True)
        assert data_temp["dep_real"].isna().sum() == 0, "still contains NaNs"
        # sort by departure (basically observation time point)
        data_temp.sort_values("dep_real", inplace=True)
        # group by day
        for _, day_grouped in data_temp.groupby("day"):
            avg_sofar_thisday = []
            running_avg_delay = 0
            counter = 0
            for _, row in day_grouped.iterrows():
                # update rolling average
                avg_sofar_thisday.append(running_avg_delay)
                if row["is_last"]:
                    running_avg_delay = ((running_avg_delay * counter) + row["final_delay"]) / (counter + 1)
                    counter += 1
            self.data.loc[day_grouped.index, "feat_delay_on_day"] = avg_sofar_thisday

    def delays_other_trains(self, order=5, minute_thresh=10):
        """
        Get delays of surrounding trains

        NOTES:
        - We consider all observations in the 10 minutes (minute_thresh) before the target observation
        - The train ID must be different --> we are interested in the delay of other trains
        - We restrict it to observations that are further ahead on the track --> if we are at obs point 100, we only
        consider observations at point 101+
        - We sort all remaining delays by their closeness in terms of observation points, and keep the <order> that
        are closest. Underlying assumption: The train that is preceding our train, i.e. just slightly ahead in terms of
        observation points, has most influence on the current delay
        - we only retrieve the current delay of these surrounding trains, not their position. We could do this by adding
        the obs-count-diff as another feature
        """
        temp_data = self.data.copy()
        # sort by departure real and transform into datetime
        temp_data.sort_values("dep_real", inplace=True)
        temp_data.dropna(subset=["dep_real"], inplace=True)
        temp_data["dep_real"] = pd.to_datetime(temp_data["dep_real"])

        # init counter
        shift_counter = 1
        any_less_than_threshold = True

        while any_less_than_threshold:
            # shift delays, obs count, train id and dep_real
            temp_data[f"prev_obs_count-{shift_counter}"] = temp_data["obs_count"].shift(shift_counter)
            prev_train_id = temp_data["train_id"].shift(shift_counter)  # temp_data[f"next_train_id-{shift_counter}"]
            prev_dep_real = temp_data["dep_real"].shift(shift_counter)  # temp_data[f"next_dep_real-{shift_counter}"]
            temp_data[f"prev_delay_dep-{shift_counter}"] = temp_data["delay_dep"].shift(shift_counter)
            # Remove the ones where the next train ID is the same
            temp_data.loc[prev_train_id == temp_data["train_id"], f"prev_delay_dep-{shift_counter}"] = pd.NA
            # Remove the ones with a smaller obs count
            temp_data.loc[
                temp_data[f"prev_obs_count-{shift_counter}"] < temp_data["obs_count"], f"prev_delay_dep-{shift_counter}"
            ] = pd.NA
            # compute how many are inside the 10min frame
            inside_10_min = prev_dep_real > temp_data["dep_real"] - timedelta(minutes=minute_thresh)
            # print(shift_counter, sum(inside_10_min))
            any_less_than_threshold = sum(inside_10_min) > 0
            shift_counter += 1

        # columns that we need for further steps
        delay_dep_cols = [col for col in temp_data.columns if col.startswith("prev_delay_dep")]
        obs_count_cols = [col for col in temp_data.columns if col.startswith("prev_obs_count")]

        # aggregate the closest columns
        def get_new_cols(row):
            """Function to aggreagte the delays of nearby trains"""
            obs_idx = row[obs_count_cols].argsort()
            closest_delays_placeholder = np.zeros(order)
            closest_delays = row[delay_dep_cols].iloc[obs_idx].dropna()
            # TODO: do we want to remove train ids that appear multiple times in the closest delay?
            closest_delays_placeholder[: len(closest_delays)] = closest_delays[:order]
            dict_with_closest = {"feat_delay_closest_" + str(i): closest_delays_placeholder[i] for i in range(order)}
            return pd.Series(dict_with_closest)

        print("Starting to aggregate closest delays...")
        closest_delay_df = []
        for _, row in temp_data.iterrows():
            closest_delay_df.append(get_new_cols(row))
        closest_delay_df = pd.DataFrame(closest_delay_df, index=temp_data.index)
        print("Finished.")

        # merge
        self.data = self.data.merge(closest_delay_df, how="left", left_index=True, right_index=True)

    def train_id_onehot(self):
        one_hot_train_id = pd.get_dummies(self.data["train_id_daily"], prefix="feat_train_id")
        self.data = self.data.merge(one_hot_train_id, left_index=True, right_index=True)

    def time_since_stop_feature(self):
        # new feature: time_since last stop
        self.data["time_since_stop"] = pd.NA
        self.data.loc[self.data["obs_type"] == "stop", "time_since_stop"] = 0

        counter = 0
        while sum(self.data["time_since_stop"].isna()) > 0:
            # shift these two
            self.data["prev_time_since_stop"] = self.data["time_since_stop"].shift(1)
            self.data["prev_remaining_runtime"] = self.data["remaining_runtime"].shift(1)
            self.data["prev_train_id"] = self.data["train_id"].shift(1)
            cond_fill = (
                pd.isna(self.data["time_since_stop"])
                & (self.data["prev_train_id"] == self.data["train_id"])
                & ~pd.isna(self.data["prev_time_since_stop"])
            )
            self.data["fill_values"] = (
                self.data["prev_remaining_runtime"] - self.data["remaining_runtime"]
            ) + self.data["prev_time_since_stop"]
            self.data.loc[cond_fill, "time_since_stop"] = self.data.loc[cond_fill, "fill_values"]
            # print(counter, sum(self.data["time_since_stop"].isna()))
            counter += 1

        self.data.drop(
            ["prev_remaining_runtime", "prev_time_since_stop", "prev_train_id", "fill_values"], axis=1, inplace=True
        )
        self.data.rename(columns={"time_since_stop": "feat_time_since_stop"}, inplace=True)

    def save(self, out_path=os.path.join("data", "data_enriched.csv")):
        # clean up columns
        self.data.drop(
            [
                "Station_1",
                "Station_last",
                "lng_final",
                "lat_final",
                "obs_short",
                "lng",
                "lat",
                "arr_real",
                "trip_first_dep_plan",
                "trip_final_arr_plan",
                "trip_final_arr_real",
                "trip_first_dep_plan_h",
                "trip_first_dep_plan_m",
                "trip_final_arr_plan_h",
                "trip_final_arr_plan_m",
                "arr_plan",
                "dep_plan",
                "dep_real",
            ],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        # save
        self.data.to_csv(out_path, index=False)
