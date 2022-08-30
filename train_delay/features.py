from xml.sax.handler import feature_external_ges
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from datetime import timedelta
from scipy.stats import pearsonr

from meteostat import Daily
from datetime import timedelta
from meteostat import Point as MeteoPoint


class Features:
    def __init__(self, path) -> None:
        self.data = pd.read_csv(path).drop(["Unnamed: 0"], axis=1, errors="ignore")
        # fill in delay_dep at last observation
        nans = pd.isna(self.data["delay_dep"])
        self.data.loc[nans, "delay_dep"] = self.data.loc[nans, "final_delay"]
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

    def remove_outliers(self, outlier_cutoff=5):
        # remove outliers
        print("Number of samples initial", len(self.data))
        self.data = self.data[
            (self.data["final_delay"] > -1 * outlier_cutoff) & (self.data["final_delay"] < outlier_cutoff)
        ]
        print("Number of samples after outlier removal", len(self.data))

    def transform_obs_count(self):
        # get maximum observation per train ID
        max_obs_per_train = pd.DataFrame(self.data.groupby("train_id")["obs_count"].max()).rename(
            columns={"obs_count": "feat_obs_count"}
        )
        # merge with data
        self.data = self.data.merge(max_obs_per_train, how="left", left_on="train_id", right_index=True)
        # feat obs is divided by the maximum
        self.data["feat_obs_count"] = self.data["obs_count"] / self.data["feat_obs_count"]
        # set the ones of the final observation to NaN! Then it is dropped later
        self.data.loc[self.data["feat_obs_count"] == 1, "feat_obs_count"] = pd.NA

    def add_weather(self, weather_path=None):
        def get_daily_weather(row):
            lng, lat = (row["lng"], row["lat"])
            loc = MeteoPoint(lat, lng)
            end = row["dep_real"].replace(tzinfo=None)
            self.data = Daily(loc, end - timedelta(days=1, minutes=1), end).fetch()
            if len(self.data) == 1:
                return pd.Series(self.data.iloc[0])
            else:
                for random_testing in [100, 250, 500]:
                    loc = MeteoPoint(lat, lng, random_testing)
                    self.data = Daily(loc, end - timedelta(days=1), end).fetch()
                    if len(self.data) == 1:
                        return pd.Series(self.data.iloc[0])
                return pd.NA

        if weather_path is not None:
            weather_data = pd.read_csv(weather_path).set_index(["day", "obs_point_id"])
            # Check how many nans there are per weather feature and remove the ones with too many
            weather_feats = weather_data.columns
            nr_nans = pd.isna(weather_data[weather_feats]).sum()
            cols_too_many_nans = list(nr_nans[nr_nans > 0.02 * len(weather_data)].index)
            weather_feats = [f for f in weather_feats if f not in cols_too_many_nans]
            # merge with data
            self.data = self.data.merge(
                weather_data[weather_feats], how="left", left_on=["day", "obs_point_id"], right_index=True
            )
        else:
            weather_input = self.data[["lng", "lat", "dep_real"]].dropna()
            weather_data = weather_input.apply(get_daily_weather, axis=1)
            weather_data["prcp"] = weather_data["prcp"].fillna(0)
            weather_data.rename(columns={c: "feat_weather_" + c for c in weather_data.columns}, inplace=True)
            self.data = self.data.merge(weather_data, how="left", left_index=True, right_index=True)

    def time_features(self, col_name):
        # add basic features
        self.data[f"feat_{col_name}_hour"] = self.data[col_name].apply(lambda x: x.hour if not pd.isna(x) else pd.NA)
        self.data[f"feat_{col_name}_day"] = self.data[col_name].apply(lambda x: x.dayofweek if ~pd.isna(x) else pd.NA)

        for time_range, period in zip(["hour", "day"], [24, 7]):
            to_sin = lambda x: np.sin(x / period * 2 * np.pi if not pd.isna(x) else pd.NA)
            to_cos = lambda x: np.cos(x / period * 2 * np.pi if not pd.isna(x) else pd.NA)
            self.data[f"feat_{col_name}_{time_range}_sin"] = self.data[f"feat_{col_name}_hour"].apply(to_sin)
            self.data[f"feat_{col_name}_{time_range}_cos"] = self.data[f"feat_{col_name}_hour"].apply(to_cos)

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
        # Problem: NaNs in dep real --> we need to sort by dep real
        # first fill with arr_real
        data_temp.loc[pd.isna(self.data["dep_real"]), "dep_real"] = self.data.loc[
            pd.isna(self.data["dep_real"]), "arr_real"
        ]
        # print(any(pd.isna(data_temp["dep_real"])))
        # secondly fill the leftover NaNs with dep_plan
        data_temp.loc[pd.isna(data_temp["dep_real"]), "dep_real"] = data_temp.loc[
            pd.isna(self.data["dep_real"]), "dep_plan"
        ]

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

    def save(self, out_path=os.path.join("data", "data_enriched.csv")):
        # save
        self.data.to_csv(out_path, index=False)
