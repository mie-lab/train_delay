import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy.stats import pearsonr


from meteostat import Daily
from datetime import timedelta
from meteostat import Point as MeteoPoint


def add_weather(data):
    def get_daily_weather(row):
        lng, lat = (row["lng"], row["lat"])
        loc = MeteoPoint(lat, lng)
        end = row["dep_real"].replace(tzinfo=None)
        data = Daily(loc, end - timedelta(days=1, minutes=1), end).fetch()
        if len(data) == 1:
            return pd.Series(data.iloc[0])
        else:
            for random_testing in [100, 250, 500]:
                loc = MeteoPoint(lat, lng, random_testing)
                data = Daily(loc, end - timedelta(days=1), end).fetch()
                if len(data) == 1:
                    return pd.Series(data.iloc[0])
            return pd.NA

    weather_input = data[["lng", "lat", "dep_real"]].dropna()
    weather_data = weather_input.apply(get_daily_weather, axis=1)

    weather_data["prcp"] = weather_data["prcp"].fillna(0)
    weather_data.rename(columns={c: "feat_weather_" + c for c in weather_data.columns}, inplace=True)
    data = data.merge(weather_data, how="left", left_index=True, right_index=True)
    return data


def time_features(data, col_name):
    # add basic features
    data[f"feat_{col_name}_hour"] = data[col_name].apply(lambda x: x.hour if not pd.isna(x) else pd.NA)
    data[f"feat_{col_name}_day"] = data[col_name].apply(lambda x: x.dayofweek if ~pd.isna(x) else pd.NA)

    for time_range, period in zip(["hour", "day"], [24, 7]):
        to_sin = lambda x: np.sin(x / period * 2 * np.pi if not pd.isna(x) else pd.NA)
        to_cos = lambda x: np.cos(x / period * 2 * np.pi if not pd.isna(x) else pd.NA)
        data[f"feat_{col_name}_{time_range}_sin"] = data[f"feat_{col_name}_hour"].apply(to_sin)
        data[f"feat_{col_name}_{time_range}_cos"] = data[f"feat_{col_name}_hour"].apply(to_cos)
    return data

def previous_delay_features(data, order = 3):
    for i in range(1, order+1):
        data[f"prev_train_id_{i}"] = data["train_id"].shift(i)
        data[f"delay-{i}"] = data["delay_dep"].shift(i)
        data.loc[(data[f"prev_train_id_{i}"] != data["train_id"]), f"delay-{i}"] = 0
    data.drop([f"prev_train_id_{i}" for i in range(1, order+1)], axis=1, inplace=True)
    return data
    
if __name__ == "__main__":

    data = pd.read_csv(os.path.join("data","test_data.csv")).drop(["Unnamed: 0"], axis=1)

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
        data[timevar] = pd.to_datetime(data[timevar])
    # create difference
    data["time_to_end_real"] = (data["trip_final_arr_plan"] - data["dep_real"]).dt.seconds
    data["time_to_end_plan"] = (data["trip_final_arr_plan"] - data["dep_plan"]).dt.seconds

    # add weather
    data = add_weather(data)

    # add train ID as one hot:
    one_hot_train_id = pd.get_dummies(data["train_id_daily"], prefix="train_id")
    data = data.merge(one_hot_train_id, left_index=True, right_index=True)

    # add time features
    data = time_features(data, "dep_real")
    data = time_features(data, "trip_final_arr_plan")

    # save
    data.to_csv(os.path.join("data", "data_enriched.csv"), index=False)
