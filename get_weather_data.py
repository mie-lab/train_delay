import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import json
import time
import geopandas as gpd
import os

from meteostat import Daily, Hourly
from datetime import timedelta
from meteostat import Point as MeteoPoint
from meteostat import Stations


def get_station_ids():
    # fetch all stations in switzerland
    stations = Stations()
    stations = stations.region("CH")
    stations = stations.fetch()

    # get subset of the data that gives the lat and long
    subset_lat_lng = data_2024[["obs_point_id", "lat", "lng"]].drop_duplicates("obs_point_id")

    # convert both to gdf
    geo_df_data = gpd.GeoDataFrame(
        subset_lat_lng, crs="4326", geometry=gpd.points_from_xy(x=subset_lat_lng["lng"], y=subset_lat_lng["lat"])
    )
    geo_df_data.to_crs(2056, inplace=True)
    geo_df_stations = gpd.GeoDataFrame(
        stations, crs="4326", geometry=gpd.points_from_xy(x=stations["longitude"], y=stations["latitude"])
    )
    geo_df_stations.to_crs(2056, inplace=True)
    closest_stations = geo_df_data.sjoin_nearest(geo_df_stations, how="left")

    # map from obs point id to closest station
    obs_point_station_mapping = closest_stations.set_index("obs_point_id")["index_right"].to_dict()
    considered_stations = closest_stations["index_right"].unique()

    return considered_stations, obs_point_station_mapping


if __name__ == "__main__":
    IN_PATH = "data/data_2024.csv"
    OUT_PATH_CSV = "data/weather_data.csv"
    OUT_PATH_MAPPING = "data/weather_data.json"

    # read data
    data_2024 = pd.read_csv(IN_PATH)

    # get time bounds
    departure_times = pd.to_datetime(data_2024["dep_real"])
    time_min, time_max = (
        (departure_times.min() - timedelta(hours=5)).to_pydatetime(),
        (departure_times.max() + timedelta(hours=5)).to_pydatetime(),
    )

    considered_stations, obs_point_station_mapping = get_station_ids()

    # Get hourly data
    all_weather = []
    for station_id in considered_stations:
        weather = Hourly(station_id, time_min, time_max)
        weather = weather.fetch()
        weather["station"] = station_id
        print(weather.head())
        all_weather.append(weather)
    all_weather = pd.concat(all_weather)
    all_weather.to_csv(OUT_PATH_CSV)

    with open(os.path.join(OUT_PATH_MAPPING), "w") as outfile:
        json.dump(obs_point_station_mapping, outfile)
