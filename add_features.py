import os
import argparse

from config import *
from train_delay.features import Features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inp_path", type=str, default=os.path.join("data", "data_2024.csv"))
    parser.add_argument("-o", "--order", default=5, type=int, help="how many steps do we look back?")
    args = parser.parse_args()

    featurizer = Features(args.inp_path)
    data_len_start = len(featurizer.data)

    order = args.order
    print(len(featurizer.data))

    # process time variables:
    featurizer.add_timevars()

    # remove outliers
    featurizer.scale_final_delay(outlier_cutoff=OUTLIER_CUTOFF)
    print(f"len after removing nan and >{OUTLIER_CUTOFF}min final delay", len(featurizer.data))

    # add weather
    # weather generated for first three months but not for afterwards
    featurizer.add_weather(weather_path=os.path.join("data", "weather_data"))

    # buffer and supplement
    featurizer.add_basic_features()

    # obs count feature
    featurizer.transform_obs_count()
    featurizer.num_stops_feature()

    # add previous delay features
    featurizer.delay_at_preceding_obs(order=order)
    featurizer.historic_delay_at_obs(order=order)
    featurizer.avg_historic_final_delay(order=order)
    featurizer.general_delay_on_day()

    # add delay of surrounding trains
    ## TODO: for this one, we would need to distinguish up and down direction!
    # featurizer.delays_other_trains(order=5, minute_thresh=10)

    # # add train ID as one hot:
    # featurizer.train_id_onehot()

    # add time features
    featurizer.time_features("dep_real")
    featurizer.time_features("trip_final_arr_plan")

    assert len(featurizer.data) == data_len_start, "Number of samples changed!"
    # save
    featurizer.save(out_path=os.path.join("data", "data_enriched.csv"))
