import os
from xml.sax.handler import feature_external_ges

from config import *
from train_delay.features import Features

if __name__ == "__main__":

    featurizer = Features(os.path.join("data", "data_2023.csv"))
    order = 3
    print(len(featurizer.data))

    # remove outliers
    featurizer.scale_final_delay(outlier_cutoff=OUTLIER_CUTOFF)
    print(f"len after removing nan and >{OUTLIER_CUTOFF}min final delay", len(featurizer.data))

    # add weather
    # featurizer.add_weather(weather_path=os.path.join("data", "weather_data.csv"))

    # obs count feature
    featurizer.transform_obs_count()

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

    # save
    featurizer.save(out_path=os.path.join("data", "data_enriched.csv"))
