import os

from config import *
from train_delay.features import Features

if __name__ == "__main__":

    featurizer = Features(os.path.join("data", "test_data.csv"))
    order = 3

    # remove outliers
    featurizer.remove_outliers(outlier_cutoff=OUTLIER_CUTOFF)

    # add previous delay features
    featurizer.delay_at_preceding_obs(order=order)
    featurizer.historic_delay_at_obs(order=order)
    featurizer.avg_historic_final_delay(order=order)
    featurizer.general_delay_on_day()

    # # add weather
    # featurizer.add_weather(self.data)

    # # add train ID as one hot:
    # featurizer.train_id_onehot()

    # add time features
    featurizer.time_features("dep_real")
    featurizer.time_features("trip_final_arr_plan")

    # save
    featurizer.save()
