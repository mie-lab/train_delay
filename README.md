# Uncertainty-aware train delay prediction

This code accompanies our paper on train delay prediction under uncertainty and the analysis of predictability.

Unfortunately, the dataset is not publicly available. With similar data, this code can be used to train uncertainty-aware neural networks and to evaluate the predictions. The data should contain observations of the current and final delay, in form of a csv file with the main columns `train_id`, `obs_count`, `final_delay`, `current_delay`, etc.

The following steps are executed:

#### 1) Add features to the data (this takes a few minutes): 

This will take the raw data and preprocess it, adding some columns with features to the data.
```
python add_features.py --inp_path path_to_data
```

#### 2) Train model

The following command will train a Neural Network with aleatoric and epistemic uncertainty estimates, and save the model in a new folder within the `trained_models` directory. The flag `-e` determines the number of epochs that the model is trained.
```
python train.py -m nn -o out_dir_name -e 50
```

All code to train and test the neural network with aleatoric and epistemic uncertainty estimation is provided [here](train_delay/mlp_model.py).

#### 3) Evaluate the results

We provide a script to run the model on test data and to compute the MSE, MAE, prediction interval width (as a metric for the precision of the uncertainty-enhanced predictions) and the likeliness of realization. This script evaluates all models in the specified folder.

```
python run.py -m model_folder_name
```

(here, model_folder_name would be the same as out_dir_name above)

#### 4) Plot the results

Reproduce figures from our paper with the following command:
```
python plotting.py
```
