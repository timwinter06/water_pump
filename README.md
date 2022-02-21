# Water-pump classifier

This repo contains the code to train a classifier on water-pump data, and then serve the trained model as an API. 
The repo contains two directories: 'train_model' and 'serve_model'. After the 'train_model.py' script in the 'train_model' is run, the 'serve_model' should contain all the necessary files to be made into a docker
container which can serve the model's prediction via an API-request. More detail is provided below. At the end a quick setup guide with instructions is provided. 

## train_model directory

The train_model directory contains a ``train_model.py`` and a ``helper.py`` file. The `train_model.py` file can be run to
train a classifier on the 'water_pump_set.csv' data in the root directory. The main steps of the file are:
* Load the 'water_pump_set.csv' data
* Perform feature-engineering and preprocessing on the data (logic defined in the `helper.py` file)
* Encodes the categorical features.
* Train a classifier on the data (at the moment it is a LightGBM, but it should relatively easy to switch to another model such as Random-Forest or XGBoost)
* Classification report of the model is printed ( includes metrics such as accuracy, precision, recall, & f1)

The ``train_model.py`` file also saves the following in the 'train_model' directory:
* 'json_test_input.json' (json file that can be used as input for testing the API)
* 'json_test_label.csv' (the correct label for the 'json_test_input.json')

And saves the following to the 'serve_model' directory:
* 'input_data_format.csv' (empty dataframe, where the headers are the input features required for the model)
* '*_encoder' (the encoder used to encode the categorical input data)
* 'target_map' (the numbers which the target labels have been mapped to)
* 'trained_lightgbm.txt' (the trained classifier).


The file can be run by going to the 'train_model' dir and running the following command in the command prompt: ``python train_model.py --model "lightgbm" --encode "ordinal"`` Note that for the model parameter the only option is 'lightgbm' but for the encode parameter you can choose either "ordinal" or "one-hot".

## serve_model directory
#### For running locally (without containers):
The serve_model directory contains an ``main.py`` file, and the files created and saved by the `train_model.py` script ( as described above). This file loads in the trained model, encoder, and the target-map.
It sets up an API with fast-api. At the endpoint "/water_pump_prediction" a request with features can be sent ( format defined in the `WaterPumpFeatures` class), and a dictionary with the classes and their predicted probabilities will be returned. 
An example of the input can de found in the 'train_model' directory under the name ``json_test_input.json``. The correct label is saved in the same dir under the name `json_test_label.csv`.
#### For building docker-containers:
The serve_model directory contains a Dockerfile, a docker-compose, cloudbuild.yml, requirements.txt, and .dockerignore file. 
For building and running a docker-container locally, the following command should be run:  `docker-compose.yml up --build`
For building and pushing to Google Cloud directly you can run `gcloud builds submit --config cloudbuild.yml .` ( Note that for this you need to have Google Cloud SDK set up, and need to create a Google Cloud project which you must set).


## Quick setup guide

This quide will provide the instructions to quickly train and save the model, set up the API (locally) to serve the model, and send a request to it and get the model's prediction.
* Make sure to install the requirements as follows: `pip install -r requirements.txt` 
* CD to the train_model dir. 
* Run the following command in the prompt: ``python train_model.py --model "lightgbm" --encode "ordinal"``
* CD to the serve_model dir.
* Run the following command in the prompt: ``uvicorn api:app``
* Go to the url displayed in the terminal ( probably something like: 'http://127.0.0.1:8000')
* Go to {YOUR_URL}/docs 
* Under the 'water_pump_prediction' header copy and paste the 'json_test_input.json' (located in the train_model dir)
* Click execute and check if the most likely label corresponds to the 'json_test_label.csv'