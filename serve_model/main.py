""" This file creates and API with fastapi to serve the trained model. """
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb
import pickle
import category_encoders as ce


# Load model
file = open('trained_model', 'rb')
clf = pickle.load(file)
file.close()
# Load input format
df_format = pd.read_csv('input_data_format.csv')
# Load encoder
file = open(f'ordinal_encoder', 'rb')
encoder = pickle.load(file)
file.close()
# Load target_mapping
file = open('target_map', 'rb')
target_map = pickle.load(file)
file.close()


# Define object we classify
class WaterPumpFeatures(BaseModel):
    amount_tsh: float
    funder: str
    gps_height: float
    installer: str
    longitude: float
    latitude: float
    basin: str
    region_code: str
    lga: str
    ward: str
    public_meeting: str
    scheme_management: str
    permit: str
    extraction_type_class: str
    management_group: str
    payment_type: str
    quality_group: str
    quantity_group: str
    source_type: str
    source_class: str
    waterpoint_type_group: str
    year: float
    month: str


def get_predictions(input_data: WaterPumpFeatures):
    """
    This function takes input data features and the most likely predicted class.
    :param WaterPumpFeatures input_data: input features to be classified.
    :return (int, np.array) pred, prob: mostly likely class and the probabilities for each class.
    """
    input_dict = {
        'amount_tsh': input_data.amount_tsh,
        'funder': input_data.funder,
        'gps_height': input_data.gps_height,
        'installer': input_data.installer,
        'longitude': input_data.longitude,
        'latitude': input_data.latitude,
        'basin': input_data.basin,
        'region_code': input_data.region_code,
        'lga': input_data.lga,
        'ward': input_data.ward,
        'public_meeting': input_data.public_meeting,
        'scheme_management': input_data.scheme_management,
        'permit': input_data.permit,
        'extraction_type_class': input_data.extraction_type_class,
        'management_group': input_data.management_group,
        'payment_type': input_data.payment_type,
        'quality_group': input_data.quality_group,
        'quantity_group': input_data.quantity_group,
        'source_type': input_data.source_type,
        'source_class': input_data.source_class,
        'waterpoint_type_group': input_data.waterpoint_type_group,
        'year': input_data.year,
        'month': input_data.month,
    }
    input_df = df_format.append(input_dict, ignore_index=True)
    input_df = encoder.transform(input_df)
    pred = clf.predict_proba(input_df)
    pred_dict = {key: pred[0][i] for i, key in enumerate(target_map.keys())}
    return pred_dict


app = FastAPI()


# Server Definition
@app.get("/")
def root():
    return {"GoTo": "/docs"}


@app.post("/water_pump_prediction")
def is_user_item(request: WaterPumpFeatures):
    # Get the predictions
    return get_predictions(request)
