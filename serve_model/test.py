import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb
import pickle
import category_encoders as ce

# Load model
clf = lgb.Booster(model_file='../train_model/trained_lightgbm.txt')
# Load input format
df_format = pd.read_csv('../train_model/input_data_format.csv')
# Load encoder
file = open(f'../train_model/ordinal_encoder', 'rb')
encoder = pickle.load(file)
file.close()

input_dict = {
    "amount_tsh": 0.0,
    "funder": "Danida",
    "gps_height": -0.31981981981981983,
    "installer": "Central government",
    "longitude": -0.5707736181298496,
    "latitude": -0.08388984220594153,
    "basin": "Lake Nyasa",
    "region_code": "12",
    "district_code": "3",
    "lga": "Kyela",
    "ward": "Lusungo",
    "public_meeting": "True",
    "scheme_management": "VWC",
    "permit": "True",
    "extraction_type_class": "gravity",
    "management_group": "user-group",
    "payment_type": "monthly",
    "quality_group": "good",
    "quantity_group": "enough",
    "source_type": "spring",
    "source_class": "groundwater",
    "waterpoint_type_group": "communal standpipe",
    # "year": "0",
    "month": "7"
}

input_df = df_format.append(input_dict, ignore_index=True)
input_df = encoder.transform(input_df)

pred = clf.predict(input_df)
# prob = clf.predict_proba(input_df)

print(str(pred[0]))