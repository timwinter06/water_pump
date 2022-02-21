""" Script to train model. The main steps are:
* load in feature data and labels.
* perform processing on data ( using functions from the helper.py file)
* Train model
* Save model and parameters.
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
import lightgbm as lgb
from sklearn.metrics import classification_report
from helper import feature_engineer, preprocess, finalize_data
import pickle
import json

# Model and encode must be set when running script
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model to train", type=str, choices=['lightgbm'])
parser.add_argument("--encode", help="type of encoder to use", type=str, choices=['ordinal', 'one-hot'])
args = parser.parse_args()

# Load data
X_original = pd.read_csv('../water_pump_set.csv')
y_original = pd.read_csv('../water_pump_labels.csv')

# Define some variables
# These columns will be checked for low occuring categories. These categories will be replaced by 'other'.
replace_low_freq = ['funder', 'installer', 'ward']
# Dictionary to make sure certain columns are the type you want.
type_dict = {
    'amount_tsh': 'float64',
    'gps_height': 'float64',
    'longitude': 'float64',
    'latitude': 'float64',
    'public_meeting': 'str',
    'permit': 'str',
    'region_code': 'str',
    'district_code': 'str',
    'month': 'str'
}
# Columns to be dropped.
drop_these = [
    'construction_year', 'wpt_name', 'num_private',
    'subvillage', 'region', 'scheme_name',
    'extraction_type', 'extraction_type_group','management',
    'payment', 'water_quality', 'quantity',
    'source', 'waterpoint_type', 'population',
    'recorded_by', 'date_recorded',
]

# Perform feature engineering and preprocessing
X = feature_engineer(X_original, replace_low_freq)
X = preprocess(X, type_dict, drop_these)
X, y = finalize_data(X, y_original)

# Map target to numeric value
target_map = {'functional': 0, 'non functional': 1, 'functional needs repair': 2}
y = y.map(lambda x: target_map[x])
# Save target map
outfile = open('../serve_model/target_map', 'wb')
pickle.dump(target_map, outfile)
outfile.close()

# Create train test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test-input as json for API
json_test = X_test.iloc[0].to_dict()
with open('json_test_input.json', 'w') as outfile:
    json.dump(json_test, outfile)
# Save label
y_test.to_frame().iloc[0].to_csv('json_test_label.csv', index=False)

# Encode input data
if args.encode == 'ordinal':
    encoder = ce.OrdinalEncoder()
elif args.encode == 'one-hot':
    encoder = ce.OneHotEncoder()

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Save encoder
outfile = open(f'../serve_model/{args.encode}_encoder', 'wb')
pickle.dump(encoder, outfile)
outfile.close()

# Train model
if args.model == 'lightgbm':
    clf = lgb.LGBMClassifier()

# Maybe hyper-tune here?
clf.fit(X_train, y_train)

# Get metrics
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_map.keys()))

# Save input format for model
pd.DataFrame(columns=X_test.columns).to_csv('../serve_model/input_data_format.csv', index=False)

# Save model
if args.model == 'lightgbm':
    clf.booster_.save_model(f'../serve_model/trained_{args.model}.txt')










