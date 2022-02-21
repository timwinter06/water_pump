""" Helper functions for the train_model.py script. Functions included are:
 * feature_engineer
 * preprocess
 * finalize_data
"""
import pandas as pd


def feature_engineer(x: pd.DataFrame, replace_low_freq: list):
    """
    Function where you perform any feature engineering. The following steps are performed:
    * Fill NaN with 'unknown'
    * Year and month column are created from 'date_recorded' column
    * Year column has the minimum year subtracted from it.
    * Replace low freq categories with "other", for columns in 'replace_low_freq' column.
    :param pd.DataFrame x: dataframe with features.
    :param list replace_low_freq: List of columns you want to replace low freq. categories of.
    :return pd.DataFrame x: dataframe with feature engineered features.
    """
    x = x.fillna('unknown')
    x['date_recorded'] = pd.to_datetime(x.date_recorded)
    x['year'] = x['date_recorded'].dt.year
    x['month'] = x['date_recorded'].dt.month
    x['year'] = x['year'] - x['year'].min()
    for col in replace_low_freq:
        x.loc[x[col].value_counts()[x[col]].values < 20, col] = "other"
    return x


def preprocess(x: pd.DataFrame, type_dict: dict, drop_these: list):
    """
    Function where you perform any preprocessing. The following steps are performed:
    * Drop any duplicates
    * Make sure some features you select are the correct type
    * Remove unrealistic values
    * Fill NaN with 'unknown'
    * Drop columns you do not want to include.
    :param pd.DataFrame x: dataframe with features
    :param dict type_dict: dictionary with types you want features to be in.
    :param list drop_these: list with columns you want to drop.
    :return pd.DataFrame x: dataframe with preprocessed features.
    """
    x = x.drop_duplicates()
    x = x.astype(dtype=type_dict)
    x = x[x.longitude != 0]
    x = x[x.latitude != 0]
    x = x.drop(drop_these, axis=1)
    return x


def finalize_data(x: pd.DataFrame, y: pd.DataFrame):
    """
    Function to make sure the input features and the target match.
    :param pd.DataFrame x: dataframe with features
    :param pd.DataFrame y: dataframe with targets
    :return (pd.DataFrame, pd.Dataframe) (x, y):
    """
    y = pd.merge(x, y, how='left', left_on='id', right_on='id')
    y = y['status_group']
    x = x.drop('id', axis=1)
    return x, y


