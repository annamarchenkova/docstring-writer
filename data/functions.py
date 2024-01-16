import os
from typing import  List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.outliers_influence import variance_inflation_factor

from project_dirs import PROJECT_DIR

##########################################################################################
# ---------------------------------  DS PREPROCESSING  ---------------------------------- #

def load_config(cnf_dir=PROJECT_DIR, cnf_name='config.yml'):
    """_summary_

    Args:
        cnf_dir (_type_, optional): _description_. Defaults to PROJECT_DIR.
        cnf_name (str, optional): _description_. Defaults to 'config.yml'.

    Returns:
        _type_: _description_
    """
    config_file = open(os.path.join(cnf_dir, cnf_name))
    return yaml.load(config_file, yaml.FullLoader)

def get_cols_too_similar(data, threshold=0.95):
    """
    Find features with too many similar values.
    :return: the pandas dataframe of sought features with the fraction of values which are similar, 
             as well as a list containing the most present value.
    
    :data: (pd.DataFrame) dataset
    :threshold: (float, default=0.95) fraction of similar values, must be a number in [0,1] interval
    """
    
    L = len(data)
    
    cols_counts = list()

    for col in data.columns:
        try:
            unique_values, unique_counts = np.unique(data[col].values, return_counts=True)
        except TypeError:
            unique_values, unique_counts = np.unique(data[col].astype(str).values, return_counts=True)

        idx_max = np.argmax(unique_counts)
        cols_counts.append((col, unique_values[idx_max], unique_counts[idx_max]))
    
    colname_and_values = map(lambda x: (x[0], x[2]), cols_counts)
    most_present_value = map(lambda x: x[1], cols_counts)

    df_similar_values = pd.DataFrame(colname_and_values)\
        .rename(columns={0: 'col_name', 1: 'frac'})\
        .sort_values('frac', ascending=False)

    df_similar_values['frac'] = df_similar_values['frac'].apply(lambda x: x / L)
    df_similar_values.query('frac >= @threshold', inplace=True)
    
    return df_similar_values, list(most_present_value)


def fill_nan_categorical_w_value(df, fill_with='Not Available'):
 
    nan_cols_cat = df.isna().sum()[(df.isna().sum() > 0) & (df.dtypes == 'object')].index.values

    for column in nan_cols_cat:
        df[column] = df[column].fillna(fill_with)
        
    return df


def get_non_collinear_features_from_vif(data, vif_threshold=5, idx=0):
  
    num_features = [i[0] for i in data.dtypes.items() if i[1] in ['float64', 'float32', 'int64', 'int32']]
    df = data[num_features].copy()
    
    if idx >= len(num_features):
        return df.columns.to_list()

    else:
        print('\rProcessing feature {}/{}'.format(idx+1, len(num_features)), end='')
        vif_ = variance_inflation_factor(df, idx)

        if vif_ > vif_threshold:
            df.drop(num_features[idx], axis=1, inplace=True)
            return get_non_collinear_features_from_vif(df, idx=idx, vif_threshold=vif_threshold)

        else:
            return get_non_collinear_features_from_vif(df, idx=idx+1, vif_threshold=vif_threshold)

def find_cols_w_2many_nan(
        data: pd.DataFrame,
        *,
        thr:float=0.95, 
        f_display:bool=False) -> Tuple[List[str], Optional[pd.DataFrame]]:
    
    na_cols = data.columns[data.isna().any()]

    df_nans = data[na_cols].copy() \
                .isna().sum() \
                .apply(lambda x: x / data.shape[0]) \
                .reset_index().rename(columns={0: 'f_nans', 'index': 'feature_name'}) \
                .sort_values(by='f_nans', ascending=False)
    
    cols_2_many_nans = df_nans.loc[df_nans.f_nans >= thr, 'feature_name'].to_list()

    if f_display:
        disp_df = df_nans.style.background_gradient(axis=0, gmap=df_nans['f_nans'], cmap='Oranges')
        return cols_2_many_nans, disp_df
    else:
        return cols_2_many_nans
    
    
