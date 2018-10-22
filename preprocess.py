import numpy as np
from utils import transform_datetime_features
from utils import drop_const_cols, filter_columns, std_scaler
from utils import count_encoding


def preprocess(df, model_config, type='train'):
    """preprocessing and feature engineering for input data"""

    print('preprocess data..')

    # extract datetime features
    df = transform_datetime_features(df)
    print('datetime features extracted')

    # categorical count encoding
    if type == 'train':
        df, categorical_values = count_encoding(df)
        model_config['categorical_values'] = categorical_values
    elif type=='test':
        df = count_encoding(df, model_config['categorical_values'])
    print('count encoding of categorical features added')

    # drop constant features
    if type == 'train':
        df = drop_const_cols(df)

    # scaling
    # if mtype == 'train':
    #     df, scaler_mean, scaler_std = std_scaler(df)
    #     model_config['scaler_mean'] = scaler_mean
    #     model_config['scaler_std'] = scaler_std
    # elif type=='test':
    #     df = model_config['scaler'].transform(df)

    # filter columns
    if type == 'train':
        df, used_columns = filter_columns(df, groups=['number', 'count'])
        model_config['used_columns'] = used_columns
    elif type=='test':
        df_pred = df[['line_id']]
        df = df[model_config['used_columns']]

    # missing values
    df.fillna(-1, inplace=True)

    # convert if dataframe is too big
    if model_config['is_big']:
        df = df.astype(np.float32)

    if type == 'train':
        return df, model_config
    else:
        return df, df_pred
