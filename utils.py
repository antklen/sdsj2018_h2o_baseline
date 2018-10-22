import pandas as pd


def transform_datetime_features(df):
    """extract datetime features"""

    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]

    for col_name in datetime_columns:
        if len(datetime_columns) < 10:
            df[col_name] = pd.to_datetime(df[col_name])
            df['number_weekday_{}'.format(col_name)] = df[col_name].dt.weekday
            df['number_month_{}'.format(col_name)] = df[col_name].dt.month
            df['number_day_{}'.format(col_name)] = df[col_name].dt.day
            df['number_hour_{}'.format(col_name)] = df[col_name].dt.hour
            df['number_hour_of_week_{}'.format(col_name)] = df[col_name].dt.hour + df[col_name].dt.weekday * 24
            df['number_minute_of_day_{}'.format(col_name)] = df[col_name].dt.minute + df[col_name].dt.hour * 60
        else:
            df[col_name] = pd.to_datetime(df[col_name])
            df['number_weekday_{}'.format(col_name)] = df[col_name].dt.weekday
            df['number_month_{}'.format(col_name)] = df[col_name].dt.month
            df['number_day_{}'.format(col_name)] = df[col_name].dt.day
            df['number_hour_{}'.format(col_name)] = df[col_name].dt.hour

    return df


def drop_const_cols(df):
    """drop constant columns"""

    constant_columns = [
        col_name
        for col_name in df.columns
        if df[col_name].nunique() == 1
        ]
    df.drop(constant_columns, axis=1, inplace=True)

    return df


def count_encoding(df, categorical_values=None):
    """count encoding of categorical features"""

    # train stage
    if categorical_values is None:
        categorical_values = {}
        for col_name in list(df.columns):
                if col_name.startswith('id') or col_name.startswith('string'):
                    categorical_values[col_name] = df[col_name].value_counts().to_dict()
                    df['count_{}'.format(col_name)] = df[col_name] \
                        .map(lambda x: categorical_values[col_name].get(x, 0))
        return df, categorical_values

    # test stage
    else:
        for col_name in list(df.columns):
            if col_name in categorical_values:
                df['count_{}'.format(col_name)] = df[col_name] \
                    .map(lambda x: categorical_values[col_name].get(x, 0))
        return df


def filter_columns(df, groups=['number']):
    """filter columns to use in model"""

    used_columns = []
    for gr in groups:
        used_columns += [col_name for col_name in df.columns
                        if col_name.startswith(gr)]
    df = df[used_columns]

    return df, used_columns


def std_scaler(df, scaler_mean=None, scaler_std=None):
    """standard scaler"""

    # train stage
    if scaler_mean is None:

        scaler_mean = {}
        scaler_std = {}
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].mean()
            df[col] = (df[col]-mean)/std
            scaler_mean[col] = mean
            scaler_std[col] = std

        return df, scaler_mean, scaler_std

    # test stage
    else:

        for col in df.columns:
            df[col] = (df[col]-scaler_mean[col])/scaler_std[col]

        return df
