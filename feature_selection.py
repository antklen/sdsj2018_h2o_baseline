import pandas as pd
import numpy as np
import lightgbm as lgb


def lgb_model(params, mode):

    if mode == 'regression':
        model = lgb.LGBMRegressor(**params)
    else:
        model = lgb.LGBMClassifier(**params)
    return model


def lgb_importance_fs(df, y, mode, BIG_DATASET_SIZE):
    """choose best features based  on lightgbm feature importance"""

    print('lightgbm feature selection..')

    # coefficient for taking fraction of data (to be sure that there won't be memory error)
    coef = 0.5

    # dataframe size
    df_size = df.memory_usage(deep=True).sum()

    # get subset of data if df is too big
    subset_size = min(df.shape[0], int(coef * df.shape[0] / (df_size / BIG_DATASET_SIZE)))
    print('subset_size {}'.format(subset_size))
    idx = np.random.choice(df.index, size=subset_size, replace=False)

    # define model
    params = {'n_estimators': 100, 'learning_rate': 0.05, 'num_leaves': 200,
              'subsample': 1, 'colsample_bytree': 1, 'random_state': 42, 'n_jobs': -1}
    model = lgb_model(params, mode)

    # train model
    model.fit(df.loc[idx], y.loc[idx])

    # feature importance
    feature_importance = pd.Series(model.booster_.feature_importance('gain'),
        index=df.columns).fillna(0).sort_values(ascending=False)
    # print(feature_importance.head(50))
    # print(feature_importance.tail(10))

    # remove totally unimportant features
    best_features = feature_importance[feature_importance>0]

    # leave most relevant features for big dataset
    if df_size > BIG_DATASET_SIZE:
        new_feature_count = min(df.shape[1], int(coef * df.shape[1] / (df_size / BIG_DATASET_SIZE)))
        best_features = best_features.head(new_feature_count)

    # select features
    used_columns = best_features.index.tolist()
    df = df[used_columns]

    print('feature selection done')
    print('number of selected features {}'.format(len(used_columns)))

    return df, used_columns
