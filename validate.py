import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
import timeit
import mlflow

datasets = ['check_1_r', 'check_2_r', 'check_3_r', 'check_4_c', 'check_5_c', 'check_6_c', 'check_7_c', 'check_8_c']
# datasets = ['check_8_c']
result_dir = '../../res'
data_dir = '../../data'

mlflow.set_tracking_uri('../../mlruns')
mlflow.set_experiment('h2o')

with mlflow.start_run():

    for i, dataset in enumerate(datasets):

        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        if not os.path.exists('{}/{}'.format(result_dir, dataset)):
            os.mkdir('{}/{}'.format(result_dir, dataset))

        print('\n### Check dataset', dataset, '\n')

        train_time = timeit.default_timer()
        os.system('python train.py --mode {} --train-csv {} --model-dir {}'.format(
            'regression' if dataset[-1] == 'r' else 'classification',
            '{}/{}/train.csv'.format(data_dir, dataset),
            '{}/{}/'.format(result_dir, dataset)
        ))
        train_time = timeit.default_timer() - train_time

        pred_time = timeit.default_timer()
        os.system('python predict.py --prediction-csv {} --test-csv {} --model-dir {}'.format(
            '{}/{}/pred.csv'.format(result_dir, dataset),
            '{}/{}/test.csv'.format(data_dir, dataset),
            '{}/{}/'.format(result_dir, dataset)
        ))
        pred_time = timeit.default_timer() - pred_time

        df = pd.read_csv('{}/{}/test-target.csv'.format(data_dir, dataset))
        df_pred = pd.read_csv('{}/{}/pred.csv'.format(result_dir, dataset))
        df = pd.merge(df, df_pred, on='line_id', left_index=True)

        score = roc_auc_score(df.target.values, df.prediction.values) if dataset[-1] == 'c' else \
                np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
        print('Score {:0.5f}'.format(score))

        n = dataset.split('_')[1]
        mlflow.log_metric('score_{}'.format(n), score)
        mlflow.log_metric('train_time_{}'.format(n), train_time)
        mlflow.log_metric('test_time_{}'.format(n), pred_time)

    mlflow.log_artifacts('./')
