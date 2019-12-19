import glob
import random
import os
import time
import yaml
from contextlib import contextmanager

import numpy as np
import pandas as pd
import requests
import dropbox
from sklearn.preprocessing import StandardScaler
from notion.client import NotionClient

from pathlib import Path
import joblib


class Timer:

    def __init__(self):
        self.processing_time = 0

    @contextmanager
    def timer(self, name):
        t0 = time.time()
        yield
        t1 = time.time()
        processing_time = t1 - t0
        self.processing_time += round(processing_time, 2)
        if self.processing_time < 60:
            print(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time:.2f} sec)')
        elif self.processing_time < 3600:
            print(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 60:.2f} min)')
        else:
            print(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 3600:.2f} hour)')

    def get_processing_time(self):
        return round(self.processing_time, 2)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# =============================================================================
# Data
# =============================================================================
class DataLoader:

    def load_x(self, features, data_type='train', reduce=False):
        dfs = [pd.read_feather(f'../features/{f}_{data_type}.feather') for f in features]
        df = pd.concat(dfs, axis=1)

        if reduce:
            df = self.reduce_mem_usage(df)

        return df

    def load_y(self, train_y_path):
        train_y = joblib.load(train_y_path)
        return train_y

    def load_folds(self, fold_name):
        folds = pd.read_feather(f'../folds/{fold_name}.feather')
        return folds

    def reduce_mem_usage(self, df):
        start_mem = df.memory_usage().sum() / 1024 ** 2
        # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        # print("column = ", len(df.columns))
        for i, col in enumerate(df.columns):
            try:
                col_type = df[col].dtype

                if col_type != object:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int32)
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float32)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float32)
            except:
                continue

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df


def make_submission(y_pred, target_name, sample_path, output_path, comp=False):
    df_sub = pd.read_feather(sample_path)
    df_sub[target_name] = y_pred
    if comp:
        output_path += '.gz'
        df_sub.to_csv(output_path, index=False, compression='gzip')
    else:
        df_sub.to_csv(output_path, index=False)


class Yml:

    def load(self, path):
        with open(path, 'r') as yf:
            yaml_file = yaml.load(yf)
        return yaml_file

    def save(self, path, data):
        with open(path, 'w') as yf:
            yf.write(yaml.dump(data, default_flow_style=False))


# =============================================================================
# Kaggle API
# =============================================================================
def kaggle_submit(submit_path, local_cv):
    cmd = f'kaggle competitions submit -c {self.compe_name} -f {submit_path}  -m "{local_cv}"'
    os.system(cmd)


# =============================================================================
# Notification
# =============================================================================
def send_line(line_token, message):
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)


def send_notion(token_v2, url, name, created, model, local_cv, time_, comment):
    client = client = NotionClient(token_v2=token_v2)
    cv = client.get_collection_view(url)
    row = cv.collection.add_row()
    row.name = name
    row.created = created
    row.model = model.split('_')[0]
    row.local_cv = local_cv
    row.time = time_
    row.comment = comment


def transfar_dropbox(input_path, output_path, token):
    dbx = dropbox.Dropbox(token)
    dbx.users_get_current_account()
    with open(input_path, 'rb') as f:
        dbx.files_upload(f.read(), output_path)
