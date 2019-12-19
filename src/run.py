import argparse
import logging
import numpy as np
import pandas as pd
import datetime
import time
import yaml

from sklearn.preprocessing import StandardScaler
from pathlib import Path

from utils import Timer, seed_everything
from utils import DataLoader, Yml, make_submission
from utils import send_line, send_notion
from runner import train_and_predict, save_importances, save_oof_plot

import warnings
warnings.filterwarnings('ignore')

# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument('--common', default='../configs/common/default.yml')
parser.add_argument('--notify', default='../configs/common/notify.yml')
parser.add_argument('-m', '--model')
parser.add_argument('-c', '--comment')
options = parser.parse_args()

yml = Yml()
config = yml.load(options.common)
config.update(yml.load(f'../configs/exp/{options.model}.yml'))

# ===============
# Constants
# ===============
COMMENT = options.comment
NOW = datetime.datetime.now()
MODEL_NAME = options.model
RUN_NAME = f'{MODEL_NAME}_{NOW:%Y%m%d%H%M%S}'

COMPE_PARAMS = config['compe']
SETTINGS_PARAMS = config['settings']
PATH_PARAMS = config['path']
MODEL_PARAMS = config['model_params']
NOTIFY_PARAMS = yml.load(options.notify)

FEATURES = SETTINGS_PARAMS['features']

LOGGER_PATH = Path(f'../logs/{RUN_NAME}')


# ===============
# Main
# ===============
t = Timer()
seed_everything(COMPE_PARAMS['seed'])

LOGGER_PATH.mkdir()
logging.basicConfig(filename=LOGGER_PATH / 'train.log', level=logging.DEBUG)

yml.save(LOGGER_PATH / 'config.yml', config)

with t.timer('load data and folds'):
    loader = DataLoader()
    train_x = loader.load_x(FEATURES, data_type='train', reduce=SETTINGS_PARAMS['reduce'])
    test_x = loader.load_x(FEATURES, data_type='test', reduce=SETTINGS_PARAMS['reduce'])
    train_y = loader.load_y(PATH_PARAMS['train_y'])
    folds = loader.load_folds(SETTINGS_PARAMS['fold_name'])

with t.timer('preprocessing'):
    if SETTINGS_PARAMS['drop_fname'] is not None:
        drop_idx = np.load(f'../pickle/{SETTINGS_PARAMS["drop_fname"]}')
        train_x = train_x.drop(drop_idx, axis=0).reset_index(drop=True)
        train_y = train_y.drop(drop_idx, axis=0).reset_index(drop=True)
        folds = folds.drop(drop_idx, axis=0).reset_index(drop=True)

    if SETTINGS_PARAMS['oof']['add'] is not None:
        OOF_RUN_NAME = SETTINGS_PARAMS['oof']['add']
        oof = np.load(f'../logs/{OOF_RUN_NAME}/oof.npy')
        pred = pd.read_csv(f'../data/output/{OOF_RUN_NAME}.csv')[COMPE_PARAMS['target_name']].values
        train_x['oof'] = oof
        test_x['oof'] = pred
        FEATURES += ['oof']

    if SETTINGS_PARAMS['std']:
        whole = pd.concat([train_x, test_x], axis=0)
        len_train = len(train_x)
        scaler = StandardScaler()
        whole = pd.DataFrame(scaler.fit_transform(whole), columns=whole.columns)
        train_x = whole.iloc[:len_train]
        test_x = whole.iloc[len_train:]


with t.timer('train and predict'):
    models, preds, oof, scores = train_and_predict(train_x, train_y, test_x,
                                                   MODEL_PARAMS,
                                                   folds,
                                                   model_name=MODEL_NAME,
                                                   cat_features=SETTINGS_PARAMS['categorical_features'],
                                                   feval=None,
                                                   metrics=SETTINGS_PARAMS['metrics'])

    logging.disable(logging.FATAL)

    if SETTINGS_PARAMS['oof']['save']:
        np.save(f'../logs/{RUN_NAME}/oof.npy', oof)
        save_oof_plot(RUN_NAME, train_y, oof, type_='reg', dia=True)

with t.timer('save features importances'):
    save_importances(RUN_NAME, models, FEATURES)

with t.timer('make submission'):
    output_path = f'../data/output/{RUN_NAME}_{np.mean(scores):.3f}.csv'
    make_submission(y_pred=np.mean(preds, axis=1), target_name=COMPE_PARAMS['target_name'],
                    sample_path=PATH_PARAMS['sample'], output_path=str(output_path), comp=False)

LOGGER_PATH.rename(f'../logs/{RUN_NAME}_{np.mean(scores):.3f}')

process_minutes = t.get_processing_time()

with t.timer('notify'):
    message = f'''{MODEL_NAME}\ncv: {np.mean(scores):.3f}\nscores: {scores}\ntime: {process_minutes:.2f}[min]'''

    send_line(NOTIFY_PARAMS['line']['token'], message)

    send_notion(token_v2=NOTIFY_PARAMS['notion']['token_v2'],
                url=NOTIFY_PARAMS['notion']['url'],
                name=RUN_NAME,
                created=NOW,
                model=MODEL_NAME.split('_')[0],
                local_cv=round(np.mean(scores), 4),
                time_=process_minutes,
                comment=COMMENT)
