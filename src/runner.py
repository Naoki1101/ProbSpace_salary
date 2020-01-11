import numpy as np
import pandas as pd
import logging

from sklearn.metrics import confusion_matrix

from models import RIDGRegression
from models import KNNClassifier, KNNRegressor
from models import SVMClassifier, SVMRegressor
from models import XGBClassifier, XGBRegressor
from models import LGBRegressor, LGBClassifier
from models import CBRegressor, CBClassifier
from models import NNRegressor#, NNClassifier
from metrics import Scorer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_predict(train_x, train_y, test_x, params, folds, model_name=None,
                      cat_features=None, feval=None, metrics=None, convert_type='raw'):

    unique_fold = np.sort(folds['fold_id'].unique())

    if convert_type == 'log':
        train_y = np.log1p(train_y)

    preds = np.zeros((len(test_x), len(unique_fold)))
    oof = np.zeros(len(train_x))
    scores = []
    models = []
    s = Scorer()

    for fold_ in unique_fold:
        print(f'\n\nfold{fold_}')
        logging.debug(f'\n\nfold{fold_}')

        tr_x, va_x = train_x[folds['fold_id'] != fold_], train_x[folds['fold_id'] == fold_]
        tr_y, va_y = train_y[folds['fold_id'] != fold_], train_y[folds['fold_id'] == fold_]

        if 'ridge' in model_name:
            model = RIDGRegression(params)
        elif 'knn_clf' in model_name:
            model = KNNClassifier(params)
        elif 'knn_reg' in model_name:
            model = KNNRegressor(params)
        elif 'svm_clf' in model_name:
            model = SVMClassifier(params)
        elif 'svm_reg' in model_name:
            model = SVMRegressor(params)
        elif 'xgb_clf' in model_name:
            model = XGBClassifier(params)
        elif 'xgb_reg' in model_name:
            model = XGBRegressor(params)
        elif 'lgbm_reg' in model_name:
            model = LGBRegressor(params)
        elif 'lgbm_clf' in model_name:
            model = LGBClassifier(params)
        elif 'cb_reg' in model_name:
            model = CBRegressor(params)
        elif 'cb_clf' in model_name:
            model = CBClassifier(params)
        elif 'nn_reg' in model_name:
            model = NNRegressor(params)
        # elif 'nn_clf' in model_name:
        #     model = NNClassifier(params)
        else:
            raise(NotImplementedError)

        model.fit(tr_x, tr_y, va_x, va_y, cat_features=cat_features, feval=feval)

        va_pred = model.predict(va_x, cat_features)
        oof[va_x.index] = va_pred

        if convert_type == 'log':
            va_y = np.where(np.expm1(va_y) >= 0, np.expm1(va_y), 0)
            va_pred = np.where(np.expm1(va_pred) >= 0, np.expm1(va_pred), 0)
            score = s.scorer(metrics, va_y, va_pred)
        else:
            score = s.scorer(metrics, va_y, va_pred)

        scores.append(np.round(score, 3))
        print(f'\nScore: {score}')
        logging.debug(f'Score: {score}')

        pred = model.predict(test_x, cat_features)
        if convert_type == 'log':
            pred = np.where(np.expm１(pred) >= 0, np.expm1(pred), 0)

        preds[:, fold_] = pred

        models.append(model)

    if convert_type == 'log':
        oof = np.where(np.expm1(oof) >= 0, np.expm1(oof), 0)

    print('\n\n===================================\n')
    print(f'CV: {np.mean(scores)}')
    logging.debug(f'\n\nCV: {np.mean(scores)}\n\n')
    print('\n===================================\n\n')

    return models, preds, oof, scores


def save_importances(run_name, models, features):
    if 'xgb' in run_name or 'lgbm' in run_name or 'cb' in run_name:
        df_feature_importance = pd.DataFrame()
        for fold_, model in enumerate(models):

            if 'lgbm' in run_name:
                fold_importance = model.extract_importances(imp_type='gain')
            elif 'cb' in run_name:
                fold_importance = model.extract_importances()
            elif 'xgb' in run_name:
                importance_dict = model.extract_importances(imp_type='total_gain')
                fold_importance = [importance_dict[f] for f in features]
            else:
                raise(NotImplementedError)

            df_fold_importance = pd.DataFrame()
            df_fold_importance['feature'] = features
            df_fold_importance['importance'] = fold_importance

            df_feature_importance = pd.concat([df_feature_importance, df_fold_importance], axis=0)

        df_unique_feature_importance = (df_feature_importance[['feature', 'importance']]
                                        .groupby('feature')
                                        .mean()
                                        .sort_values(by='importance', ascending=False))
        df_unique_feature_importance.to_csv(f'../logs/{run_name}/importances.csv', index=True)

        cols = df_unique_feature_importance.index
        df_best_features = df_feature_importance.loc[df_feature_importance['feature'].isin(cols)]

        plt.figure(figsize=(14, int(np.log(len(cols)) * 50)))
        sns.barplot(x='importance',
                    y='feature',
                    data=df_best_features.sort_values(by="importance",
                                                      ascending=False))

        plt.title(f'{run_name} Features (avg over folds)')
        plt.tight_layout()
        plt.savefig(f'../logs/{run_name}/importances_plot.png')


def save_oof_plot(run_name, y_true, y_pred, type_='reg', dia=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    if type_ == 'reg':
        ax1.hist(y_pred, bins=50)
        ax1.set_title('y_pred distribution')

        ax2.scatter(y_true, y_pred)
        max_ = max(max(y_true), max(y_pred))
        min_ = min(min(y_true), min(y_pred))
        if dia:
            ax2.plot([min_, max_], [min_, max_], 'orange', '-')
        pad_ = min_ * 0.1
        ax2.set_xlim([min_ - pad_, max_ + pad_])
        ax2.set_ylim([min_ - pad_, max_ + pad_])

        ax2.set_xlabel('y_true')
        ax2.set_ylabel('y_pred')

        ax2.set_title('scatter')

    plt.tight_layout()
    plt.savefig(f'../logs/{run_name}/oof_plot.png')


def save_learning_curve(run_name, models):
    num_model = len(models)
    r = int((num_model + 2) / 3)
    plt.figure(figsize=(16, 5 * r))

    for i, model in enumerate(models):
        all_train_loss, all_val_loss = model.get_train_log()
        plt.subplot(r, 3, i + 1)
        plt.plot(all_train_loss, c='blue', label='train')
        plt.plot(all_val_loss, c='orange', label='val')
        plt.legend()
        plt.title('learning curve')
    plt.tight_layout()
    plt.savefig(f'../logs/{run_name}/learning_curve.png')