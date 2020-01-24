from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
import xgboost as xgb
import lightgbm as lgb
import optuna.integration.lightgbm_tuner as lgb_tuner
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from lightgbm.callback import _format_eval_result

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

import logging


class Model(metaclass=ABCMeta):

    def __init__(self, params):
        self.params = params
        self.model = None

    @abstractmethod
    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None):
        pass

    @abstractmethod
    def predict(self, te_x, cat_features=None):
        pass


# ===============
# Ridge
# ===============
class RIDGRegression(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        ridge = Ridge(**self.params)
        self.model = ridge.fit(tr_x, tr_y)

    def predict(self, te_x, cat_features=None):
        return self.model.predict(te_x)


# ===============
# KNN
# ===============
class KNNClassifier(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        knn = KNeighborsClassifier(**self.params)
        self.model = knn.fit(tr_x, tr_y)

    def predict(self, te_x, cat_features=None):
        return self.model.predict(te_x)


class KNNRegressor(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        knn = KNeighborsRegressor(**self.params)
        self.model = knn.fit(tr_x, tr_y)

    def predict(self, te_x, cat_features=None):
        return self.model.predict(te_x)


# ===============
# SVM
# ===============
class SVMClassifier(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):
        
        svm = SVC(**self.params)
        self.model = svm.fit(tr_x, tr_y)

    def predict(self, te_x, cat_features=None):
        return self.model.predict(te_x)


class SVMRegressor(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):
        
        svm = SVR(**self.params)
        self.model = svm.fit(tr_x, tr_y)

    def predict(self, te_x, cat_features=None):
        return self.model.predict(te_x)


# ===============
# Xgboost
# ===============
class XGBClassifier(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):
        validation = va_x is not None
        xgb_train = xgb.DMatrix(tr_x, label=tr_y)
        if validation:
            xgb_eval = xgb.DMatrix(va_x, label=va_y)
            watchlist = [(xgb_train, 'train'), (xgb_eval, 'valid')]

        evals_result = {}

        if validation:
            self.model = xgb.train(self.params,
                                   xgb_train,
                                   num_boost_round=10000,
                                   evals=watchlist,
                                   verbose_eval=200,
                                   early_stopping_rounds=200,
                                   evals_result=evals_result,
                                   feval=feval)
        else:
            self.model = xgb.train(self.params,
                                   xgb_train,
                                   num_boost_round=10000)

    def predict(self, te_x, cat_features=None):
        xgb_test = xgb.DMatrix(te_x)
        return self.model.predict(xgb_test, ntree_limit=self.model.best_ntree_limit)

    def extract_importances(self, imp_type):
        return self.model.get_score(importance_type=imp_type)


class XGBRegressor(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):
        validation = va_x is not None
        xgb_train = xgb.DMatrix(tr_x, label=tr_y)
        if validation:
            xgb_eval = xgb.DMatrix(va_x, label=va_y)
            watchlist = [(xgb_train, 'train'), (xgb_eval, 'valid')]

        evals_result = {}

        if validation:
            self.model = xgb.train(self.params,
                                   xgb_train,
                                   num_boost_round=10000,
                                   evals=watchlist,
                                   verbose_eval=200,
                                   early_stopping_rounds=200,
                                   evals_result=evals_result,
                                   feval=feval)
        else:
            self.model = xgb.train(self.params,
                                   xgb_train,
                                   num_boost_round=10000)

    def predict(self, te_x, cat_features=None):
        xgb_test = xgb.DMatrix(te_x)
        return self.model.predict(xgb_test, ntree_limit=self.model.best_ntree_limit)

    def extract_importances(self, imp_type):
        return self.model.get_score(importance_type=imp_type)


# ===============
# LightGBM
# ===============
class LGBClassifier(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        validation = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y, categorical_feature=cat_features)
        if validation:
            lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train, categorical_feature=cat_features)

        logger = logging.getLogger('main')
        callbacks = [self.log_evaluation(logger, period=100)]

        if validation:
            self.model = lgb.train(self.params,
                                   lgb_train,
                                   num_boost_round=10000,
                                   valid_sets=[lgb_eval],
                                   verbose_eval=200,
                                   early_stopping_rounds=200,
                                   callbacks=callbacks,
                                   feval=feval)
        else:
            self.model = lgb.train(self.params,
                                   lgb_train,
                                   num_boost_round=10000,
                                   callbacks=callbacks)

        logging.debug(self.model.best_iteration)
        logging.debug(self.model.best_score['valid_0'][self.params['metric']])

    def predict(self, te_x, cat_features=None):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    def extract_importances(self, imp_type='gain'):  # imp_type = 'gain' or 'split'
        return self.model.feature_importance(importance_type=imp_type)

    def log_evaluation(self, logger, period=1, show_stdv=True, level=logging.DEBUG):
        def _callback(env):
            if period > 0 and env.evaluation_result_list \
                    and (env.iteration + 1) % period == 0:
                result = '\t'.join([
                    _format_eval_result(x, show_stdv)
                    for x in env.evaluation_result_list
                ])
                logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
        _callback.order = 10
        return _callback


class LGBRegressor(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        validation = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y, categorical_feature=cat_features)
        if validation:
            lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train, categorical_feature=cat_features)

        logger = logging.getLogger('main')
        callbacks = [self.log_evaluation(logger, period=100)]

        if validation:
            self.model = lgb.train(self.params,
                                   lgb_train,
                                   num_boost_round=10000,
                                   valid_sets=[lgb_eval],
                                   verbose_eval=200,
                                   early_stopping_rounds=200,
                                   callbacks=callbacks,
                                   feval=feval)
        else:
            self.model = lgb.train(self.params,
                                   lgb_train,
                                   num_boost_round=10000,
                                   callbacks=callbacks)

        logging.debug(self.model.best_iteration)
        logging.debug(self.model.best_score['valid_0'][self.params['metric']])

    def predict(self, te_x, cat_features=None):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)

    def extract_importances(self, imp_type='gain'):  # imp_type = 'gain' or 'split'
        return self.model.feature_importance(importance_type=imp_type)

    def log_evaluation(self, logger, period=100, show_stdv=True, level=logging.DEBUG):
        def _callback(env):
            if period > 0 and env.evaluation_result_list \
                    and (env.iteration + 1) % period == 0:
                result = '\t'.join([
                    _format_eval_result(x, show_stdv)
                    for x in env.evaluation_result_list
                ])
                logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
        _callback.order = 10
        return _callback


class LGBTuner(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        validation = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y, categorical_feature=cat_features)
        if validation:
            lgb_eval = lgb.Dataset(va_x, va_y, reference=lgb_train, categorical_feature=cat_features)

        logger = logging.getLogger('main')

        best_params, tuning_history = dict(), list()
        if validation:
            self.model = lgb_tuner.train(self.params,
                                         lgb_train,
                                         num_boost_round=10000,
                                         valid_sets=[lgb_eval],
                                         verbose_eval=0,
                                         early_stopping_rounds=200,
                                         feval=feval,
                                         best_params=best_params,
                                         tuning_history=tuning_history)
        else:
            self.model = lgb_tuner.train(self.params,
                                         lgb_train,
                                         num_boost_round=10000,
                                         best_params=best_params,
                                         tuning_history=tuning_history)

        logging.debug('Best Params:', best_params)
        logging.debug('Tuning history:', tuning_history)

    def predict(self, te_x, cat_features=None):
        pass


# ===============
# Catboost
# ===============
class CBClassifier(Model):

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        if cat_features is not None:
            for col in cat_features:
                tr_x[col] = tr_x[col].astype('category')
                va_x[col] = va_x[col].astype('category')

        validation = va_x is not None
        cb_train = Pool(tr_x, label=tr_y, cat_features=cat_features)
        if validation:
            cb_valid = Pool(va_x, label=va_y, cat_features=cat_features)

        cb = CatBoostClassifier(**self.params)

        if validation:
            self.model = cb.fit(cb_train,
                                eval_set=cb_valid,
                                use_best_model=True,
                                verbose_eval=200,
                                plot=False)
        else:
            self.model = cb.fit(cb_train)

        logging.debug(self.model.best_iteration_)
        logging.debug(self.model.best_score_['validation'][self.params['eval_metric']])

    def predict(self, te_x, cat_features=None):
        for col in cat_features:
            te_x[col] = te_x[col].astype('category')

        return self.model.predict(te_x)

    def extract_importances(self):
        return self.model.feature_importances_


class CBRegressor(Model):
    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        if cat_features is not None:
            for col in cat_features:
                tr_x[col] = tr_x[col].astype('category')
                va_x[col] = va_x[col].astype('category')

        validation = va_x is not None
        cb_train = Pool(tr_x, label=tr_y, cat_features=cat_features)
        if validation:
            cb_valid = Pool(va_x, label=va_y, cat_features=cat_features)

        cb = CatBoostRegressor(**self.params)

        if validation:
            self.model = cb.fit(cb_train,
                                eval_set=cb_valid,
                                use_best_model=True,
                                verbose_eval=200,
                                plot=False)
        else:
            self.model = cb.fit(cb_train)

        logging.debug(self.model.best_iteration_)
        logging.debug(self.model.best_score_['validation'][self.params['eval_metric']])

    def predict(self, te_x, cat_features=None):
        for col in cat_features:
            te_x[col] = te_x[col].astype('category')

        return self.model.predict(te_x)

    def extract_importances(self):
        return self.model.feature_importances_


# ===============
# NN
# ===============
class CustomLinear(nn.Module):
    def __init__(self, input_features):
        super(CustomLinear, self).__init__()
        self.fc1 = nn.Linear(input_features, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 25)
        self.fc5 = nn.Linear(25, 1)
        self.dropout = nn.Dropout2d(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        return x


class NNRegressor(Model):
    def fit(self, tr_x, tr_y, va_x=None, va_y=None, cat_features=None, feval=None):

        epochs = self.params['epochs']
        batch_size = self.params['batch_size']
        lr = self.params['lr']
        device = self.params['device']
        early_stopping = self.params['early_stopping']

        validation = va_x is not None
        tr_x = torch.tensor(tr_x.values, dtype=torch.float32)
        tr_y = torch.tensor(tr_y.values, dtype=torch.float32)
        train = torch.utils.data.TensorDataset(tr_x, tr_y)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        if validation:
            va_x = torch.tensor(va_x.values, dtype=torch.float32)
            va_y = torch.tensor(va_y.values, dtype=torch.float32)
            valid = torch.utils.data.TensorDataset(va_x, va_y)
            valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        model = CustomLinear(tr_x.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=1e-4)

        best_loss = 1e+10
        counter = 0

        self.all_train_loss = []
        self.all_val_loss = []

        for epoch in range(epochs):

            if counter == early_stopping:
                break
            else:
                model.train()
                avg_loss = 0.
                for x_batch, y_batch in train_loader:
                    y_pred = model(x_batch.float())
                    loss = criterion(y_pred.float(), y_batch.float())
                    optimizer.zero_grad()
                    loss.backward()
                    # optimizer.step()
                    scheduler.step()
                    avg_loss += loss.item() / len(train_loader)
                    
                    model.eval()
                    valid_preds_fold = np.zeros((va_x.size(0)))
                    avg_val_loss = 0.
                    
                    for i, (x_batch, y_batch) in enumerate(valid_loader):
                        y_pred = model(x_batch.float()).detach()
                        avg_val_loss += criterion(y_pred.float(), y_batch.float()).item() / len(valid_loader)
                        valid_preds_fold[i * batch_size: (i+1) * batch_size] = y_pred.float().numpy()[:, 0]

                self.all_train_loss.append(avg_loss)
                self.all_val_loss.append(avg_val_loss)

                if best_loss >= avg_val_loss:
                    best_loss = avg_val_loss
                    counter = 0
                    self.model = model
                else:
                    counter += 1

                print(f"[{epoch + 1:02}]  training's mse: {avg_loss:.4f}     valid_1's mse: {avg_val_loss:.4f}")
                logging.debug(f"[{epoch + 1:02}]  training's mse: {avg_loss:.4f}     valid_1's mse: {avg_val_loss:.4f}")

    def predict(self, te_x, cat_features=None):

        batch_size = self.params['batch_size']
        
        test_preds_fold = np.zeros(len(te_x))
        x_test_tensor = torch.tensor(te_x.values, dtype=torch.float32)
        test = torch.utils.data.TensorDataset(x_test_tensor)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

        self.model.eval()
        for i, (x_batch,) in enumerate(test_loader):
            y_pred = self.model(x_batch.float()).detach()
            test_preds_fold[i * batch_size: (i+1) * batch_size] = y_pred.numpy()[:, 0]
        return test_preds_fold

    def get_train_log(self):
        return self.all_train_loss, self.all_val_loss

