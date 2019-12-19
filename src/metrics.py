import numpy as np
import scipy as sp
from functools import partial

from sklearn import metrics


class Scorer:

    def scorer(self, metrics, y_true, y_pred):
        if metrics == 'rmse':
            return self._rmse(y_true, y_pred)
        elif metrics == 'rmsle':
            return self._rmsle(y_true, y_pred)
        elif metrics == 'mae':
            return self._mae(y_true, y_pred)
        elif metrics == 'mape':
            return self._mape(y_true, y_pred)
        elif metrics == 'auc':
            return self._auc(y_true, y_pred)
        elif metrics == 'logloss':
            return self._logloss(y_true, y_pred)
        elif metrics == 'qwk':
            return self._qwk(y_true, y_pred)
        else:
            raise(NotImplementedError)

    # ===============
    # RMSE
    # ===============
    def _rmse(self, y_true, y_pred):
        return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    # ===============
    # RMSLE
    # ===============
    def _rmsle(self, y_true, y_pred):
        return np.sqrt(metrics.mean_squared_log_error(y_true, y_pred))

    # ===============
    # MAE
    # ===============
    def _mae(self, y_true, y_pred):
        return metrics.mean_absolute_error(y_true, y_pred)

    # ===============
    # MAPE
    # ===============
    def _mape(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true))

    # ===============
    # AUC
    # ===============
    def _auc(self, y_true, y_pred):
        return metrics.roc_auc_score(y_true, y_pred)

    # ===============
    # Logloss
    # ===============
    def _logloss(self, y_true, y_pred):
        return metrics.log_loss(y_true, y_pred)

    # ===============
    # QWK
    # ===============
    def _qwk(y_true, y_pred):
        return metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic')


class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.0]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3
        return X_p

    def coefficients(self):
        return self.coef_['x']
