import numpy as np


def target_encoding(tr, te, target, feat, fold):

    target_tr = np.zeros(len(tr))
    target_te = np.zeros(len(te))

    le = tr.groupby(feat)[target].mean().to_dict()
    target_te = te[feat].map(le).values

    for fold_ in fold['fold_id'].unique():
        X_tr = tr[fold['fold_id'] != fold_]
        X_val = tr[fold['fold_id'] == fold_]
        le = X_tr.groupby(feat)[target].mean().to_dict()
        target_tr[X_val.index] = X_val[feat].map(le).values

    return target_tr, target_te
