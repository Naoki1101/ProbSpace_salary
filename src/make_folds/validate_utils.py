import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


class FoldValidation:

    def __init__(self, x_arr, stratify_arr=None, fold_num=5, random_state=42, shuffle_flg=True, test_size=0.2):
        self.n_fold = fold_num
        self.random_state = random_state
        self.shuffle_flg = shuffle_flg
        self.x_arr = x_arr
        self.stratify_arr = stratify_arr
        self.df_fold = pd.DataFrame(np.nan, columns=['fold_id'], index=range(len(x_arr)))
        self.test_size = test_size

    def make_split(self, valid_type):
        if valid_type == 'KFold':
            folds = KFold(n_splits=self.n_fold,
                          shuffle=self.shuffle_flg,
                          random_state=self.random_state)
            list_valid_idx = [val_idx for tr_idx, val_idx in folds.split(self.x_arr, self.stratify_arr)]
        elif valid_type == 'StratifiedKFold':
            folds = StratifiedKFold(n_splits=self.n_fold,
                                    shuffle=self.shuffle_flg,
                                    random_state=self.random_state)
            list_valid_idx = [val_idx for tr_idx, val_idx in folds.split(self.x_arr, self.stratify_arr)]
        elif valid_type == 'train_test_split':
            list_valid_idx = train_test_split(range(len(self.x_arr)),
                                              test_size=self.test_size,
                                              shuffle=self.shuffle_flg,
                                              random_state=self.random_state)
        else:
            raise(NotImplementedError)

        for fold_, valid_index in enumerate(list_valid_idx):
            self.df_fold.loc[valid_index, 'fold_id'] = fold_

        return self.df_fold[['fold_id']].astype('int')
