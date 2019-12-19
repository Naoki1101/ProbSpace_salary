import numpy as np
import pandas as pd

from base import Feature, get_arguments, generate_features
from feature_utils import target_encoding

import warnings
warnings.filterwarnings('ignore')


# positionのターゲットエンコーディング(folds2)
class position_target_encoding_folds2(Feature):
    def create_features(self):
        folds = pd.read_feather('../folds/02_stkfold.feather')
        tr_feat, te_feat = target_encoding(train, test, 'salary', 'position', folds)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


# areaのターゲットエンコーディング(folds2)
class area_target_encoding_folds2(Feature):
    def create_features(self):
        folds = pd.read_feather('../folds/02_stkfold.feather')
        tr_feat, te_feat = target_encoding(train, test, 'salary', 'area', folds)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


# sexのターゲットエンコーディング(folds2)
class sex_target_encoding_folds2(Feature):
    def create_features(self):
        folds = pd.read_feather('../folds/02_stkfold.feather')
        tr_feat, te_feat = target_encoding(train, test, 'salary', 'sex', folds)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


# educationのターゲットエンコーディング(folds2)
class education_target_encoding_folds2(Feature):
    def create_features(self):
        folds = pd.read_feather('../folds/02_stkfold.feather')
        tr_feat, te_feat = target_encoding(train, test, 'salary', 'education', folds)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


# position + educationのターゲットエンコーディング(folds2)
class position_education_target_encoding_folds2(Feature):
    def create_features(self):
        folds = pd.read_feather('../folds/02_stkfold.feather')
        train['position_education'] = train['position'].astype(str) + '_' + train['education'].astype(str)
        test['position_education'] = test['position'].astype(str) + '_' + test['education'].astype(str)
        tr_feat, te_feat = target_encoding(train, test, 'salary', 'position_education', folds)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


# positionのターゲットエンコーディング(folds3)
class position_target_encoding_folds3(Feature):
    def create_features(self):
        folds = pd.read_feather('../folds/03_stkfold.feather')
        tr_feat, te_feat = target_encoding(train, test, 'salary', 'position', folds)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


# areaのターゲットエンコーディング(folds3)
class area_target_encoding_folds3(Feature):
    def create_features(self):
        folds = pd.read_feather('../folds/03_stkfold.feather')
        tr_feat, te_feat = target_encoding(train, test, 'salary', 'area', folds)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


# sexのターゲットエンコーディング(folds3)
class sex_target_encoding_folds3(Feature):
    def create_features(self):
        folds = pd.read_feather('../folds/03_stkfold.feather')
        tr_feat, te_feat = target_encoding(train, test, 'salary', 'sex', folds)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


# educationのターゲットエンコーディング(folds3)
class education_target_encoding_folds3(Feature):
    def create_features(self):
        folds = pd.read_feather('../folds/03_stkfold.feather')
        tr_feat, te_feat = target_encoding(train, test, 'salary', 'education', folds)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat


# position + educationのターゲットエンコーディング(folds3)
class position_education_target_encoding_folds3(Feature):
    def create_features(self):
        folds = pd.read_feather('../folds/03_stkfold.feather')
        train['position_education'] = train['position'].astype(str) + '_' + train['education'].astype(str)
        test['position_education'] = test['position'].astype(str) + '_' + test['education'].astype(str)
        tr_feat, te_feat = target_encoding(train, test, 'salary', 'position_education', folds)
        self.train[self.__class__.__name__] = tr_feat
        self.test[self.__class__.__name__] = te_feat





if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('../data/input/train_data.feather')
    test = pd.read_feather('../data/input/test_data.feather')

    len_train = len(train)

    generate_features(globals(), args.force)
