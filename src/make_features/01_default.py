import numpy as np
import pandas as pd

from base import Feature, get_arguments, generate_features

import warnings
warnings.filterwarnings('ignore')


# position
class position(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# age
class age(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# area
class area(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        le = {k: i for i, k in enumerate(whole[self.__class__.__name__].unique())}
        self.train[self.__class__.__name__] = train[self.__class__.__name__].map(le)
        self.test[self.__class__.__name__] = test[self.__class__.__name__].map(le)


# areaのone-hot-encoding
class area_onehot(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        df_dummy = pd.get_dummies(data=whole['area'], columns=['area'])
        for col in df_dummy.columns:
            col_name = f'area_{col}'
            self.train[col_name] = df_dummy[col].values[:len_train]
            self.test[col_name] = df_dummy[col].values[len_train:]


# areaのカウントエンコーディング
class area_count_encoding(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        le = whole.groupby('area').size().to_dict()
        self.train[self.__class__.__name__] = train['area'].map(le)
        self.test[self.__class__.__name__] = test['area'].map(le)


# sex
class sex(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# partner
class partner(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# num_child
class num_child(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# partner + num_child
class num_family(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train['partner'] + train['num_child']
        self.test[self.__class__.__name__] = test['partner'] + test['num_child']


# education
class education(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# service_length
class service_length(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# study_time
class study_time(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# commute
class commute(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# overtime
class overtime(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train[self.__class__.__name__]
        self.test[self.__class__.__name__] = test[self.__class__.__name__]


# positionとareaを結合&ラベルエンコーディング
class position_area(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole['position_area_str'] = whole['position'].astype(str) + '_' + whole['area']
        le = {k: i for i, k in enumerate(whole['position_area_str'].unique())}
        self.train[self.__class__.__name__] = whole['position_area_str'].map(le).values[:len_train]
        self.test[self.__class__.__name__] = whole['position_area_str'].map(le).values[len_train:]


# positionとeducationを結合&ラベルエンコーディング
class position_education(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole['position_education_str'] = whole['position'].astype(str) + '_' + whole['education'].astype(str)
        le = {k: i for i, k in enumerate(whole['position_education_str'].unique())}
        self.train[self.__class__.__name__] = whole['position_education_str'].map(le).values[:len_train]
        self.test[self.__class__.__name__] = whole['position_education_str'].map(le).values[len_train:]


# areaとpartnerを結合&ラベルエンコーディング
class area_partner(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole['area_partner_str'] = whole['area'] + '_' + whole['partner'].astype(str)
        le = {k: i for i, k in enumerate(whole['area_partner_str'].unique())}
        self.train[self.__class__.__name__] = whole['area_partner_str'].map(le).values[:len_train]
        self.test[self.__class__.__name__] = whole['area_partner_str'].map(le).values[len_train:]


# age / positionの平均年齢
class age_div_mean_each_position(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('position')['age'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# partner / positionの平均年齢
class partner_div_mean_each_position(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('position')['partner'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# num_child / positionの平均年齢
class num_child_div_mean_each_position(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('position')['num_child'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# education / positionの平均年齢
class education_div_mean_each_position(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('position')['education'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# service_length / positionの平均年齢
class service_length_div_mean_each_position(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('position')['service_length'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# study_time / positionの平均年齢
class study_time_div_mean_each_position(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('position')['study_time'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# commute / positionの平均年齢
class commute_div_mean_each_position(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('position')['commute'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# overtime / positionの平均年齢
class overtime_div_mean_each_position(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('position')['overtime'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# age / areaの平均年齢
class age_div_mean_each_area(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('area')['age'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# partner / areaの平均年齢
class partner_div_mean_each_area(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('area')['partner'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# num_child / areaの平均年齢
class num_child_div_mean_each_area(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('area')['num_child'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# education / areaの平均年齢
class education_div_mean_each_area(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('area')['education'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# service_length / areaの平均年齢
class service_length_div_mean_each_area(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('area')['service_length'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# study_time / areaの平均年齢
class study_time_div_mean_each_area(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('area')['study_time'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# commute / areaの平均年齢
class commute_div_mean_each_area(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('area')['commute'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# overtime / areaの平均年齢
class overtime_div_mean_each_area(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('area')['overtime'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# age / sexの平均年齢
class age_div_mean_each_sex(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('sex')['age'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# partner / sexの平均年齢
class partner_div_mean_each_sex(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('sex')['partner'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# num_child / sexの平均年齢
class num_child_div_mean_each_sex(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('sex')['num_child'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# education / sexの平均年齢
class education_div_mean_each_sex(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('sex')['education'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# service_length / sexの平均年齢
class service_length_div_mean_each_sex(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('sex')['service_length'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# study_time / sexの平均年齢
class study_time_div_mean_each_sex(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('sex')['study_time'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# commute / sexの平均年齢
class commute_div_mean_each_sex(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('sex')['commute'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# overtime / sexの平均年齢
class overtime_div_mean_each_sex(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('sex')['overtime'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# age / educationの平均年齢
class age_div_mean_each_education(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('education')['age'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# partner / educationの平均年齢
class partner_div_mean_each_education(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('education')['partner'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# num_child / educationの平均年齢
class num_child_div_mean_each_education(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('education')['num_child'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# education / educationの平均年齢
class education_div_mean_each_education(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('education')['education'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# service_length / educationの平均年齢
class service_length_div_mean_each_education(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('education')['service_length'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# study_time / educationの平均年齢
class study_time_div_mean_each_education(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('education')['study_time'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# commute / educationの平均年齢
class commute_div_mean_each_education(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('education')['commute'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# overtime / educationの平均年齢
class overtime_div_mean_each_education(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole[self.__class__.__name__] = whole['age'] / whole.groupby('education')['overtime'].transform('mean')
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]


# age - servece_length　新卒かどうかを判定できる？
class age_diff_service_length(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train['age'] - train['service_length']
        self.test[self.__class__.__name__] = test['age'] - test['service_length']


# 結婚の有無・東京大阪勤務かどうかでグループ分け
class partner_capital_group(Feature):
    def create_features(self):
        whole = pd.concat([train, test], axis=0)
        whole['is_tokyo'] = whole['area'].apply(lambda x: 1 if x == '東京都' else 0)
        whole['is_osaka'] = whole['area'].apply(lambda x: 1 if x == '大阪府' else 0)
        whole['is_capital'] = whole['is_tokyo'] + whole['is_osaka']
        whole['partner_capital_group_str'] = whole['partner'].astype(str) + '_' + whole['is_capital'].astype(str)
        le = {'0_0': 0, '0_1': 1, '1_0': 2, '1_1': 3}
        whole[self.__class__.__name__] = whole['partner_capital_group_str'].map(le)
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_train]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_train:]



if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('../data/input/train_data.feather')
    test = pd.read_feather('../data/input/test_data.feather')

    len_train = len(train)

    generate_features(globals(), args.force)
