import pandas as pd
import joblib


def main():
    train = pd.read_feather('../data/input/train_data.feather')
    joblib.dump(train['salary'], '../pickle/salary_raw.pkl')


if __name__ == '__main__':
    main()
