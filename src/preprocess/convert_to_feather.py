import pandas as pd
import glob


def main():
    extension = 'csv'

    path_list = glob.glob(f'../data/input/*.{extension}')

    for path in path_list:
        (pd.read_csv(path, encoding="utf-8"))\
            .to_feather(path.replace(extension, 'feather'))


if __name__ == '__main__':
    main()
