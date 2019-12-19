import argparse
import inspect
from abc import ABCMeta, abstractmethod
import pandas as pd

from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--force', '-f', action='store_true', help='Overwrite existing files'
    )
    return parser.parse_args()


def get_classes(namespace):
    list_class = []
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) \
                and not inspect.isabstract(v):
            list_class.append(v)
    return list_class


def get_features(classes):
    for v in tqdm(classes):
        yield v()


def generate_features(namespace, overwrite):
    list_class = get_classes(namespace)
    for f in get_features(list_class):
        f.run().save()


class Feature(metaclass=ABCMeta):

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()

    def run(self):
        self.create_features()
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        for col in self.train.columns:
            self.train[[col]].to_feather(f'../features/{col}_train.feather')
            self.test[[col]].to_feather(f'../features/{col}_test.feather')
