import json
import random
import pandas as pd
from pprint import pprint


DATA_DIR = '../../../Data/split_dataset/'
MODEL_DIR = "../Saved_Models/"
OUTPUT_DIR = "../Output/"
LOG_DIR = "../Logs/"


def get_config(location):
    """Return model name and paramters after reading it from json file"""
    try:
        # location = 'config.json'
        with open(location) as file:
            configs = json.load(file)
            vals = [str(v).upper() for v in configs.values()]
            model_name = "_".join(vals[:-1])
        return configs, model_name
    except FileNotFoundError as error:
        print(error)


def read_data(file):
    """Read in data sets and returns sentences and labels"""
    sentences = []
    labels = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            sentences.append(" ".join(tokens[:-1]))
            labels.append(int(tokens[-1]))
    return sentences, labels


def read_official_test_set(file):
    """Read in data sets and returns sentences and labels"""
    sentences = []
    labels = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line + ' 0'
            tokens = line.strip().split()
            sentences.append(" ".join(tokens[:-1]))
            labels.append(int(tokens[-1]))
    return sentences, labels


def shuffle_dependent_lists(l1, l2):
    '''Shuffle two lists, but keep the dependency between them'''
    tmp = list(zip(l1, l2))
    # Seed the random generator so results are consistent between runs
    random.Random(123).shuffle(tmp)
    new_l1, new_l2 = list(zip(*tmp))
    new_l1 = list(new_l1)
    new_l2 = list(new_l2)
    return new_l1, new_l2

