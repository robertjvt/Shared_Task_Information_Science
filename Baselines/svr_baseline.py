#!/usr/bin/env python

import os
import csv
import random
import numpy as np
import pandas as pd
import scipy.stats
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.metrics import classification_report as report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict


DATA_DIR = '../Data/split_dataset/Regression_Task/'

def read_data(file):
    """Read in data sets and returns sentences and labels"""
    sentences = []
    labels = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            sentences.append(" ".join(tokens[:-1]))
            labels.append(float(tokens[-1]))
    return sentences, labels


def shuffle_dependent_lists(l1, l2):
    """
    Description:

    Shuffle two lists, but keep the dependency between them

    Parameters:

    l1 = token lists

    l2 = labels

    Return:

    Returns a list of tokens lists and a list of labels
   """
    tmp = list(zip(l1, l2))
    # Seed the random generator so results are consistent between runs
    random.Random(123).shuffle(tmp)
    return zip(*tmp)


def load_data(dir):
    """Return appropriate training and validation sets reading from csv files"""

    X_train, Y_train = read_data(dir+'train_reg.txt')


    X_test, Y_test = read_data(dir+'test_reg.txt')

    # shuffle data
    X_train, Y_train = shuffle_dependent_lists(X_train, Y_train)
    X_test, Y_test = shuffle_dependent_lists(X_test, Y_test)

    return X_train, Y_train, X_test, Y_test


# def identity(x):
#     '''Dummy function that just returns the input'''
#     return x


def train_svr(X_train, Y_train):
    vec = TfidfVectorizer(tokenizer=word_tokenize)
    svr_classifier = Pipeline([('vec', vec), ('svr', LinearSVR())])
    svr_classifier = svr_classifier.fit(X_train, Y_train)
    return svr_classifier


def main():
    X_train, Y_train, X_test, Y_test = load_data(DATA_DIR)
    print(Y_test[:5])

    svr_classifier = train_svr(X_train, Y_train)

    # cv_pred = cross_val_predict(svr_classifier, X_full, Y_full, cv=5)
    # print('CV scores: ', cv_score)

    Y_pred = svr_classifier.predict(X_test)
    print(Y_pred[:5])
    rho = scipy.stats.spearmanr(Y_test, Y_pred)[0]
    print(rho) #rho = 0.6926822546986958

    # print(report(Y_test, Y_pred, digits=3))


if __name__ == "__main__":
    main()
