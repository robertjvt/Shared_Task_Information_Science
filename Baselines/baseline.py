#!/usr/bin/env python

import os
import csv
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='reviews.txt', type=str,
                        help="Input file to learn from (default reviews.txt)")
    args = parser.parse_args()
    return args

def read_data():
    sentences = []
    labels = []
    os.chdir('../Data')
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name[:11] == 'En-Subtask1':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='\t'))
                for row in text[1:]:
                    sentences.append(row[1])
                    labels.append(row[2])
    return sentences, labels


def train_dummy_classifier(X_train, Y_train):
    dummy_classifier = DummyClassifier(strategy='prior')
    dummy_classifier.fit(X_train, Y_train)
    return dummy_classifier

def train_forest_classifier(X_train, Y_train):
    vec = TfidfVectorizer()
    forest_classifier = Pipeline([('vec', vec), ('cls', RandomForestClassifier())])
    forest_classifier = forest_classifier.fit(X_train, Y_train)

    return forest_classifier

def train_naive_bayes(X_train, Y_train):
    vec = TfidfVectorizer()
    naive_classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
    naive_classifier = naive_classifier.fit(X_train, Y_train)

    return naive_classifier

def train_svm(X_train, Y_train):
    vec = TfidfVectorizer()
    svm_classifier = Pipeline([('vec', vec), ('svc', SVC())])
    svm_classifier = svm_classifier.fit(X_train, Y_train)

    return svm_classifier

def main():
    args = create_arg_parser()
    X_full, Y_full = read_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.2, random_state=0)

    print("-------------")
    dummy_classifier = train_dummy_classifier(X_train, Y_train)
    print("Dummy classifier accuracy: {}".format(round(dummy_classifier.score(X_test, Y_test), 3)))

    forest_classifier = train_forest_classifier(X_train, Y_train)
    print("Random Forest accuracy: {}".format(round(forest_classifier.score(X_test, Y_test), 3)))

    naive_classifier = train_naive_bayes(X_train, Y_train)
    print("Naive Bayes accuracy: {}".format(round(naive_classifier.score(X_test, Y_test), 3)))

    svm_classifier = train_svm(X_train, Y_train)
    print("SVM accuracy: {}".format(round(svm_classifier.score(X_test, Y_test), 3)))
    print("-------------")

if __name__ == "__main__":
    main()
