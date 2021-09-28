#!/usr/bin/env python

import os
import csv

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier


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


def train_classifier(X_train, Y_train):
    classifier = DummyClassifier(strategy='prior')
    classifier.fit(X_train, Y_train)
    return classifier

def main():
    X_full, Y_full = read_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.2, random_state=0)
    classifier = train_classifier(X_train, Y_train)
    print(round(classifier.score(X_test, Y_test), 3))

if __name__ == "__main__":
    main()