#!/usr/bin/env python

import os
import csv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


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


def train_svm(X_train, Y_train):
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    classifier = Pipeline([('vec', vec), ('svc', svm.SVC(random_state=0))])
    classifier = classifier.fit(X_train, Y_train)
    return classifier


def identity(x):
    '''Dummy function that just returns the input'''
    return x


def optimize_hyperparameters(classifier, X_train, Y_train):
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': [1, 0.1, 0.01],
        'svc__kernel': ['rbf']
    }
    gs_classifier = GridSearchCV(estimator=classifier,
                                 param_grid=param_grid,
                                 cv=3,
                                 scoring='f1_macro',
                                 n_jobs=1,
                                 verbose=3)
    gs_classifier.fit(X_train, Y_train)
    print("Best estimator found by grid search:\n")
    print(gs_classifier.best_estimator_)
    print("Best parameters found by the grid search and macro f1-score:\n")
    print(gs_classifier.best_params_, gs_classifier.best_score_)
    print("Results on cross validation during grid search:\n")
    print(gs_classifier.cv_results_)
    return gs_classifier.best_estimator_


def main():
    X_full, Y_full = read_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.2, random_state=0)
    classifier = train_svm(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    print(classification_report(Y_test, Y_pred))

    gs_classifier = optimize_hyperparameters(classifier, X_train, Y_train)
    Y_pred = gs_classifier.predict(X_test)
    print(classification_report(Y_test, Y_pred))


if __name__ == "__main__":
    main()