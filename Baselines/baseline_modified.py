#!/usr/bin/env python

import os
import csv
import argparse
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report as report
from sklearn.model_selection import cross_val_predict


def create_arg_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input_file", default='reviews.txt', type=str,
    #                     help="Input file to learn from (default reviews.txt)")
    parser.add_argument("-hy", "--hyper", action="store_true",
                        help="Identified obtimized hyper parameters")
    parser.add_argument("-c", "--C",type=float, default=1,
                        help="Iput C value")
    parser.add_argument("-g", "--gamma",type=float, default=1,
                        help="Input gamma value")
    parser.add_argument("-k", "--kernel",type=str, default='rbf',
                        help="Input kernel")
    args = parser.parse_args()
    return args


def read_data():
    sentences = []
    labels = []
    for root, dirs, files in os.walk('../Data', topdown=False):
        for name in files:
            if name[:11] == 'En-Subtask1':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='\t'))
                for row in text[1:]:
                    tokens = row[1].strip().split()
                    sentences.append(tokens)
                    labels.append(row[2])
    return sentences, labels


def shuffle_dependent_lists(l1, l2):
    '''Shuffle two lists, but keep the dependency between them'''
    tmp = list(zip(l1, l2))
    # Seed the random generator so results are consistent between runs
    random.Random(123).shuffle(tmp)
    return zip(*tmp)


def train_dummy_classifier(X_train, Y_train):
    dummy_classifier = DummyClassifier(strategy='prior')
    dummy_classifier.fit(X_train, Y_train)
    return dummy_classifier

def train_forest_classifier(X_train, Y_train):
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    print(X_train)
    forest_classifier = Pipeline([('vec', vec), ('cls', RandomForestClassifier())])
    forest_classifier = forest_classifier.fit(X_train, Y_train)

    return forest_classifier

def train_naive_bayes(X_train, Y_train):
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    naive_classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
    naive_classifier = naive_classifier.fit(X_train, Y_train)

    return naive_classifier


def train_svm(X_train, Y_train, C, gamma, kernel):
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    if kernel == 'linear':
        svm_classifier = Pipeline([('vec', vec), ('svc', SVC(kernel=kernel, C=C))])
        svm_classifier = svm_classifier.fit(X_train, Y_train)
    elif kernel == 'rbf':
        svm_classifier = Pipeline([('vec', vec), ('svc', SVC(kernel=kernel, C=C,gamma = gamma))])
        svm_classifier = svm_classifier.fit(X_train, Y_train)
    return svm_classifier


def get_optimal_hyperParmeters(kernel, X_train, Y_train, X_test, Y_test):
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    List_C = list([0.0001,0.001,0.01,0.1,10,100,1000])
    C=0
    gamma = 0
    f1_opt = 0
    if kernel == "rbf":
        List_gamma = list([0.0001,0.001,0.01,0.1,10,100,1000])
        for i in List_C:
            for x in List_gamma:
                cls = SVC(kernel='rbf', C=i, gamma = x)
                classifier = Pipeline([('vec', vec), ('cls',cls)])
                classifier.fit( X_train, Y_train)
                pred = classifier.predict(X_test)
                f1 = report(Y_test, pred, digits=3, output_dict = True, zero_division= 0).get('macro avg').get('f1-score')
                if f1 > f1_opt:
                    f1_opt = f1
                    C = i
                    gamma = x
        print(f"rbf kernel: C = {C} gamma = {gamma} f1 = {f1_opt}")
    elif kernel == "linear" :
        for i in List_C:
            cls = SVC(kernel='linear', C=i)
            classifier = Pipeline([('vec', vec), ('cls',cls)])
            classifier.fit(X_train, Y_train)
            pred = classifier.predict(X_test)
            f1 = report(Y_test, pred, digits=3, output_dict = True, zero_division = 0).get('macro avg').get('f1-score')
            if f1 > f1_opt:
                f1_opt = f1
                C = i
        print(f"linear kernel: C = {C}  f1 = {f1_opt}")


def identity(x):
    '''Dummy function that just returns the input'''
    return x


def main():
    args = create_arg_parser()
    X_full, Y_full = read_data()
    X_full, Y_full = shuffle_dependent_lists(X_full, Y_full )
    X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.2, random_state=0)
    print("-------------")
    svm_classifier = train_svm(X_train, Y_train, args.C, args.gamma, args.kernel)
    pred = cross_val_predict(svm_classifier, X_full, Y_full, cv=5)
    if args.hyper:
        get_optimal_hyperParmeters(args.kernel, X_train, Y_train, X_test, Y_test)
    print("SVM accuracy: {}".format(round(svm_classifier.score(X_test, Y_test), 3)))
    print("-------------")
    pred = svm_classifier.predict(X_test)
    print(report(Y_test, pred, digits=3))


if __name__ == "__main__":
    main()
