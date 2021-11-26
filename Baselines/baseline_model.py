"""
This script can run multiple models.

When running this script without providing a training set, it will use the datasets located 
in the Data folder as training set

By default this script will run an optimized RBF kernel SVM with C == 100 and gamma = 0.1

Please enter -h to get an overview of the functions of the parameters
"""
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict


def create_arg_parser():
    """
    Description:
    
    This method is an arg parser
    
    Return
    
    This method returns a map with commandline parameters taken from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str,
                         help="Input file to learn from")
    parser.add_argument("-ts", "--test_file", type=str,
                         help="Input file to ltest model")
    parser.add_argument("-hy", "--hyper", action="store_true",
                        help="Identified obtimized hyper parameters")
    parser.add_argument("-c", "--C",type=float, default=1,
                        help="Iput C value")
    parser.add_argument("-g", "--gamma",type=float, default=1,
                        help="Input gamma value")
    parser.add_argument("-k", "--kernel",type=str, default='rbf',
                        help="Input kernel")
    parser.add_argument("-d", "--dummy",action="store_true",
                        help="Input kernel")
    parser.add_argument("-r", "--randomforest",action="store_true",
                        help="Input kernel")
    parser.add_argument("-n", "--naivebayes",action="store_true",
                        help="Input kernel")
    args = parser.parse_args()
    return args


def read_data():
    """
   Description:
    
   This method reads the En-TrainingSet-SemEval-Subtask1 dataset located in the Data folder.

   Following this, the method also extracts the labels and tokinize the sentences
   
   Return:
   
   This method returns a collection of tokenised strings (list of words) and a collection of labels """
    
    sentences = []
    labels = []
    for root, dirs, files in os.walk('../Data', topdown=False):
        for name in files:
            if name[:11] == 'En-Subtask1':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='\t'))
                #Remove rowId
                for row in text[1:]:
                    tokens = row[1].strip().split()
                    sentences.append(tokens)
                    labels.append(row[2])
    return sentences, labels


def read_corpus(corpus_file):
    """
    Description:
    
    Read a corpus_file
    
    Return:
    
    Returns a list of tokenized texts and a list of labels
   """
    sentences = []
    labels = []
    file = open(corpus_file)
    text = list(csv.reader(file, delimiter='\t'))
    #Remove rowId
    for row in text[1:]:
        tokens = row[1].strip().split()
        sentences.append(tokens)
        labels.append(row[2])

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


def train_dummy_classifier(X_train, Y_train):
    """
    Description:
    
    This method vectorize the tokes using TF-IDF and creates a Dummy model
    
    Parameters:
    
    X_train = token lists
    
    Y_train = labels
    
    Return:
    
    Returns a fitted Dummy model
    """ 
    dummy_classifier = DummyClassifier(strategy='prior')
    dummy_classifier.fit(X_train, Y_train)
    return dummy_classifier

def train_forest_classifier(X_train, Y_train):
    """
    Description:
    
    This method vectorize the tokes using TF-IDF and creates a RandomForest model
    
    Parameters:
    
    X_train = token lists
    
    Y_train = labels
    
    Return:
    
    Returns a fitted RandomForest model
    """ 

    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    forest_classifier = Pipeline([('vec', vec), ('cls', RandomForestClassifier())])
    forest_classifier = forest_classifier.fit(X_train, Y_train)
    return forest_classifier

def train_naive_bayes(X_train, Y_train):
    """
    Description:
    
    This method vectorize the tokes using TF-IDF and creates a Naive Based model
    
    Parameters:
    
    X_train = token lists
    
    Y_train = labels
    
    Return:
    
    Returns a fitted Naive Based model
    """ 
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    naive_classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
    naive_classifier = naive_classifier.fit(X_train, Y_train)
    return naive_classifier


def train_svm(X_train, Y_train, C, gamma, kernel):
    """
    Description:
    
    This method vectorize the tokes using TF-IDF and trains a SVM model
    
    Parameters:
    
    C = Regularization Parameter
    
    gamma = Curvature in decision boundary
    
    kernel = Weighing factor
    
    Return:
    
    Returns a fitted SVM model
    """ 
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    if kernel == 'linear':
        svm_classifier = Pipeline([('vec', vec), ('svc', SVC(kernel=kernel, C=C))])
        svm_classifier = svm_classifier.fit(X_train, Y_train)
    elif kernel == 'rbf':
        svm_classifier = Pipeline([('vec', vec), ('svc', SVC(kernel=kernel, C=C,gamma = gamma))])
        svm_classifier = svm_classifier.fit(X_train, Y_train)
    return svm_classifier


def get_optimal_hyperParmeters(kernel, X_train, Y_train, X_test, Y_test):
    """
    Description:
    
    This method vectorize the tokes using TF-IDF and trains several SVM models
     using different hyperparameter values to find the best performing model
    
    Parameters:
    
    X_train = token lists for training
    
    Y_train = labels for training set
    
    X_test = token lists for testing/validating
    
    Y_test = labels for test/validation set
    
    """ 
     
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    #For each model the C value increase by an order of magnitude
    List_C = list([0.0001,0.001,0.01,0.1,10,100,1000])
    C=0
    gamma = 0
    f1_opt = 0
    if kernel == "rbf":
        #For each model the gamma value increase by an order of magnitude
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
            #Linear kernel doesn't need gamma
            cls = SVC(kernel='linear', C=i)
            classifier = Pipeline([('vec', vec), ('cls',cls)])
            classifier.fit(X_train, Y_train)
            pred = classifier.predict(X_test)
            f1 = report(Y_test, pred, digits=3, output_dict = True, zero_division = 0).get('macro avg').get('f1-score')
            #If F1 of model is higher than the maximum F1 of previous models, set new maximum F1
            if f1 > f1_opt:
                f1_opt = f1
                C = i
        print(f"linear kernel: C = {C}  f1 = {f1_opt}")


def identity(x):
    """
    Description:
    
    This method just returns the input
    
    Parameters:
    
    x = token
    
    Return:
    
    Returns x
    """ 
    return x


def main():
    args = create_arg_parser()
    #Check if test set and dev set are provided
    if args.input_file!=None:
        X_full, Y_full = read_corpus(args.input_file)
    else:
        X_full, Y_full = read_data()
    
    X_full, Y_full = shuffle_dependent_lists(X_full, Y_full )
   
    if args.test_file!=None:
        X_test, Y_test = read_corpus(args.test_file)
        X_train, Y_train = X_full, Y_full
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.2, random_state=0)
    print("-------------")
    # Check which model was chosen
    if args.dummy:
        classifier = train_dummy_classifier(X_train, Y_train)
    elif args.randomforest:
        classifier = train_forest_classifier(X_train, Y_train)
    elif args.naivebayes:
        classifier = train_naive_bayes(X_train, Y_train)
    else:
        classifier = train_svm(X_train, Y_train, args.C, args.gamma, args.kernel)
    
    #Run hyperparameter tuning experiment
    if args.hyper:
        get_optimal_hyperParmeters(args.kernel, X_train, Y_train, X_test, Y_test)
    print("Accuracy: {}".format(round(classifier.score(X_test, Y_test), 3)))
    print("-------------")
    pred = classifier.predict(X_test)
    print(report(Y_test, pred, digits=3))


if __name__ == "__main__":
    main()
