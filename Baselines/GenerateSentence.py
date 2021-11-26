#!/usr/bin/env python
import random
import pandas as pd
import os
import csv
import argparse
import random
import torch
import torchtext
import nltk
import requests
import  tarfile
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report as report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
import numpy as np  
import random
#nltk.download()
from nltk.corpus import wordnet as wn



"""
1 = hyponyms - hypernyms
2 = hypernyms - hyponyms
3 = no hypernyms
"""

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


def read_data_with_acc_split():
    sentences = []
    labels = []
    sentences1 = []
    labels1 = []
    test = False
    os.chdir('../Data')
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name[:11] == 'En-Subtask1':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='\t'))

                for row in text[1:]:
                    
                    if 'and more specifically' in row[1]:
                        tokens = row[1].strip().split()
                        tokens =  [word for word in tokens if not word in wwords]
                        sentences1.append(tokens)
                        labels1.append(row[2])
                    elif 'I prefer' in row[1]:
                        tokens = row[1].strip().split()
                        tokens =  [word for word in tokens if not word in wwords]
                        sentences1.append(tokens)
                        labels1.append(row[2])
                        
                    else:
                        tokens = row[1].strip().split()
                        tokens =  [word for word in tokens if not word in wwords]
                        sentences.append(tokens)
                        labels.append(row[2])
    return sentences, sentences1, labels, labels1

def read_data_template():
    templates = {}
    metadata = []
    os.chdir('../Data')
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name == 'templates.txt':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='&'))
                for row in text:
                    templates[row[0].strip()] = []
                    templates[row[0].strip()].append(int(row[1].strip()))
                    templates[row[0].strip()].append(int(row[2].strip())) 
                    templates[row[0].strip()].append(int(row[3].strip()))
    return templates
    
def sentence_tempalate_match(template, sentence):
    templateWordList = template.split()
    sentenceWordList = sentence.split()
    for i in templateWordList:
        if i in sentenceWordList:
            sentenceWordList.remove(i)
    return sentenceWordList

def getRelation(word1, word2):
    word1_h = wn.synsets(word1)[0]
    word2_h = wn.synsets(word2)[0]
    hypo_word1_h = set([i for i in word1_h.closure(lambda s:s.hyponyms())])
    hypo_word2_h = set([i for i in word2_h.closure(lambda s:s.hyponyms())])
    if word2_h in hypo_word1_h:
        return "hypo"
    elif word1_h in hypo_word2_h:
        return "hyper"
    else:
        return "no"

def getNewSentences(template, word1):
    templateWordList = template.split()
    word1_h = wn.synsets(word1)[0]
    word2 = [i for i in word1_h.closure(lambda s:s.hyponyms())][0]
    word2 = word2.lemma_names()[0]
    templateWordList = [word1 if x=="[word1]" else x for x in templateWordList]
    templateWordList = [word2 if x=="[word2]" else x for x in templateWordList]

    return " ".join(templateWordList)




def applify(templates, data, labels):
    for i in templates:
        for x,y in data, labels:
            if len(sentence_tempalate_match(i,x)) == 2:
                hypernyms = sentence_tempalate_match(i,x)
                if label == 1:
                    if getRelation(hypernyms[0], hypernyms[1]) == "hyper":
                        print()
                    elif getRelation(hypernyms[0], hypernyms[1]) == "hypo":
                        print()
                    else:
                        print()
                else:
                    if getRelation(hypernyms[0], hypernyms[1]) == "hyper":
                        print()
                    elif getRelation(hypernyms[0], hypernyms[1]) == "hypo":
                        print()
                    else:
                        print()

def getNormalSentences(n_sentence, template):
    nouns = list(wn.all_synsets('n'))
    sentences = []
    while len(sentences) < n_sentence:
        templateWordList = template.split()
        word1 = random.choice(nouns).lemma_names()[0]
        word2 = random.choice(nouns).lemma_names()[0]
        if word1 != word2 and getRelation(word1, word2) == "no":
                templateWordList = [word1 if x=="[word1]" else x for x in templateWordList]
                templateWordList = [word2 if x=="[word2]" else x for x in templateWordList]
                sentences.append(" ".join(templateWordList))
    return sentences

def getHypernymSentences(n_sentence, template):
    nouns = list(wn.all_synsets('n'))
    sentences = []
    while len(sentences) < n_sentence:
        templateWordList = template.split()
        word1 = random.choice(nouns).lemma_names()[0]
        word1_h = wn.synsets(word1)[0]
        word2 = [i for i in word1_h.closure(lambda s:s.hyponyms())]
        if len(word2) > 0:
            word2 = word2[0].lemma_names()[0]
            templateWordList = [word1 if x=="[word1]" else x for x in templateWordList]
            templateWordList = [word2 if x=="[word2]" else x for x in templateWordList]
            sentences.append(" ".join(templateWordList))
    return sentences

def getHyponymSentences(n_sentence, template):
    nouns = list(wn.all_synsets('n'))
    sentences = []
    while len(sentences) < n_sentence:
        templateWordList = template.split()
        word1 = random.choice(nouns).lemma_names()[0]
        word1_h = wn.synsets(word1)[0]
        word2 = [i for i in word1_h.closure(lambda s:s.hyponyms())]
        if len(word2) > 0:
            word2 = word2[0].lemma_names()[0]
            templateWordList = [word2 if x=="[word1]" else x for x in templateWordList]
            templateWordList = [word1 if x=="[word2]" else x for x in templateWordList]
            sentences.append(" ".join(templateWordList))
    return sentences

def main():
    templates = read_data_template()
    sentence_tempalate_match("I like [word] , but not [word] .","I like ham , but not fish .")
    print(getRelation("apple","fruit"))
    #print(getNewSentences("I like [word1] , but not [word2] .", "dog"))
    print(getNormalSentences(3, "I like [word1] , but not [word2] ."))
    print(getHypernymSentences(3, "I like [word1] , and more specifically [word2] ."))
    print(getHyponymSentences(3, "I like [word1] , an interesting type of [word2] ."))


if __name__ == "__main__":
    main()
