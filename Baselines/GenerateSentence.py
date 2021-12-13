#!/usr/bin/env python
import random
import pandas as pd
import os
import csv
import argparse
import random
import nltk
import requests
import  tarfile
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
   
    parser.add_argument("-d", "--dev",action="store_true",
                        help="Create devset with overlapping templates")
    args = parser.parse_args()
    return args

def read_data():
    sentences = []
    labels = []

    test = False
    os.chdir('../Data/split_dataset')
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name == 'label_balanced_train.txt':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='.'))

                for row in text:
                    print(row)
                    tokens = row[0].strip().split()
                    tokens.append('.')
                    tokens = " ".join(tokens)
                    sentences.append(tokens)
                    labels.append(row[1])
    return sentences, labels

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
                    if "more specifically" in row[0].strip():
                        continue
                    elif "interesting type" in row[0].strip():
                        continue
                    else:
                        templates[row[0].strip()] = []
                        templates[row[0].strip()].append(int(row[1].strip()))
                        templates[row[0].strip()].append(int(row[2].strip())) 
                        templates[row[0].strip()].append(int(row[3].strip()))
                        templates[row[0].strip()].append(int(row[4].strip()))
    return templates
    
def sentence_tempalate_match(template, sentence):
    """
    Description:
    
    Check if a sentence mathes the template
    
    Return
    
    List of token not in template
    """
    templateWordList = template.split()
    sentenceWordList = sentence.split()
    for i in templateWordList:
        if i in sentenceWordList:
            sentenceWordList.remove(i)
    return sentenceWordList

def getRelation(word1, word2):
    """
    Description:
    
    Check if word1 is hypernym/hyponym of word2
    
    Return
    
    hypo if hyponym
    hyper if hypernym
    no if no hypernym/hyponym relation exists 
    """
    print(wn.synsets(word1))
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




def amplify(templates, data, labels, method = "balance_labels"):
    """
    Description:
    
    This function will amplify the given dataset (data+labels based on the method given
    
    Return
    new dataset + labels
    """
    
    
    new_data = []
    new_labels = []
    for i in templates:
        new_data_temp, new_labels_temp = getNewSentences(i, templates[i])
        new_data.extend(new_data_temp)
        new_labels.extend(new_labels_temp)
    labels.extend(new_labels)
    data.extend(new_data)
    return data, labels
                     
def applify2(templates, data, labels):
    new_data = []
    new_labels = []
    for i in templates:
        new_data_temp, new_labels_temp = getNewSentences2(i, templates[i])
        new_data.extend(new_data_temp)
        new_labels.extend(new_labels_temp)
    labels.extend(new_labels)
    data.extend(new_data)
    return data, labels

def applify3(templates, data, labels):
    new_data = []
    new_labels = []
    for i in templates:
        new_data_temp, new_labels_temp = getNewSentences3(i, templates[i])
        new_data.extend(new_data_temp)
        new_labels.extend(new_labels_temp)
    return new_data, new_labels

def getIsAdataset():
    new_data = []
    new_labels = []
    new_data.extend(getHyponymSentences(10000,'[word1] is [word2]'))
    new_labels.extend(getLabels(1,10000))
    new_data.extend(getNormalSentences(10000,'[word1] is [word2]'))
    new_labels.extend(getLabels(0,10000))
    return new_data, new_labels


def getNewSentences(template, data):
    new_sentences = []
    label = 0
    print(data)
    if data[2] >  data[1]:
        n_sentences = data[2]-data[1]  
        label = 1
        if data[3] == 1:
            new_sentences = getHypernymSentences(n_sentences, template)
        elif data[3] == 2:
            new_sentences = getHyponymSentences(n_sentences, template)
        elif data[3] == 3:
            new_sentences = getNormalSentences(n_sentences, template)
    elif data[1] >  data[2]:
        n_sentences = data[1]-data[2] 
        label = 0 
        if data[3] == 1:
            new_sentences = getNormalSentences(n_sentences, template)
        elif data[3] == 2:
            new_sentences = getHypernymSentences(n_sentences, template)
        elif data[3] == 3:
            new_sentences = getHyponymSentences(n_sentences, template)
    return new_sentences ,getLabels(label, n_sentences)   

def getNewSentences2(template, data):
    new_sentences = []
    sentences = []
    labels = []
    if data[1] > data[2]:
        n_sentences = 611 -data[1]
    elif data[1] < data[2]:
        n_sentences = 611 -data[2]
    print(n_sentences)
    label = 1
    if data[3] == 1:
        new_sentences = getHypernymSentences(n_sentences, template)
    elif data[3] == 2:
        new_sentences = getHyponymSentences(n_sentences, template)
    elif data[3] == 3:
        new_sentences = getNormalSentences(n_sentences, template)
    sentences.extend(new_sentences)
    labels.extend(getLabels(label, n_sentences))

    label = 0 
    if data[3] == 1:
        new_sentences = getNormalSentences(n_sentences, template)
    elif data[3] == 2:
        new_sentences = getHypernymSentences(n_sentences, template)
    elif data[3] == 3:
        new_sentences = getHyponymSentences(n_sentences, template)
    sentences.extend(new_sentences)
    labels.extend(getLabels(label, n_sentences))
    return sentences , labels

def getNewSentences3(template, data):
    new_sentences = []
    sentences = []
    labels = []
    n_sentences = 50

    label = 1
    if data[3] == 1:
        new_sentences = getHypernymSentences(n_sentences, template)
    elif data[3] == 2:
        new_sentences = getHyponymSentences(n_sentences, template)
    elif data[3] == 3:
        new_sentences = getNormalSentences(n_sentences, template)
    sentences.extend(new_sentences)
    labels.extend(getLabels(label, n_sentences))

    label = 0 
    if data[3] == 1:
        new_sentences = getNormalSentences(n_sentences, template)
    elif data[3] == 2:
        new_sentences = getHypernymSentences(n_sentences, template)
    elif data[3] == 3:
        new_sentences = getHyponymSentences(n_sentences, template)
    sentences.extend(new_sentences)
    labels.extend(getLabels(label, n_sentences))
    return sentences , labels


def getLabels(label, n_labels):
    labels = []
    while len(labels) != n_labels:
        labels.append(label)
    return labels




def getNormalSentences(n_sentence, template):
    """
    Description:
    
    Generate n sentences equal to the n_sentence paramater that has no words 
    with hypernym/hyponym 
    
    Return
    
    List of sentences
    """
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
    """
    Description:
    
    Generate n sentences equal to the n_sentence paramater where word1 is 
    hypernym of word2
    
    Return
    
    List of sentences
    """
    
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
    """
    Description:
    
    Generate n sentences equal to the n_sentence paramater where word1 is 
    hyponym of word2
    
    Return
    
    List of sentences
    """

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

def writetocsv(X_full, Y_full):
    output = '\n'.join('\t'.join(map(str,row)) for row in zip(X_full, Y_full))
    with open('custom_train_large.txt', 'w') as f:
        f.write(output)
    
    
def main():
    templates = read_data_template()
    X_full, Y_full = read_data()


    #new_data, new_labels = amplify(templates, X_full, Y_full)
    #new_data, new_labels = applify2(templates, X_full, Y_full)
    new_data, new_labels = getIsAdataset()
    writetocsv(new_data, new_labels)


if __name__ == "__main__":
    main()
