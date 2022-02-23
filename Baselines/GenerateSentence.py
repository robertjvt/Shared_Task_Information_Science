"""
This script contains several methods for synthesizing additional data for the machine learning task
"""

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
                        help="Create devset for implementation3")
    parser.add_argument("-i1", "--implementation1",action="store_true",
                        help="Create training set using implementation1")
    parser.add_argument("-i3", "--implementation3",action="store_true",
                        help="Create training set using implementation3")
    args = parser.parse_args()
    return args

def read_data_tab(dataset):
    '''This method reads a tab seperated text file containing documents and labels'''
    sentences = []
    labels = []

    test = False
    os.chdir('../Data/split_dataset')
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name == f'{dataset}.txt':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='\t'))

                for row in text:
                    tokens = row[0].strip().split()
                    tokens.append('.')
                    tokens = " ".join(tokens)
                    sentences.append(tokens)
                    labels.append(row[1])
    return sentences, labels

def read_data_point(dataset):
    '''This method reads a point seperated text file containing documents and labels'''
    sentences = []
    labels = []

    test = False
    os.chdir('../Data/split_dataset')
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name == f'{dataset}.txt':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='.'))

                for row in text:
                    tokens = row[0].strip().split()
                    tokens.append('.')
                    tokens = " ".join(tokens)
                    sentences.append(tokens)
                    labels.append(row[1])
    return sentences, labels

def read_data_template():
    '''This method reads the the text file containing the templates and return a dict containing the template and template support'''
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
    """
    Description:
    
    This function obtains a random hyponym of word1 and insert them to the given template to create a new sentence.
    The substring [word1] (from the template) is replaced with the word1 value and the substring [word2] 
    is replaced with its hyponym 
    
    Return
    
    String containing the newly created sentence
    """
    templateWordList = template.split()
    word1_h = wn.synsets(word1)[0]
    word2 = [i for i in word1_h.closure(lambda s:s.hyponyms())][0]
    word2 = word2.lemma_names()[0]
    templateWordList = [word1 if x=="[word1]" else x for x in templateWordList]
    templateWordList = [word2 if x=="[word2]" else x for x in templateWordList]

    return " ".join(templateWordList)




def upsample(templates, data, labels):
    """
    Description:
    
    This function will upsample the given dataset (data+labels)
    
    Return
    new dataset + labels
    """
    
    new_data = []
    new_labels = []
    for i in templates:
        new_data_temp, new_labels_temp = balanceTemplates(i, templates[i])
        new_data.extend(new_data_temp)
        new_labels.extend(new_labels_temp)
    labels.extend(new_labels)
    data.extend(new_data)
    return data, labels



def balanceTemplates(template, data):
    '''
    Balance the templates by generated new data based on label skewness per template
    '''
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



def getCustomTemplateDataset():
    """
    Create a custom training set using implimentation 3
    """
    new_date = []
    new_labels = []
    new_date.extend(getNormalSentences(200, "[word1] such as [word2]")) 
    new_date.extend(getNormalSentences(200, "[word1] was a [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] are [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] were [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] are examples of [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] is an example of [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] is a kind of [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] an interesting type of [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] are kinds of [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] is a form of [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] are forms of [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] includes [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] include  [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] is type of [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] and more specifically [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] except [word2]"))
    new_date.extend(getHypernymSentences(200, "[word1] , I prefer [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] , I prefer [word2]"))
    new_date.extend(getHyponymSentences(400, "[word1] , but not [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] , more than [word2]"))
    new_date.extend(getHypernymSentences(200, "[word1] , more than [word2]"))
    new_labels.extend(getLabels(0,4400))
    
    new_date.extend(getHyponymSentences(200, "[word1] such as [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] was a [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] are [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] were [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] are examples of [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] is an example of [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] is a kind of [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] an interesting type of [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] are kinds of [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] is a form of [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] are forms of [word2]"))
    new_date.extend(getHypernymSentences(200, "[word1] includes [word2]"))
    new_date.extend(getHypernymSentences(200, "[word1] include  [word2]"))
    new_date.extend(getHyponymSentences(200, "[word1] is type of [word2]"))
    new_date.extend(getHypernymSentences(200, "[word1] and more specifically [word2]"))
    new_date.extend(getHypernymSentences(200, "[word1] except [word2]"))
    new_date.extend(getNormalSentences(400, "[word1] , I prefer [word2]"))
    new_date.extend(getHypernymSentences(200, "[word1] , but not [word2]"))
    new_date.extend(getNormalSentences(200, "[word1] , but not [word2]"))
    new_date.extend(getNormalSentences(400, "[word1] , more than [word2]"))
    new_labels.extend(getLabels(1,4400))

    print(new_date)
    return new_date, new_labels


def getCustomTemplateDataset_dev():
    """
    Create custom dev set using implimentation 3
    """
    new_date = []
    new_labels = []
    new_date.extend(getNormalSentences(40, "[word1] such as [word2]")) 
    new_date.extend(getNormalSentences(40, "[word1] was a [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] are [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] were [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] are examples of [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] is an example of [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] is a kind of [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] an interesting type of [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] are kinds of [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] is a form of [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] are forms of [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] includes [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] include  [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] is type of [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] and more specifically [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] except [word2]"))
    new_date.extend(getHypernymSentences(40, "[word1] , I prefer [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] , I prefer [word2]"))
    new_date.extend(getHyponymSentences(80, "[word1] , but not [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] , more than [word2]"))
    new_date.extend(getHypernymSentences(40, "[word1] , more than [word2]"))
    new_labels.extend(getLabels(0,880))
    
    new_date.extend(getHyponymSentences(40, "[word1] such as [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] was a [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] are [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] were [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] are examples of [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] is an example of [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] is a kind of [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] an interesting type of [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] are kinds of [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] is a form of [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] are forms of [word2]"))
    new_date.extend(getHypernymSentences(40, "[word1] includes [word2]"))
    new_date.extend(getHypernymSentences(40, "[word1] include  [word2]"))
    new_date.extend(getHyponymSentences(40, "[word1] is type of [word2]"))
    new_date.extend(getHypernymSentences(40, "[word1] and more specifically [word2]"))
    new_date.extend(getHypernymSentences(40, "[word1] except [word2]"))
    new_date.extend(getNormalSentences(80, "[word1] , I prefer [word2]"))
    new_date.extend(getHypernymSentences(40, "[word1] , but not [word2]"))
    new_date.extend(getNormalSentences(40, "[word1] , but not [word2]"))
    new_date.extend(getNormalSentences(80, "[word1] , more than [word2]"))
    new_labels.extend(getLabels(1,880))

    print(new_date)
    return new_date, new_labels




def getAllNouns(templates,data):
    '''Get all nouns from data'''
    words = []
    for i in data:
        for x in templates:
            if len(sentence_tempalate_match(x, i)) == 2:
                word_pair = sentence_tempalate_match(x, i)
                for j in word_pair:
                    if j not in words:
                        words.append(j)
    return words

def getLabels(label, n_labels):
    '''This method creates copies of the "label" variable based on "n_labels"'''
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
        # check if there is no taxonomic relation between words
        if word1 != word2 and getRelation(word1, word2) == "no":
                templateWordList = [word1.replace("_"," ") if x=="[word1]" else x for x in templateWordList]
                templateWordList = [word2.replace("_"," ") if x=="[word2]" else x for x in templateWordList]
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
            templateWordList = [word1.replace("_"," ") if x=="[word1]" else x for x in templateWordList]
            templateWordList = [word2.replace("_"," ") if x=="[word2]" else x for x in templateWordList]
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
            templateWordList = [word2.replace("_"," ") if x=="[word1]" else x for x in templateWordList]
            templateWordList = [word1.replace("_"," ") if x=="[word2]" else x for x in templateWordList]
            sentences.append(" ".join(templateWordList))
    return sentences

def writetocsv(X_full, Y_full, dataset):
    output = '\n'.join('\t'.join(map(str,row)) for row in zip(X_full, Y_full))
    with open(f'{dataset}.txt', 'w') as f:
        f.write(output)
    
    
def main():
    templates = read_data_template()
    args = create_arg_parser();
    if args.implementation1:
        X_full, Y_full = read_data_point("train")
        new_data, new_labels = upsample(templates, X_full, Y_full)
        writetocsv(new_data, new_labels,"label_template_balanced_train")
    elif args.implementation3:
        new_data, new_labels = getCustomTemplateDataset()
        X_full, Y_full = read_data_point("train")

        new_data.extend(X_full)
        new_labels.extend(Y_full)
        writetocsv(new_data, new_labels,"simple_templates_train_2")
    elif args.dev:
        X_full, Y_full = read_data("simple_templates_train_2.txt")
        new_data, new_labels = getCustomTemplateDataset_dev(templates, X_full)
        for i,j in zip(new_data, new_labels) :
            if i in X_full:
                new_data.remove(i)
                new_labels.remove(j)
                writetocsv(new_data, new_labels)
    
    
    


if __name__ == "__main__":
    main()

