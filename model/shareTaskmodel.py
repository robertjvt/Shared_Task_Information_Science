#!/usr/bin/env python

import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# import torch
from transformers import DistilBertModel, DistilBertTokenizer
import warnings
warnings.filterwarnings('ignore')

from pprint import pprint


def read_data():
    sentences = []
    labels = []
    for root, dirs, files in os.walk('../Data', topdown=False):
        for name in files:
            if name[:11] == 'En-Subtask1':
                with open(os.path.join(root, name)) as fd:
                    text = list(csv.reader(fd, delimiter='\t'))
                for row in text[1:]:
                    sentences.append(row[1])
                    labels.append(row[2])
    return sentences, labels


def main():
    # Read in the data
    X_full, Y_full = read_data()
    pprint(X_full[:3])
    model, tokenizer, pretrained_weights =
            (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
    # Load pretrained model/tokenizer
    tokenizer = tokenizer.from_pretrained(pretrained_weights)
    model = model.from_pretrained(pretrained_weights)

    tokenized_X_full = tokenizer(X_full, padding=True, max_length=100,
                    truncation=True, return_tensors="np").data


main()
