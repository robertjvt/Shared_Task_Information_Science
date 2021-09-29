#!/usr/bin/env python

import os
import csv
import random as python_random
import json
import argparse
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Flatten
from keras.layers.core import Dense, Activation
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam

from pprint import pprint


# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--embeddings", default='glove_filtered.json', type=str,
                        help="Embedding file we are using (default glove_filtered.json)")

    args = parser.parse_args()

    return args


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


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''

    with open(embeddings_file, 'r') as fd:
        embeddings = json.load(fd)
    return {word: np.array(embeddings[word]) for word in embeddings}


def main():
    args = create_arg_parser()
    # Read in the data
    X_full, Y_full = read_data()
    # pprint(X_full[:10])
    # pprint(Y_full[:10])

    # statis of dataset
    ones = sum([int(n) for n in Y_full])
    zeroes = len(Y_full) - ones

    print("Total of label 1: ", ones)
    print("Total of lable 0: ", zeroes)

    # Split train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.20)
    # print("Y_train: ", Y_train[:10])
    # print("Y_test: ", Y_test[:10])
    # print("Type Y_train: ", Y_train[1])
    Y_train = np.array(Y_train, dtype=np.float32)

    Y_test = np.array(Y_test, dtype=np.float32)

    # Load embeddings indexs
    embeddings_index = read_embeddings(args.embeddings)
    # pprint(embeddings_index['like'])

    # Prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(X_train)
    vocab_size = len(t.word_index) + 1

    # Interger encode the trainset
    encoded_X_train = t.texts_to_sequences(X_train)
    # pprint(encoded_X_train[:10])

    # Creating fixed length vectors using padding
    max_length = 300
    padded_X_train = pad_sequences(encoded_X_train, maxlen=max_length, padding = 'post')
    pprint(padded_X_train[:10])
    print("padded_X_train: ", padded_X_train.shape)
    print("Y_train: ", Y_train.shape)
    # pprint(Y_train[:10])

    encoded_X_test = t.texts_to_sequences(X_test)
    padded_X_test = pad_sequences(encoded_X_test, maxlen=max_length, padding = 'post')
    print("padded_X_test: ", padded_X_test.shape)
    print("Y_test: ", Y_test.shape)

    # Create a weight matrix for words in trainset
    embedding_matrix = np.zeros((vocab_size, max_length))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    # pprint(embedding_matrix)

    learning_rate = 0.02
    # SGD optimizer
    sgd = SGD(lr=learning_rate)
    batch_size = 50

    # Adam optimizer
    opt_1 = keras.optimizers.Adam(learning_rate=0.01)

    # RMSprop optimizer
    opt_2 = tf.keras.optimizers.RMSprop()

    embedding_layer = Embedding(vocab_size, max_length, weights=[embedding_matrix], input_length=max_length, trainable=False)
    embedding_dim = 30
    # Model is defined with embedding layer as the first layer,
    # followed by 2 LSTM layers, then a dense layer with 6 units
    # and a final output layer.
    model = Sequential([
            embedding_layer,
            Bidirectional(LSTM(embedding_dim, return_sequences=True)),
            Bidirectional(LSTM(embedding_dim,)),
            Dense(6, activation='sigmoid'),
            Dense(1, activation='softmax')
            ])
    model.compile(loss='binary_crossentropy',optimizer=opt_2, metrics=['accuracy'])
    model.summary()

    model.fit(padded_X_train, Y_train, epochs=32, batch_size=batch_size, validation_split=0.1)
    Y_pred = model.predict(padded_X_test)
    # Y_pred = [str(int(n)) for n in pred]
    # print(Y_pred[:10])

    # Obtain the accuracy score of the model
    acc = accuracy_score(Y_test, Y_pred)
    print("Final accuracy: {}".format(acc))

    # Obtain performance scores of the model
    print(classification_report(Y_test, Y_pred, zero_division=0))


if __name__ == '__main__':
    main()
