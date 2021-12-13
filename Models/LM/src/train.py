#!/usr/bin/env python

'''Train BERT-based model on different train sets,
trained models will be saved to test and evaluate.'''

import logging
# get TF logger for pre-trained transformer model
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info

import random as python_random
import numpy as np
import os
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import tensorflow as tf
from tqdm.keras import TqdmCallback

from pprint import pprint

import utils


def load_data(dir, config):
    """Return appropriate training and validation sets reading from csv files"""
    training_set = config["training-set"]

    if training_set.lower() == "train":
        X_train, Y_train = utils.read_data(dir+'train.txt')

    elif training_set.lower() == "label_balanced_train":
        X_train, Y_train = utils.read_data(dir+'label_balanced_train.txt')

    elif training_set.lower() == "data_generated_from_other_templates":
        X_train, Y_train = utils.read_data(dir+'data_generated_from_other_templates.txt')

    X_dev, Y_dev = utils.read_data(dir+'dev_shared_template.txt')

    return X_train, Y_train, X_dev, Y_dev


def set_log(model_name):

    #Create Log file to save info
    try:
        os.mkdir(utils.LOG_DIR)
        log.setLevel(logging.INFO)

    except OSError as error:
        log.setLevel(logging.INFO)


    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs info
    fh = logging.FileHandler(utils.LOG_DIR+model_name+".log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def classifier(X_train, X_dev, Y_train, Y_dev, config, model_name):

    """Train and Save model for test and evaluation"""

    # set random seed to make results reproducible
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    python_random.seed(config['seed'])

    # set model parameters
    max_length  =  config['max_length']
    learning_rate =  config["learning_rate"]
    epochs = config["epochs"]
    patience = config["patience"]
    batch_size = config["batch_size"]

    if config["loss"].upper() == "BINARY":
        loss_function = BinaryCrossentropy(from_logits=True)
    elif config["loss"].upper() == "CATEGORY":
        loss_function = CategoricalCrossentropy(from_logits=True)

    if config['optimizer'].upper() == "ADAM":
        optim = Adam(learning_rate=learning_rate)
    elif config['optimizer'].upper() == "SGD":
        optim = SGD(learning_rate=learning_rate)

    if config["model"].upper() =='BERT':
        lm = 'bert-base-uncased'
    # enable if trying with other pre-trained model.
    # elif config["model"].upper() ==' ':
    #     lm = ''

    # set tokenizer according to pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(lm)

    # get transformer text classification model based on pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)

    # transform raw texts into model input
    tokens_train = tokenizer(X_train, padding=True, max_length=max_length,
        truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_length,
        truncation=True, return_tensors="np").data

    pprint(f'tokens_train: {tokens_train}')
    pprint(f'tokens_train: {tokens_train.keys()}')

    #convert Y into one hot encoding
    Y_train = tf.one_hot(Y_train,depth=2)
    Y_dev = tf.one_hot(Y_dev,depth=2)

    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    # callbacks for ealry stopping and saving model history
    es = EarlyStopping(monitor="val_loss", patience=patience,
        restore_best_weights=True, mode='max')
    history_logger = CSVLogger(utils.LOG_DIR+model_name+"_HISTORY.csv",
        separator=",", append=True)

    # train models
    model.fit(tokens_train, Y_train, verbose=0, epochs=epochs,
        batch_size= batch_size, validation_data=(tokens_dev, Y_dev),
        callbacks=[es, history_logger, TqdmCallback(verbose=2)])

    # save models in directory
    model.save_pretrained(save_directory=utils.MODEL_DIR+model_name)


def main():

    # enable memory growth for a physical device so that the
    # runtime initialization will not allocate all memory on the device
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # get parameters for experiments
    config, model_name = utils.get_config()

    set_log(model_name)

    # load data from train-test-dev folder
    X_train, Y_train, X_dev, Y_dev = load_data(utils.DATA_DIR, config)

    # print(f'X_train: {X_train}')
    # print(f'X_dev: {X_dev}')

    # print(f'Y_dev: {Y_dev}')
    # print(f'Y_train: {Y_train}')

    # return

    # run model
    classifier(X_train, X_dev, Y_train, Y_dev, config, model_name)


if __name__ == '__main__':
    main()
