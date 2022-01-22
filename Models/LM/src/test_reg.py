import logging
# get TF logger
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info
import os
import pandas as pd
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
import random as python_random
import utils
import argparse
from pathlib import Path

import sys
import utils
from pprint import pprint

DATA_DIR = '../../../Data/split_dataset/Regression_Task/'


def load_test_set(dir):

    """Return test sets reading from csv files"""
    # X_test, Y_test = utils.read_data(dir+'test.txt')
    X_test, Y_test = utils.read_data(dir+'simple_templates_dev_2.txt')

    #convert Y into one hot encoding
    Y_test = tf.one_hot(Y_test, depth=2)

    return X_test, Y_test


def load_official_test_set(dir):

    """Return test sets reading from csv files"""
    X_test, Y_test = utils.read_official_test_set(dir+'official_test_set.txt')

    #convert Y into one hot encoding
    Y_test = tf.one_hot(Y_test, depth=2)

    return X_test, Y_test


def save_output(Y_test, Y_pred, model_name):

    """save models prediction as csv file"""
    df = pd.DataFrame()
    df['Test'] = Y_test
    df['Predict'] = Y_pred

    #save output in directory
    try:
        os.mkdir(utils.OUTPUT_DIR)
        df.to_csv(utils.OUTPUT_DIR+model_name+".csv", index=False)
        
    except OSError as error:
        df.to_csv(utils.OUTPUT_DIR+model_name+".csv", index=False)
   

def test(X_test, Y_test, config, model_name):

    """Return models prediction"""

    #set random seed to make results reproducible 
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    python_random.seed(config['seed'])

    #set model parameters 
    max_length  =  config['max_length']
   
    if config["model"].upper() =='BERT_REG':
        lm = 'bert-base-uncased'

    #set tokenizer according to pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(lm)

    #transform raw texts into model input 
    tokens_test = tokenizer(X_test, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    
    #get transformer text classification model based on pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(utils.MODEL_DIR+model_name, num_labels=1)

    #get model's prediction
    Y_pred = model.predict(tokens_test, batch_size=1)["logits"]

    # probabilities = tf.math.softmax(Y_pred, axis=-1)
    probabilities = tf.nn.softmax(Y_pred, axis=-1)

    # Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
    return Y_test, probabilities

def set_log(model_name):

    #create log file
    try:
        os.mkdir(utils.LOG_DIR)
        log.setLevel(logging.INFO)

    except OSError as error:
    
        log.setLevel(logging.INFO)
    
    #create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #create file handler which logs info
    fh = logging.FileHandler(utils.LOG_DIR+model_name+".log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def main():

    #enable memory growth for a physical device so that the runtime initialization will not allocate all memory on the device
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

    try:
        config_location = sys.argv[1]
    except IndexError:
        print("config not given!")
        sys.exit(1)

    if not Path(config_location).exists():
        print(f"does not exist: {config_location}")
        sys.exit(2)

    #get parameters for experiments
    config, model_name = utils.get_config(config_location)
    
    # if config['training-set'] != 'trial':
    #     model_name = model_name+"_"+str(config['seed'])

    #set log settings
    set_log(model_name)

    # args = create_arg_parser()

    #load test set

    # X_test, Y_test = load_test_set(utils.DATA_DIR)
    X_test, Y_test = load_official_test_set(DATA_DIR)
    # print(len(X_test), len(Y_test))
    Y_test, Y_pred = test(X_test, Y_test, config, model_name)

    #save output in directory
    save_output(Y_test, Y_pred, model_name)

  
    
if __name__ == "__main__":
    main()
