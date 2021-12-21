import os
import pandas as pd
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
import random as python_random
import utils
import argparse

import utils

from pprint import pprint

def create_arg_parser():
    ''' Creates argparses for all different models. These will be run through both
    train.py and predict.py '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', default = '', type=str)
    parser.add_argument('--model', default = '../Saved_Models/tf_model.h5', type=str)
    parser.add_argument('--file', default = '', type=str)
    parser.add_argument('--cmd', default = 'False', type=str)
    args = parser.parse_args()
    return args

def test(to_predict, config, model_name):
    """Return models prediction"""
    #set random seed to make results reproducible 
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    python_random.seed(config['seed'])

    #set model parameters 
    max_length  =  config['max_length']
   
    if config["model"].upper() =='BERT':
        lm = 'bert-base-uncased'

    #set tokenizer according to pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(lm)

    #transform raw texts into model input 
    tokens_test = tokenizer(to_predict, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    
    #get transformer text classification model based on pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained('../Saved_Models/BERT_50_0.0003_10_3_8_CATEGORY_SGD_DATA_GENERATED_FROM_OTHER_TEMPLATES')

    #get model's prediction
    Y_pred = model.predict(tokens_test, batch_size=1)["logits"]
    #print(Y_pred)

    Y_pred = np.argmax(Y_pred, axis=1)

    return Y_pred

def main():
    args = create_arg_parser()

    #get parameters for experiments
    config, model_name = utils.get_config()
    
    if args.file != '':
        to_predict = []
        with open(args.file, encoding='utf-8', errors = 'ignore') as f:
            for sentence in f:
                to_predict.append(sentence)
    elif args.t != '':
        to_predict = args.t
    elif args.cmd != 'False':
        while True:
            to_predict = input("Enter sentence (ctrl+c to quit): ")
            Y_pred = test(to_predict, config, model_name)
            print("----------")
            print("Predictions")
            print("----------")
            print(Y_pred)
            print("----------")
    else:
        print("please provide a sentence using --t or a file using --file.")

    # if config['training-set'] != 'trial':
    #     model_name = model_name+"_"+str(config['seed'])

    #load test set

    Y_pred = test(to_predict, config, model_name)

    print("----------")
    print("Predictions")
    print("----------")
    print(Y_pred)
    print("----------")
    
if __name__ == "__main__":
    main()
