"""
This script is created to test our models
"""
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



def create_arg_parser():

    '''Returns a map with commandline parameters taken from the user'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-ts",
        "--testset",
        default="24",
        type=str,
        help="define the test set. By default it uses "
             "the 24th meeting as test set. Input "
             " '25' to use the 25th meeting as test set."
        )

    args = parser.parse_args()
    return args
    

def load_data(dir, testset):

    """Return test sets reading from csv files"""
    try:
        if testset=="24":
            df_test = pd.read_csv(dir+'/test.csv')
        elif testset=="25":
            df_test = pd.read_csv(dir+'/test_25th.csv')
    except FileNotFoundError as error:
        print("#########################################################################\n")
        print("Please ensure that test.csv or test_25th.csv files are present in the train-test-dev folder\n")
        print("#########################################################################\n")
            
    """Return appropriate training and validation sets reading from csv files"""
    Ids, X_test = utils.read_data2(dir+'/official_test.txt')
    print(Ids[0:5])
    print(X_test[0:5])
    #convert Y into one hot encoding
    #Y_test = tf.one_hot(Y_test, depth=2)

    return Ids, X_test

def save_output(Y_test, Y_pred, model_name):

    """save models prediction as csv file"""


    df = pd.DataFrame()
    df['ID'] = Y_test
    df['Labels'] = Y_pred
    print("Writing output")

    #save output in directory
    try:
        os.mkdir(utils.OUTPUT_DIR)
        df.to_csv(utils.OUTPUT_DIR+model_name+".tsv", index=False, sep="\t")
        
    except OSError as error:
        df.to_csv(utils.OUTPUT_DIR+model_name+".tsv", index=False, sep="\t")
   

def test(X_test, config, model_name):

    """Return models prediction"""

    #set random seed to make results reproducible 
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    python_random.seed(config['seed'])

    #set model parameters 
    max_length  =  config['max_length']
   
    if config["model"].upper() =='LONG':
        lm = "allenai/longformer-base-4096"
    elif config["model"].upper() =='BERT':
        lm = 'bert-base-uncased'
    elif config["model"].upper() =='ERNIE':
        lm = 'nghuyong/ernie-2.0-en'
    elif config["model"].upper() =='XLNET':
        lm = 'xlnet-base-cased'
        
    #set tokenizer according to pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(lm)
    print(X_test)

    #transform raw texts into model input 
    tokens_test = tokenizer(X_test, padding=True, max_length=max_length,truncation=True, return_tensors="np").data
    
    #change the data type of model inputs to int32 
    if config["model"] =='LONG':
        tokens_test = utils.change_dtype(tokens_test)

    #get transformer text classification model based on pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(utils.MODEL_DIR+model_name+"_1234", num_labels = 2)

    #get model's prediction
    Y_pred = model.predict(tokens_test, batch_size=1)["logits"]

    Y_pred = np.argmax(Y_pred, axis=1)
    
    return X_test, Y_pred

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
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
   # if len(physical_devices) > 0:
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #get parameters for experiments
    config, model_name = utils.get_config()

    #set log settings
    set_log(model_name)

    args = create_arg_parser()

    #load data from train-test-dev folder
    Ids, X_train = load_data(utils.DATA_DIR,args.testset)
    Y_test, Y_pred = test(X_train, config, model_name)
    
    #save output in directory
    save_output(Y_test, Y_pred, model_name)
  
    

if __name__ == "__main__":
    main()

