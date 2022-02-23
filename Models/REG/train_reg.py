'''This script creates our regression model '''
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
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.losses import MeanSquaredError
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm.keras import TqdmCallback
import csv
from sklearn.preprocessing import LabelBinarizer
import utils
from pprint import pprint

    
def load_data(dir, config):
    """Return appropriate training and validation sets reading from csv files"""
    training_set = config["training-set"]

    if training_set.lower() == "reg":
        X_train, Y_train = utils.read_data_reg(dir+'/Regression_Task/train_reg.txt')

    elif training_set.lower() == "balanced_train+other_templates":
        X_train, Y_train = utils.read_data(dir+'/balanced_train+other_templates_reg.txt')
    elif training_set.lower() == "label_template_balanced_train":
        X_train, Y_train = utils.read_data(dir+'/label_template_balanced_train.txt')
    elif training_set.lower() == "custom":
        X_train, Y_train = utils.read_data(dir+'/Regression_Task/simple_templates_train_2.txt')
    elif training_set.lower() == "custom_train_more_template":
        X_train, Y_train = utils.read_data(dir+'/custom_train_more_template.txt')
    elif training_set.lower() == "data_generated_from_other_templates":
        X_train, Y_train = utils.read_data(dir+'/data_generated_from_other_templates.txt')

    X_dev, Y_dev = utils.read_data_reg(dir+'/Regression_Task/dev_reg.txt')

    return X_train, Y_train, X_dev, Y_dev


def save_dataset(X, Y, model_name):

    """Save models prediction as csv file"""
    os.chdir('../Data/split_dataset')
    df = pd.DataFrame()
    df['Document'] = X
    df['Label'] = Y

    #save output in directory
    try:
        df.to_csv(model_name+".csv", index=False)
        
    except OSError as error:
        df.to_csv(model_name+".csv", index=False)
    os.chdir('../../LM')
   

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

    if config["loss"].upper() == "MSE":
        loss_function =  MeanSquaredError()

    if config['optimizer'].upper() == "ADAM":
        optim = Adam(learning_rate=learning_rate)
    elif config['optimizer'].upper() == "SGD":
        optim = SGD(learning_rate=learning_rate)

    if config["model"].upper() =='BERT':
        lm = 'bert-base-uncased'
    elif config["model"].upper() =='XLNET':
        lm = 'xlnet-base-cased'
    elif config["model"].upper() =='ERNIE':
        lm = 'nghuyong/ernie-2.0-en'
        

    # set tokenizer according to pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(lm)

    # get transformer text classification model based on pre-trained model
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=1)

    # transform raw texts into model input
    tokens_train = tokenizer(X_train, padding=True, max_length=max_length,
        truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_length,
        truncation=True, return_tensors="np").data

    pprint(f'tokens_train: {tokens_train}')
    pprint(f'tokens_train: {tokens_train.keys()}')

    Y_train = tf.cast(Y_train, tf.float32)

    Y_dev = tf.cast(Y_dev, tf.float32)

    print(Y_train)


    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    # callbacks for ealry stopping and saving model history
    es = EarlyStopping(monitor="val_loss", patience=patience,
        restore_best_weights=True, mode='max')
    history_logger = CSVLogger(utils.LOG_DIR+model_name+"_HISTORY.csv",
        separator=",", append=True)

    # train models
    model.fit(tokens_train, Y_train, verbose=1, epochs=epochs,
        batch_size= batch_size, validation_data=(tokens_dev, Y_dev),
        callbacks=[ history_logger, TqdmCallback(verbose=2)])


    # save models in directory
    model.save_pretrained(save_directory=utils.MODEL_DIR+model_name)


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


def main():

    #enable memory growth for a physical device so that the runtime initialization will not allocate all memory on the device 
    physical_devices = tf.config.experimental.list_physical_devices('GPU')


    #get parameters for experiments
    config, model_name = utils.get_config()
    
    if config['training-set'] != 'trial':
        model_name = model_name+"_"+str(config['seed'])

    set_log(model_name)
    

    X_train, Y_train, X_dev, Y_dev = load_data(utils.DATA_DIR, config)
   
    #run model

    classifier(X_train,X_dev,Y_train, Y_dev, config, model_name)

  

if __name__ == "__main__":
    main()

