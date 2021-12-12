import logging

# get TF logger for pre-trained transformer model
# envornment = tf_m1  conda activate tf_m1
log = logging.getLogger('transformers')
log.setLevel(logging.INFO)
print = log.info

import random as python_random
import numpy as np
import os
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.losses import BinaryCrossentropy
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


def load_data1(dir, config):

    """Return appropriate training and validation sets reading from csv files"""

    training_set = config["training-set"]

    if training_set.lower()=="balanced":
        df_train = pd.read_csv(dir+'/label_balanced_train.csv')
    elif training_set.lower()=="resample":
        df_train = pd.read_csv(dir+'/train_aug.csv')
    elif training_set.lower()=="resample-balance":

        df_train = pd.read_csv(dir+'/train_down.csv')   
        if config["model"].upper() =='LONG':
            df_train = df_train[:-1] #remove one sample to fix longformer's incompatibility with shapes
    else:
        df_train = pd.read_csv(dir+'/train.csv')
    
    print(df_train)
    X_train = df_train["Document"].ravel().tolist()
    Y_train = df_train["Label"]

    df_dev = pd.read_csv(dir+'/dev_shared_template.csv')

    X_dev = df_dev['Document'].ravel().tolist()
    Y_dev = df_dev['Label']

    #convert Y into one hot encoding
    Y_train = tf.one_hot(Y_train,depth=2)
    Y_dev = tf.one_hot(Y_dev,depth=2)
    
    return X_train, Y_train, X_dev, Y_dev
    
def load_data(dir, config):
    """Return appropriate training and validation sets reading from csv files"""
    training_set = config["training-set"]

    if training_set.lower() == "train":
        X_train, Y_train = utils.read_data(dir+'/train.txt')

    elif training_set.lower() == "label_balanced_train":
        X_train, Y_train = utils.read_data(dir+'/label_balanced_train.txt')

    X_dev, Y_dev = utils.read_data(dir+'/dev_shared_template.txt')

    return X_train, Y_train, X_dev, Y_dev



def read_data():
    sentences_train = []
    labels_train = []
    sentences_dev = []
    labels_dev = []

    test = False
    os.chdir('../../Data/split_dataset')
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if name == 'label_balanced_train.txt':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='\t'))

                for row in text:

                    tokens = row[0].strip().split()
                    tokens = " ".join(tokens)
                    sentences_train.append(tokens)
                    labels_train.append(row[1].strip())
            elif name == 'dev.txt':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='.'))

                for row in text:
                    print(row)
                    tokens = row[0].strip().split()
                    tokens.append('.')
                    tokens = " ".join(tokens)
                    sentences_dev.append(tokens)
                    labels_dev.append(row[1].strip())
    os.chdir('../../../Models/LM_BERT_Balanced_labels')
    return sentences_train, labels_train, sentences_dev, labels_dev

def read_data_dev():
    sentences = []
    labels = []

    test = False
  #  os.chdir('../Data/split_dataset')
    for root, dirs, files in os.walk('\t', topdown=False):
        for name in files:
            if name == 'dev.txt':
                file = open(os.path.join(root, name))
                text = list(csv.reader(file, delimiter='.'))

                for row in text:
                    print(row)
                    tokens = row[0].strip().split()
                    tokens.append('.')
                    tokens = " ".join(tokens)
                    sentences.append(tokens)
                    labels.append(row[1].strip())
   # os.chdir('../../LM')
    return sentences, labels

def save_dataset(X, Y, model_name):

    """save models prediction as csv file"""
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

    if config["loss"].upper() == "BINARY":
        loss_function = BinaryCrossentropy(from_logits=True)
    # elif config["loss"].upper() == "CUSTOM":
    #     loss_function = weighted_loss_function

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

    # transform string labels to one-hot encodings
    # encoder = LabelBinarizer()
    # Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    # Y_dev_bin = encoder.fit_transform(Y_dev)

    # pprint(f'Y_dev_bin: {Y_dev_bin}')
    # pprint(f'Y_train_bin: {Y_train_bin}')

    # Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
    # Y_dev = np.asarray(Y_dev).astype('float32').reshape((-1,1))

    # pprint(f'Y_dev: {Y_dev}')
    # pprint(f'Y_train: {Y_train}')

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
    model.fit(tokens_train, Y_train, verbose=1, epochs=epochs,
        batch_size= batch_size, validation_data=(tokens_dev, Y_dev),
        callbacks=[es, history_logger, TqdmCallback(verbose=2)])

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
    #if len(physical_devices) > 0:
       # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)



    #get parameters for experiments
    config, model_name = utils.get_config()
    
    if config['training-set'] != 'trial':
        model_name = model_name+"_"+str(config['seed'])

    set_log(model_name)

    #load data from train-test-dev folder
    #X_train, Y_train = read_data()
   # X_dev, Y_dev = read_data_dev()
    #X_train, Y_train, X_dev, Y_dev = read_data()
    

    X_train, Y_train, X_dev, Y_dev = load_data(utils.DATA_DIR, config)
    #run model

    classifier(X_train,X_dev,Y_train, Y_dev, config, model_name)

  

if __name__ == "__main__":
    main()
