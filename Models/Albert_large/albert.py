import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tokenization
from keras.models import load_model
from tensorflow.keras.layers import Dense, Input, Bidirectional, LSTM, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder  
from keras.utils import np_utils
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input
import random


def load_data():
    """Read in data sets and returns sentences and labels"""
    sentences = []
    labels = []
    with open('../../Data/split_dataset/balanced_train+other_templates.txt', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            sentences.append(" ".join(tokens[:-1]))
            labels.append(int(tokens[-1]))
    return sentences, labels

def load_data_dev():
    """Read in data sets and returns sentences and labels"""
    sentences = []
    labels = []
    a = b = 0
    with open('../../Data/split_dataset/dev.txt', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            sentences.append(" ".join(tokens[:-1]))
            labels.append(int(tokens[-1]))
            # if tokens[-1] == "1" and b < 50:
            #     sentences.append(" ".join(tokens[:-1]))
            #     labels.append(int(tokens[-1]))
            #     b+=1
    return sentences, labels

def bert_encode(texts, tokenizer, max_len=512):
    ''' Encodes all text for bert '''
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def train_bert(bert_layer, max_len=512):
    ''' Trains a bert model'''

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    #x = Dropout(0.25)(x)
    x = Dense(2, activation='softmax')(clf_output)
    opt = Adam(learning_rate=0.00001)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=x)
    model.compile(optimizer=opt , loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    #module_url = "https://tfhub.dev/google/albert_base/3"
    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    bert_layer = hub.KerasLayer(module_url, trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    X_train, Y_train = load_data()
    X_dev, Y_dev = load_data_dev()

    c = list(zip(X_train, Y_train))
    random.shuffle(c)
    X_train, Y_train = zip(*c)

    le = LabelEncoder()
    Y_train = np_utils.to_categorical(le.fit_transform(Y_train))
    Y_dev = np_utils.to_categorical(le.fit_transform(Y_dev))

    X_train = bert_encode(X_train, tokenizer, max_len=100)
    X_dev = bert_encode(X_dev, tokenizer, max_len=100)

    bert_model = train_bert(bert_layer, max_len=100)

    bert_model.fit(
        X_train, Y_train,
        validation_data=(X_dev, Y_dev),
        epochs=5,
        batch_size=64,
    )
    bert_model.save("initial_model.h5")

if __name__ == '__main__':
    main()