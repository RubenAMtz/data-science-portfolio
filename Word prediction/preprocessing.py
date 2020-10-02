import tensorflow as tf
import numpy as np
import os
import zipfile
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import random


def unzip_data(path):
    zip = zipfile.ZipFile(open(path, "rb"))
    zip.extractall(path.split("/")[-2])
    zip.close()


def load_data(path):
    messages = []
    for (root, dir, files) in os.walk(path):
        for file in files:
            if file.split(".")[-1] == 'json':
                with open(os.path.join(root, file), "r", encoding='LATIN1') as json_file:
                    # print(os.path.join(root, file))
                    j_content = json.load(json_file)
                    for message in j_content.get('messages'):
                        if message['sender_name'] == 'Ruben Alvarez':
                            if message.get('content'):
                                messages.append(message.get('content').lower())
    return messages


def clean_messsages(messages):
    cln_msgs = []
    # regrex_pattern = re.compile(pattern="["
    #                                     u"\U0001F600-\U0001F64F"  # emoticons
    #                                     u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    #                                     u"\U0001F680-\U0001F6FF"  # transport & map symbols
    #                                     u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    #                                     "]+", flags=re.UNICODE)
    for message in messages:
        # print(message)
        message = re.sub(r'\s*(?:https?://)?www\.\S*\.[A-Za-z]{2,5}\s*', '', message)
        message = re.sub(r'http\S+', '', message)
        # message = regrex_pattern.sub(r'', message)
        if message == "enviaste un archivo adjunto.":
            continue
        # if len(message.split()) > 200:
        #     continue
        if message == '' or message == ' ':
            continue
        if len(message.split()) == 1:
            continue
        cln_msgs.append(message)
    return cln_msgs


def get_sequences(text, max_len, padding, trunc):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding=padding, truncating=trunc)
    return tokenizer, padded_sequences


def n_grams(text):
    n_grams = []
    for sentence in text:
        for i in range(1, len(sentence.split())):
            n_grams.append(sentence.split()[:i+1])
    return n_grams


def to_dataset(padded_sequences, tokenizer):

    X = tf.data.Dataset.from_tensor_slices(padded_sequences)

    X = X.map(lambda x: x[:-1])
    X = X.batch(100).prefetch(1)
    print(X)

    y = tf.data.Dataset.from_tensor_slices(padded_sequences[:, -1])
    y = y.batch(100).prefetch(1)
    y = y.map(lambda x: tf.one_hot(x, len(tokenizer.word_index)))
    print(y)
    dataset = tf.data.Dataset.zip((X, y))
    return dataset


