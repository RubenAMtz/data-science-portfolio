import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def train(tokenizer, max_len, output_path):

    class myCallBack(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # st.progress(epoch)
            st.text("Loss: " + str(logs.get('loss')))
            st.text("Accuracy: " + str(logs.get('accuracy')))

    my_cb = myCallBack()
    reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index), output_dim=50, input_length=max_len-1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(500, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(500)),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(len(tokenizer.word_index), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    model.summary()


    history = model.fit(
        dataset,
        epochs=100,
        callbacks=[my_cb, reduce]
    )

    model.save(output_path)

    return history



