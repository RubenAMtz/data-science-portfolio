from .preprocessing import clean_messsages, get_sequences,\
                    load_data, n_grams, pad_sequences, \
                    unzip_data, to_dataset
from .train import train
import random
import os
import matplotlib.pytplot as plt

# unzip data
if not os.path.exists("./tmp/messages"):
    unzip_data("./tmp/facebook-rbbbn.zip")

# load data
messages = load_data("./tmp/messages/inbox/")
messages = random.sample(messages, len(messages))

# clean data
cln_messages = clean_messsages(messages)

# create n grams
n_grams = n_grams(cln_messages)

# produce sequences
max_len = 50
tokenizer, padded_sequences = get_sequences(n_grams, max_len, 'pre', 'post')

# produce a Dataset
dataset = to_dataset(padded_sequences, tokenizer)
print(dataset)

# train model
output_path = "./models/word_prediction.h5"
if not os.path.exists(output_path):
    history = train(output_path)

    plt.figure()
    plt.plot(range(len(history.history['loss'])), history.history['loss'])
    plt.show()
    # st.plotly_chart(plt)

    plt.figure()
    plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'])
    plt.show()
    # st.plotly_chart(plt)