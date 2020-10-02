import tensorflow as tf
import numpy as np
"""
# Learning to produce text from Messenger messages (my own messages)  
- Downloaded my messenger data from facebook
- Loaded the data as it is
"""


"""
# Cleaning  
The process removes hyperlinks, messages with a single word, emojis and some facebook automated messages.
"""

"""
# N-grams: 
Create n-grams >= 2 elements per each sentence
"""

"""
# Padded n-grams
Pad to the left the n-grams to an arbitrary length (50 default) 
"""

"""
# X and y
Split the padded n-grams into X (first 49 elements) and y (last element, index 50 given that padded to 50 elements)
"""

model = tf.keras.models.load_model("./models/embeddings/my_word_prediction.h5")

"""
# Prediction
Start the first words of the sentence:
"""
start = st.text_input("Starting sequence: ", 'xd', 120)

num_words_to_predict = 50
for i in range(num_words_to_predict):
    sequences = tokenizer.texts_to_sequences([start.lower()])
    if len(start.split()) > max_len:
        last_sequence = start.split()[-max_len:]
        sequences = tokenizer.texts_to_sequences([last_sequence])
    padded = pad_sequences(sequences, maxlen=max_len - 1, truncating='post', padding='pre')
    # print(padded.shape)
    # padded = tf.expand_dims(padded, axis=-1)
    # print(padded.shape)
    prediction = model.predict(padded)
    index = np.argmax(prediction, axis=1)[0]
    word = tokenizer.index_word.get(index, '<OOV>')
    start += " " + word

"""Output: """
print(start)