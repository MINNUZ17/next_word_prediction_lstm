import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = load_model("next_word_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_len = model.input_shape[1]

def predict_top_k_words(seed_text, k=3):
    token_list = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    predicted = model.predict(token_list, verbose=0)[0]

    top_k_indices = predicted.argsort()[-k:][::-1]
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}

    top_k_words = [index_to_word.get(i, "") for i in top_k_indices]
    return top_k_words
