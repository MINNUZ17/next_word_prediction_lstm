import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the corpus
corpus_path = os.path.join('data', 'corpus.txt')
with open(corpus_path, 'r', encoding='utf-8') as file:
    text = file.read().lower()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# Create sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

# Print to confirm
print("Total sequences:", len(input_sequences))
print("Max sequence length:", max_seq_len)
print("Done preparing data.")
