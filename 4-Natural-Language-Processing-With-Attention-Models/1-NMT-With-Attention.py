import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from collections import Counter
from utils import sentences, train_data, val_data, english_vectorizer, portugese_vectorizer, masked_loss, masked_acc, tokens_to_text
import w1_unittest

portugese_sentences, english_sentences = sentences
print(f"English (to translate) sentence:\n\n{english_sentences[-5]}\n")
print(f"Portugese (translation) sentence:\n\n{portugese_sentences[-5]}")

del portugese_sentences
del english_sentences
del sentences

print(f"First 10 words of the english vocabulary:\n\n{english_vectorizer.get_vocabulary()[:10]}\n")
print(f"First))")

# size of the vocabulary
vocab_size_por
vocab_size_eng

print
print


# this helps you convert words to ids
word_to_id
# this helps you convert from ids to words
id_to_word

unk_id
sos_id
eos_id
baunilha_id

print
print
print
print

for (to_translate, sr_translation), translation in train_data.take(1):
    print
    print
    print

VOCAB_SIZE = 12000
UNITS = 256

class Encoder(tf.keras.layers.Layer):
    def __init__(self,vocab_size,units):
        """
          Initializes an instance of this class

          Args: 
            vocab_size (int): Size of the vocabulary
            units (unit): Number of units in the LSTM layer
        """
        super(Encoder,self).__init__()
        
