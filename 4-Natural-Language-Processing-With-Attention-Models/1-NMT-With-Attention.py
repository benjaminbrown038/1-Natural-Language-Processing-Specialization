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
        
        self.embedding = tf.keras.layers.Embedding(input_dim = None,output_dim = None,mask_zero = None)
        
        self.rnn = tf.keras.layers.Bidirectional(merge_mode = "sum", layer = tf.keras.layers.None(units = None, return_sequences = None))


        def call(self,context):
            
            """Forward pass of this layer

            Args: 
                context (tf.Tensor): The sentence to translate
            Returns:
                tf.Tensor: Encoded sentence to translate
            """
            
            x = None
            x = None
            return x 

encoder = Encoder(VOCAB_SIZE,UNITS)


# pass a batch of sentences to translate from english to portugese
encoder_output = encoder(to_translate)


print(f'Tensor of sentences in english has shape: {to_translate.shape}\n')
print(f'Encoder output has shape: {enocder_output.shape}')


class CrossAttention(tf.keras.layers.Layer):
    
    def __init__(self,units):
        super().__init__()
        
        self.mha = (tf.keras.layers.None(key_dim = None,
                                         num_heads = None))
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self,context,target):
        """Forward pass of this layer

        Args:
            context (tf.Tensor): Encoded sentence to translate
            target (tf.Tensor): The embedded shifted-to-the-right translation

        Returns:
            tf.Tensor: Cross attention between context and target
        """

            attn_output = None(query = None, 
                               value = None)

            x = self.add([target,attn_output])
            x = self.layernorm(x)
            return x 
        
        

        
        
