'''




'''

import nltk 
import numpy as np
import pickle
# WARNING:tensorflow:Gradients do not exist for variables ['vae/dense_1/kernel:0', 'vae/dense_1/bias:0'] when minimizing the loss. TODO 

import tensorflow as tf 
from tensorflow.keras import layers
#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu_devices[0], True)
#print("memory growth enabled")




verbose = False
max_length = 32
num_pos_inc = 4 # start, end,pad,unk tokens!!!
#num_pos = len(data['pos_dic'].keys()) + num_pos_inc


def make_pos_list(sentence, co, pos_dic, index_dic):
    if verbose: print(sentence)
    tokens = nltk.word_tokenize(sentence)
    if verbose: print(tokens)

    tagged = nltk.pos_tag(tokens)
#    print(tagged)
#    global co, pos_dic, index_dic
    target_pos = [-1] # <start> token
    for word,pos in tagged:
#        print(word, pos)
        if word=='<' or word=='>':
          continue
        if word=='unk':
          target_pos+= [-3] 
          continue 
          
        if not pos in pos_dic.keys():
            pos_dic[pos] = co[0]
            index_dic[co[0] ] = pos 
            co[0] +=1
        target_pos+=[ pos_dic[pos]]
    target_pos += [-2] # <end> token 
    target_pos += [-4]*max_length # <pad> token 
#    print(pos_dic, '\n', index_dic, '  ' , co[0] )
    if verbose: print(target_pos[:max_length] )
    return np.array(target_pos[:max_length] )+num_pos_inc
    
 

#baseaddr = 'vidfeat/'
#filename = baseaddr +'captionPOS4_100'
#batch_size = 64
##batch_size = 8

#data = load(filename)
#print(data)


#pos_ds = tf.data.Dataset.from_tensor_slices((data['target_list']+num_pos_inc,data['cap_id_list'],
#  data['vid_id_list']))
#pos_ds = pos_ds.batch(batch_size)



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class PosEncoder(tf.keras.Model):
  def __init__(self, max_length, fc_dims,embed_dim, num_pos,lstm_dim,latent_dim ):
    super(PosEncoder, self).__init__()
    self.max_length = max_length; self.lstm_dim=lstm_dim; 
    self.embed_dim= embed_dim; self.num_pos = num_pos 
    self.embed = layers.Embedding(input_dim=num_pos, output_dim=embed_dim)
    self.lstm = layers.LSTM(lstm_dim,
      activation  = 'tanh', recurrent_activation='sigmoid', dropout = 0.2,recurrent_dropout=0.1)  
    self.fc1 = layers.Dense(fc_dims,activation="sigmoid")
#    self.do1 = layers.Dropout(dropout)
    self.fc_mean = layers.Dense(latent_dim, name="z_mean")
    self.fc_var = layers.Dense(latent_dim, name="z_log_var")
  
  def call(self, caption  ):
    mask = tf.math.logical_not(tf.math.equal(caption , 0)) # TODO# each word is not *pad (,start, end?)token. 
    # batch,?, max_length (max: num_pos)
    caption = self.embed(caption) # embedding using trained embedding! 
    output = self.lstm(inputs = caption, mask = mask )
    output = self.fc1(output)
    
    z_mean = self.fc_mean(output)
    z_log_var = self.fc_var(output)
    z = Sampling()([z_mean, z_log_var])
    return [z_mean, z_log_var, z]


