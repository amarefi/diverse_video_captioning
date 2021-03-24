#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encoder4 changes to model diversity in captions (sentence length as input to decoder)

*** d_word : word embedding dim 
d_object : object embedding dim 

padding_mask in Decoder and mask in EncoderT 
"""


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.keras import layers
from capsc13data import *
import cocoeval 
from pycocoevalcap.meteor.meteor import Meteor
from subprocess import Popen, PIPE

#%% =================== parameters 
keep_boxes = 32; # boxes are sorted by score. 
class_num = 91

#embed_dim = 256
#embedding_dim = 256
units = 512
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
#features_shape = 2048
attention_features_shape = 64
prefix = 'captioning'
refcount = 20
refbatch = 'captions/refbatch.txt'
testbatch= 'captions/testbatch.txt'
base = '/home/amirhossein/Desktop/implement/dataset/meteor-1.5/'
#%% =================== models and classes


#%% =================== models and classes

def show(var,name="", rate=0.01):
  if np.random.random()<rate: 
    tf.print("DEB:",name,var)
  
def get_angles(pos, i, d_word):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_word))
  return pos * angle_rates
  
def positional_encoding(position, d_word):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_word)[np.newaxis, :],
                          d_word)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask_enc(mask): #(seq): # TODO this function is only for video
  seq = 1-tf.cast(mask, tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_padding_mask_dec(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_word, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_word = d_word
    
    assert d_word % self.num_heads == 0
    
    self.depth = d_word // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_word)
    self.wk = tf.keras.layers.Dense(d_word)
    self.wv = tf.keras.layers.Dense(d_word)
    
    self.dense = tf.keras.layers.Dense(d_word)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
#    tf.print('q,v,k',q.shape, v.shape, k.shape)

    q = self.wq(q)  # (batch_size, seq_len, d_word)
    k = self.wk(k)  # (batch_size, seq_len, d_word)
    v = self.wv(v)  # (batch_size, seq_len, d_word)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_word))  # (batch_size, seq_len_q, d_word)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_word)
        
    return output, attention_weights

def point_wise_feed_forward_network(d_word, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_word)  # (batch_size, seq_len, d_word)
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_word, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_word, num_heads)
    self.ffn = point_wise_feed_forward_network(d_word, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_word)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_word)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_word)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_word)
    
    return out2


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_word,d_object, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.query_fc = tf.keras.layers.Dense(d_object)
    self.att2_fc = tf.keras.layers.Dense(d_word)

    self.mha1 = MultiHeadAttention(d_word, num_heads)
    self.mha2 = MultiHeadAttention(d_word, num_heads)

    self.ffn = point_wise_feed_forward_network(d_word, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_word)
    
    if training:
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_word)
    else:
#        embed()
        attn1, attn_weights_block1 = self.mha1(x, x, x[:,-1:,:], look_ahead_mask[:,-1:,:]) # -1: in order to preserve dimension 
    attn1 = self.dropout1(attn1, training=training)
    if training:
        out1 = self.layernorm1(attn1 + x)
    else:
#        tf.print('214' , attn1.shape, x.shape)
        out1 = self.layernorm1(attn1 + x[:,-1:,:])
    
    # enc_output (batch_size, all_frames_boxes, d_object)
    # key: (batch_size, all_frames_boxes, ??? ) 
    # query: (batch_size, sentence_length, ??? ) 
    # => bigger key dims or smaller query dims... 
    enc_query = self.query_fc(out1)
    attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, enc_query, padding_mask)  # (batch_size, target_seq_len, d_object)
    attn2 = self.dropout2(attn2, training=training)
    attn2 = self.att2_fc(attn2)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_word)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_word)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_word)
    
    return out3, attn_weights_block1, attn_weights_block2

class EncoderT(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_object, num_heads, dff, maximum_position_encoding,param, rate=0.1):# input_vocab_size,
    super(EncoderT, self).__init__()

#    self.d_word = d_word
    self.num_layers = num_layers
    self.d_object = d_object
    
#    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_word)
#    self.embedding = tf.keras.layers.Dense(d_object, activation = 'relu') # TODO test w linear
    self.embedding = tf.keras.layers.Embedding(class_num,d_object-3)
#    self.pos_encoding = positional_encoding(maximum_position_encoding, 
#                                            self.d_object)
    
    
    self.enc_layers = [EncoderLayer(d_object, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
    
    self.param = param 
    self.box_encoding = tf.keras.layers.Dense(2,activation="sigmoid") #d_object
    self.frame_encoding = tf.keras.layers.Dense(1,activation="sigmoid")

    
        
  def call(self, x, training, mask):
#    tf.print(x)
    seq_len = tf.shape(x)[1]
    
    # x ( batch_size,seq_len, 6,300 (classes))   (not all 300 boxes are kept)
    # first seperate data: 
    x = tf.transpose(x[...,:keep_boxes], perm=[0,1,3,2])
    # tf.print(x.shape) TensorShape([64, 113, 32, 6])
    boxes = x[...,:4] # boxes are sorted by score 
    classes = x[...,4]
    scores = x[...,5]
    box_size = ((boxes[...,0]-boxes[...,2])*(boxes[...,1]-boxes[...,3]))[...,tf.newaxis]
    box_enc = self.box_encoding(tf.concat([boxes, box_size],-1))
#    class_enc  
#    scores 
#    class_enc * scores + box_enc + frame_enc 
    
    
    p = self.param 
    self.frame_num = tf.multiply(tf.ones_like(classes), tf.range(0,tf.cast(seq_len,tf.float32),dtype = tf.float32)[tf.newaxis,...,tf.newaxis]) 
    frame_enc = self.frame_encoding(self.frame_num[...,tf.newaxis])
    x = self.embedding(classes)  # (batch_size, input_seq_len, keep_boxes, d_object)
#    tf.print(x.shape, self.d_object, scores.shape,keep_boxes)

    assert(x.shape[3]==self.d_object-3)
    assert(scores.shape[2]==keep_boxes)
    if p.Score:
        x *= scores[...,tf.newaxis]
    x *= tf.math.sqrt(tf.cast(self.d_object, tf.float32))
    
    # adding embedding and position encoding.
#    x += self.pos_encoding[:, :seq_len, :] # it is not trainable 
    if p.Frame:
#        x += frame_enc 
        x = tf.concat([x,frame_enc], axis=-1)
    if p.Box:
#        x += box_enc
        x = tf.concat([x,box_enc], axis=-1)


    x = self.dropout(x, training=training)
    
    batch_size = x.shape[0]; 
    new_dim = x.shape[3]
    x = tf.reshape(x,(batch_size, -1,new_dim))
    
    mask = tf.repeat(mask, keep_boxes, axis = 3)
#    tf.print(x.shape, mask.shape)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
#      tf.print(x)
    return x  # (batch_size, input_seq_len, d_object)


class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_word,d_object, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_word = d_word
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_word)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_word)
    
    self.dec_layers = [DecoderLayer(d_word,d_object, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_word)
    x *= tf.math.sqrt(tf.cast(self.d_word, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)
    padding_mask = tf.repeat(padding_mask, keep_boxes, axis = 3)
    
    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_word)
    return x, attention_weights

class parameter():
    def __init__(self, box,score,frame):
        self.Box=box; self.Score=score; self.Frame = frame

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_word,d_object, num_heads, dff,  
               target_vocab_size, pe_input, pe_target,param, rate=0.1):
    super(Transformer, self).__init__()

    
    self.encoder = EncoderT(num_layers, d_object, num_heads, dff, 
                           pe_input,param, rate )

    self.decoder = Decoder(num_layers, d_word,d_object, num_heads, dff, 
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size,activation="softmax")
    
  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_object)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_word)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights




#def create_masks(inp, tar):
#  # Encoder padding mask
#  enc_padding_mask = create_padding_mask(inp)
#  
#  # Used in the 2nd attention block in the decoder.
#  # This padding mask is used to mask the encoder outputs.
#  dec_padding_mask = create_padding_mask(inp)
#  
#  # Used in the 1st attention block in the decoder.
#  # It is used to pad and mask future tokens in the input received by 
#  # the decoder.
#  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#  dec_target_padding_mask = create_padding_mask(tar)
#  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
#  
#  return enc_padding_mask, combined_mask, dec_padding_mask

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]



class training_trnsf():
    def __init__(self,trnsf,opt, loss_object,RL):
#        self.img_tensor=img_tensor; self.target=target; 
        self.transformer = trnsf; 
        self.loss_object = loss_object; self.RL = RL
        self.optimizer = opt
    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.


    @tf.function #(input_signature=train_step_signature)
#        def train_step(self, img_tensor, target,mask,vid_id,decoder,encoder,loss_object,ep):
    def train_step(self, img_tensor,i3d_tensor, target,mask,vid_id,loss_object,ep):
        
#        self.transformer = trnsf; 
        tar_inp = target[:, :-1]
        tar_real = target[:, 1:]
      
#      im, tr, ma = trainds.__iter__().next()
        # Encoder padding mask
        # also Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        enc_padding_mask = create_padding_mask_enc(mask)
        dec_padding_mask = enc_padding_mask
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        # tokenizer.index_word[0] = '<pad>'
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        dec_target_padding_mask = create_padding_mask_dec(tar_inp)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
#      fn_out, _ = sample_transformer(im, tr, training=False, 
#                                       enc_padding_mask=ma, 
#                                       look_ahead_mask=combined_mask,
#                                       dec_padding_mask=ma)

#      enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(img_tensor, tar_inp, 
                                     True, 
                                     enc_padding_mask, 
                                     combined_mask, 
                                     dec_padding_mask)
            loss = loss_function(tar_real, predictions, self.loss_object)
#        tf.print('393', tar_real.shape, predictions.shape)
    
        gradients = tape.gradient(loss, self.transformer.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
      
        return loss 
    
#      train_loss(loss) TODO
#      train_accuracy(tar_real, predictions)

#    @tf.function  # by changing decoder as an input, tf.function is not redefined. it is re arranged. so graph is changed and not accepted
#    def train_step(self, img_tensor, target,mask,decoder,encoder,loss_object):
#        loss = 0
#        # initializing the hidden state for each batch
#        # because the captions are not related from image to image
#        hidden = decoder.reset_state(batch_size=target.shape[0])
#
#        #  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1) 
#        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)#TODO 
#        # caused error, since last batch size is not 64
#
#        with tf.GradientTape() as tape:
#            features = encoder([img_tensor,mask])
#
#            for i in range(1, target.shape[1]):
#                # passing the features through the decoder
#                predictions, hidden = decoder(dec_input, hidden, features)#
#
#                loss += loss_function(target[:, i], predictions,loss_object)
#
#                # using teacher forcing
#                dec_input = tf.expand_dims(target[:, i], 1)
#
#                # increasing feedback in training --- avg target and prediction . 
#              
#
#        # total_loss = (loss / int(target.shape[1]))
#
#        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
#
#        gradients = tape.gradient(loss, trainable_variables)
#
#        optimizer.apply_gradients(zip(gradients, trainable_variables))
#
#        return loss
        
    
def create_masks(img_tensor, target, mask):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask_enc(mask)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = enc_padding_mask #create_padding_mask_enc(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
  dec_target_padding_mask = create_padding_mask_dec(target)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask

class testing_trnsf():
    def __init__(self,trnsf,loss_object,beam=1):
#        self.img_tensor=img_tensor; self.target=target; 
        self.transformer = trnsf; 
        self.loss_object = loss_object; 
        self.name = "trnsf"
        self.beam = beam
        
    @tf.function
    def test_step(self,img_tensor,i3d_tensor, target,mask,length):
    
#      hidden = self.decoder.reset_state(batch_size=target.shape[0]) we have no hidden state. 
      dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
      # output is the sentence. at first it is just [<start>], then new words will be appended
      output = dec_input
      encoder_input = img_tensor
      loss = 0
      
      result_sent = []#tf.zeros_like(target)
      # make new words, one by one
      for i in range(1, target.shape[2]): # max sentence length  target==(batch_size,20,sentence length)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks( encoder_input, output,mask)
#        tf.print("combined mask at test, ", combined_mask.shape)
#        tf.print("output at test, ", output.shape)
        
        # predictions.shape == (batch_size, seq_len, vocab_size)
        # output.shape == (batch_size, uncompleted_seq_len**, 1?)
        predictions, attention_weights = self.transformer(encoder_input, 
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
        
        # select the last word from the seq_len dimension ***
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
    
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # return the result if the predicted_id is equal to the end token
#        if tokenizer.index_word[predicted_id[0]] == '<end>':
#          return tf.squeeze(output, axis=0), attention_weights  batch calculation
        
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)
        
#        tf.print('508',tf.expand_dims(target[:, i],-1).shape, predictions.shape)
        for j in range(target.shape[1]): 
            loss += loss_function(target[:,j, i], predictions,self.loss_object) # first prediction is not <start> / ok
        result_sent.append(predicted_id)#result_sent[:,i].assign(predicted_id)

      # total_loss = (loss / int(target.shape[1]))
      result_sent = tf.transpose(tf.stack(result_sent))
      return loss,result_sent[0]#, total_loss
        
        
    
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_word, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_word = d_word
    self.d_word = tf.cast(self.d_word, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_word) * tf.math.minimum(arg1, arg2)




class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):# 512 seems good
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, img_tensor,mask, hidden):
    # features(CNN_encoder output) shape == (batch_size, maxVidlen = 113(needs mask), features_shape = 2048)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 113, hidden_size)
    score = tf.nn.tanh(self.W1(img_tensor) + self.W2(hidden_with_time_axis))

    mask = tf.cast( mask[:,:,tf.newaxis] , tf.float32)
    # mask shape == (batch_size , 113,1)
    mask = 1-mask # logical not
    # attention_weights shape == (batch_size, 113, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score)-mask*1e9, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * img_tensor
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
    
class attention_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, top_k, fc2_type):
    super(attention_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(top_k, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    if fc2_type=='softmax':
        self.fc2 = tf.keras.layers.Dense(top_k,activation="softmax")
    elif fc2_type== 'linear':
        pass
#        self.fc2 = tf.keras.layers.Dense(top_k)
#    self.fc3 = tf.keras.layers.Dense(self.units,activation="sigmoid")
    self.attention = BahdanauAttention(self.units)
    
#    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, x, hidden, image_and_mask):
    img_tensor,mask = image_and_mask

    # defining attention as a separate model
    # context_vector shape == (batch_size, features_shape = 2048)
    context_vector, attention_weights = self.attention(img_tensor, mask, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, features_shape + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x,tf.expand_dims(hidden, 1)], axis=-1)

    #x = self.layernorm1(x)
    
    # passing the concatenated vector to the GRU
    output, state = self.gru(x)
    
#    output = self.layernorm2(output)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state #, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

def attention_Encoder_void(maxVidlen, dim,lstm_dim):
    encoder_input = layers.Input(shape=(maxVidlen,dim)) # TODO masking TODO (None, )
    input_mask = layers.Input(shape=(maxVidlen),dtype='bool')
    encoder = tf.keras.Model([encoder_input,input_mask] ,[encoder_input,input_mask])
    return encoder    
    
def Encoder(maxVidlen, dim,lstm_dim):
    encoder_input = layers.Input(shape=(maxVidlen,dim)) # TODO masking TODO (None, )
    #encoder_embedded = layers.Embedding(input_dim=dim, output_dim=embed_dim)(encoder_input) TODO 
    # seems it is just for 1D data.
    #embedding = layers.Embedding(input_dim=6, output_dim=6, mask_zero=True)
    #masked_output = embedding(padded_inputs)

    # Return states in addition to output
    output, state_h, state_c = layers.LSTM(
        lstm_dim, return_state=True, name='encoder')(encoder_input)#, return_sequence=True TODO
    #encoder_state = [state_h, state_c] # IMPortant 
    # TODO use output instead...
    encoder_state = tf.concat([state_c, state_h],axis=-1)
    encoder = tf.keras.Model(encoder_input ,encoder_state)
    return encoder

def Encoder2(maxVidlen, dim,lstm_dim):
    encoder_input = layers.Input(shape=(maxVidlen,dim)) # TODO masking TODO (None, )
    #encoder_embedded = layers.Embedding(input_dim=dim, output_dim=embed_dim)(encoder_input) TODO 
    # seems it is just for 1D data.
    #embedding = layers.Embedding(input_dim=6, output_dim=6, mask_zero=True)
    #masked_output = embedding(padded_inputs)

    # Return states in addition to output
    output, state_h, state_c = layers.LSTM(
        lstm_dim, return_state=True, name='encoder')(encoder_input)#, return_sequence=True TODO 
    #encoder_state = [state_h, state_c] # IMPortant 
    # TODO use output instead...
#    encoder_state = tf.concat([state_c, state_h],axis=-1)
    encoder = tf.keras.Model(encoder_input ,[output])
    return encoder
    
def Encoder3mask(maxVidlen, dim,lstm_dim):
    encoder_input = layers.Input(shape=(maxVidlen,dim)) # TODO masking TODO (None, )
    #encoder_embedded = layers.Embedding(input_dim=dim, output_dim=embed_dim)(encoder_input) TODO 
    # seems it is just for 1D data.
    #embedding = layers.Embedding(input_dim=6, output_dim=6, mask_zero=True)
    #masked_output = embedding(padded_inputs)
    input_mask = layers.Input(shape=(maxVidlen),dtype='bool')
    
    # Return states in addition to output
    output, state_h, state_c = layers.LSTM(
        lstm_dim, return_state=True, name='encoder')(encoder_input, mask = input_mask)#, return_sequence=True TODO 
    #encoder_state = [state_h, state_c] # IMPortant 
    # TODO use output instead...
#    encoder_state = tf.concat([state_c, state_h],axis=-1)
    encoder = tf.keras.Model([encoder_input,input_mask] ,[tf.expand_dims(output,1)])
    return encoder

class POS_generator(tf.keras.Model):
  def __init__(self, dim, latent_dim):
    super(POS_generator, self).__init__()
    self.fc_pos1 = layers.Dense(dim ,activation="sigmoid")
    self.fc_pos2 = layers.Dense(dim ,activation="sigmoid")
    self.fc_pos3 = layers.Dense(latent_dim,activation="linear")
    
  def call(self, features):
    pos = self.fc_pos1(features)
#    tf.print('POS1',pos[0])
    pos = self.fc_pos2(pos)
    pos = self.fc_pos3(pos)[:, tf.newaxis, :]
    return pos 
    
class Encoder4(tf.keras.Model):
    def __init__(self,dim,lstm_dim,dropout, use_pos):
        super(Encoder4, self).__init__()
#        encoder_input = layers.Input(shape=(maxVidlen,dim)) # TODO masking TODO (None, )
#        input_mask = layers.Input(shape=(maxVidlen),dtype='bool')
        self.lstm = layers.LSTM(lstm_dim, return_state=True, name='encoder')#, return_sequence=True TODO 
#        self.encoder = tf.keras.Model([encoder_input,input_mask] ,[tf.expand_dims(output,1)])
        self.do= tf.keras.layers.Dropout(dropout)
        self.use_pos = use_pos
        if use_pos:
          self.fc_pos = POS_generator(dim, latent_dim)
        
    def call(self,inp_mask, training = False):
        encoder_input,input_mask = inp_mask
        enc2 = self.do(encoder_input)
        output, state_h, state_c = self.lstm(enc2, mask = input_mask)
        if self.use_pos:
          pos = self.fc_pos(output)
          return [tf.expand_dims(output,1), pos] 
        return tf.expand_dims(output,1)

class Encoder5_SDP(tf.keras.Model):
    def __init__(self,maxVidlen, dim,lstm_dim,dropout, use_pos):
        super(Encoder5_SDP, self).__init__()

        self.fc = layers.Dense(lstm_dim ,activation="sigmoid")
        self.do= tf.keras.layers.Dropout(dropout)
        self.use_pos = use_pos
        if use_pos:
          self.fc_pos = POS_generator(dim, latent_dim)
        self.lstmdim = lstm_dim
        self.soft = tf.nn.softmax #layers.Dense(maxVidlen , activation = 'softmax')
        
    def call(self,inp_mask, training = False):
        encoder_input,input_mask = inp_mask
        enc2 = self.do(encoder_input)
        #tf.print('\n', encoder_input.shape, input_mask.shape, input_mask[0])
        input_mask = tf.cast( input_mask , tf.float32)
        x = self.fc(encoder_input) * input_mask[...,tf.newaxis]
        #tf.print('X ',x[0])
        m = tf.matmul(x, tf.transpose(x, perm = [0,2,1])) / tf.sqrt(tf.cast(self.lstmdim,dtype=tf.float32)) # [[10,20,0],[20,50,0],[0,0,0]] # m:T*T
        # zero values are not good for softmax... 
        input_mask2 = 1-input_mask
        m -= 1e9*input_mask2[:,tf.newaxis,:]
        #tf.print('m', m[0])
        ac = self.soft( m)
        #tf.print('ac',ac.shape,x.shape,ac[0],summarize= -1)
        mm = tf.matmul(ac,x)* input_mask[...,tf.newaxis]
        #tf.print('matmul ', mm.shape, mm[0])
        vc = tf.math.reduce_sum( mm , axis = -2)
        #tf.print(vc.shape)
        #tf.print(vc[0],'\n', tf.math.reduce_sum(input_mask,axis = 1))    
        vc = vc/ tf.math.reduce_sum(input_mask,axis = 1)[...,tf.newaxis]
        output = vc
        
        if self.use_pos:
          pos = self.fc_pos(output)
          return [tf.expand_dims(output,1), pos] 
        return tf.expand_dims(output,1)
        
def Encoder_single_frame(maxVidlen, dim,lstm_dim):
    encoder_input = layers.Input(shape=(maxVidlen,dim)) # TODO masking TODO (None, )
    input_mask = layers.Input(shape=(maxVidlen),dtype='bool')
    
    fc = tf.keras.layers.Dense(lstm_dim,activation="sigmoid")
    encoder_state = fc(encoder_input)
    encoder_state = tf.expand_dims(encoder_state, 1)
#    fc2 = tf.keras.layers.Dense(lstm_dim,activation="sigmoid")
#    encoder_state = fc2(encoder_state)
    encoder = tf.keras.Model([encoder_input,input_mask] ,encoder_state)
    return encoder

class Encoder_single2(tf.keras.Model):
    def __init__(self,maxVidlen, dim,lstm_dim,dropout):
        super(Encoder_single2, self).__init__()
        self.do= tf.keras.layers.Dropout(dropout)
        self.fc = tf.keras.layers.Dense(lstm_dim,activation="sigmoid")
        
    def call(self,inp_mask, training = False):
        encoder_input,input_mask = inp_mask
        enc2 = self.do(encoder_input)
        encoder_state = self.fc(enc2)
        encoder_state = tf.expand_dims(encoder_state, 1)
        return encoder_state
            
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, top_k, fc2_type,layernorm,dropout,recurrent_do,feature_do, cell_type,embedding_layer=None ):
        super(RNN_Decoder, self).__init__()
        self.units = units; self.layernorm = layernorm
        self.cell_type = cell_type; self.rco = recurrent_do
        
        if embedding_layer!='':
          self.embedding = layers.Embedding(top_k, embedding_dim)
          self.embedding.trainable = True
          self.embedding(0)
#          print(self.embedding.get_weights())
          self.embedding(1)
#          print(self.embedding.get_weights()[0].shape, embedding_layer.shape)
                    
          self.embedding.set_weights([embedding_layer])
          print('using glove embedding')
        else:
          self.embedding = layers.Embedding(top_k, embedding_dim)
        if self.cell_type=='lstm_cell':
          self.lstm_cell = layers.LSTMCell(self.units, activation='linear',
                                       recurrent_initializer='orthogonal',
                                       dropout = dropout,
                                       recurrent_dropout = recurrent_do)
          self.fc_feature = layers.Dense(self.units,activation="sigmoid")
        if self.cell_type=='lstm_cell2':
          self.lstm_cell = layers.LSTMCell(self.units, activation='tanh',
                                       recurrent_initializer='orthogonal',
                                       recurrent_activation='tanh',
                                       dropout = dropout,
                                       recurrent_dropout = recurrent_do)
          self.lstm_cell2 = layers.LSTMCell(self.units, activation='linear',
                                       recurrent_initializer='orthogonal',
                                       recurrent_activation='tanh',
                                       dropout = dropout,
                                       recurrent_dropout = recurrent_do)							   
          self.fc_feature = layers.Dense(self.units,activation="sigmoid")
		
        if self.cell_type=='gru':
          self.gru = layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        if self.cell_type=='rnn':                               
          self.fc_s1 = layers.Dense(self.units,activation="sigmoid")
          self.fc_s2 = layers.Dense(self.units,activation="sigmoid")
          self.fc1 = layers.Dense(self.units)#,activation="sigmoid"

        if fc2_type=='softmax':
            self.fc2 = layers.Dense(top_k,activation="softmax")
        elif fc2_type== 'linear':
            self.fc2 = layers.Dense(top_k)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.do= layers.Dropout(feature_do)
        

        
#        self.do2= layers.Dropout(dropout)
        #    self.attention = BahdanauAttention(self.units)

    def call(self, x, state, visu_feat,training = False): # encoder_state is passed here as visual features
        # defining attention as a separate model
        #    context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
#        tf.print(visu_feat.shape, x.shape,'\n')
        
        if self.cell_type=='lstm_cell':
          # TODO layer normaliz ... 
          x,state = self.lstm_cell( inputs = x[:,0,:], states = state) 
          # state is a list 
        if self.cell_type=='lstm_cell2':
          # TODO layer normaliz ... 
          state, state2 = state 
          out1,state = self.lstm_cell( inputs = x[:,0,:], states = state) 
         # tf.print(out1[0])
          x,state2 = self.lstm_cell2( inputs = tf.concat([out1,x[:,0,:]],axis = -1), states = state2) 
          state = [state, state2]
          # state is a list         
        
        if self.cell_type=='rnn':
          if self.layernorm:
              x = self.layernorm1(x)
              visu_feat = self.layernorm2(visu_feat)
              state = self.layernorm3(state)
          x = tf.concat([visu_feat,x, state], axis=-1) # [tf.expand_dims(visu_feat, 1)
  ###         TODO ! x and visufeat shape must be the same to Concat.
          x = self.do(x,training)
          # passing the concatenated vector to the GRU
          # output, state = self.gru(x)
          state = self.fc_s1(x)
          state = self.fc_s2(state)
          # shape == (batch_size, max_length, hidden_size)
          x = self.fc1(state)
          # x shape == (batch_size * max_length, hidden_size)
          x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state # attention weights were returned, too . TODO

    def reset_state(self, batch_size, features= None,training = False):
      if self.cell_type=='rnn':
        return tf.zeros((batch_size,1, self.units))
      if self.cell_type=='lstm_cell':
        features = self.do(features,training)
        init_state = self.fc_feature(features)[:,0,:]
        return [tf.zeros((batch_size,self.units)), init_state]
      if self.cell_type=='lstm_cell2':
        init_state = self.fc_feature(features)[:,0,:]
        state = [tf.zeros((batch_size,self.units)), init_state]
        state2 = [tf.zeros((batch_size,self.units)),init_state]
        return [state, state2]

class Caption_scorer(tf.keras.Model):
  def __init__(self, lstm_dim,dropout ):
    super(Caption_scorer, self).__init__()
    # embedding_dim, units, top_k, fc2_type,layernorm,dropout,recurrent_do, cell_type ):
    self.lstm_dim = lstm_dim 
    self.lstm = layers.LSTM(lstm_dim,
      activation  = 'tanh', recurrent_activation='sigmoid', dropout = dropout,recurrent_dropout=0.1)# return_state=True,return_sequence=True TODO 
    self.fc_f = layers.Dense(lstm_dim,activation="sigmoid")
    self.fc_o = layers.Dense(lstm_dim,activation="sigmoid")
#    self.lstm = layers.LSTM(lstm_dim,
#      activation  = 'relu', recurrent_activation='relu', dropout = dropout,recurrent_dropout=0.1)
#    self.fc_f = layers.Dense(lstm_dim,activation="relu")
#    self.fc_o = layers.Dense(lstm_dim,activation="relu")
    self.fc_o2 = layers.Dense(1, activation= 'linear')
    self.do1 = layers.Dropout(dropout)
    self.do2 = layers.Dropout(0.1)
    
  def set_embed_layer(self, embedding_layer):
    self.embed = embedding_layer
    
  @tf.function
  def call(self, feature, caption, training = False  ):
    # TODO pay attention to masking the caption 
#    print(feature.shape)
#    assert(feature.shape[-1] == 600) # 600 for msvd
#    feature = tf.concat( [feature[..., :500], feature[..., 550:]],axis=-1)
    feature = self.fc_f(feature) 
    mask = tf.math.logical_not(tf.math.equal(caption , 0)) # TODO# each word is not *pad (,start, end?)token. 
    caption = self.embed(caption) # embedding using trained embedding! 
    output = self.lstm(inputs = caption, mask = mask )
    output = tf.concat([feature[:,0,:], output], -1)
    output = self.do1(output,training )
    output = self.fc_o(output)
    output = self.do2(output, training)
    output = self.fc_o2(output)  
    return output 

# at word level
def loss_function(real, pred,loss_object,ax=True):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
#    tf.print(tf.reduce_mean(loss_),mask,tf.math.maximum(tf.reduce_sum(mask),1),real[:3])
    if ax:
      return tf.reduce_mean(loss_)#/tf.math.maximum(tf.reduce_sum(mask),1)# it's very imp. if not used, sentence len will affect loss! 
    else:
      return loss_
      
class training():
    def __init__(self,decoder,encoder,opt,tkn,RL,posencoder = None,i3dencoder = None,pos_do = 0):
#        self.img_tensor=img_tensor; self.target=target; 
      self.optimizer = opt; self.RL = RL; 
      self.i3d = 'i3d' in conf['features'] 
      self.i3denc = 'i3d_clips' in conf['features'] and 'i3d' not in conf['features']
      self.i3dencoder = i3dencoder
      self.decoder = decoder; self.encoder = encoder ; self.tkn = tkn
      self.normalize_pos =tf.keras.layers.LayerNormalization(epsilon=1e-6)
      self.opt_pos = tf.keras.optimizers.Adam(learning_rate=0.05)
#      self.loss_object_pos = tf.keras.losses.CategoricalCrossentropy(from_logits=True )# Sparse - reduction='none'
      self.loss_object_pos = tf.keras.losses.MeanSquaredError()
      self.posencoder = posencoder
      self.pos_do = layers.Dropout(pos_do)
      if RL!=0:
        self.loss_ax = False
        cmd = 'java -Xmx2G -jar {}meteor-1.5.jar -norm -writeAlignments -f {} -l en -r {} -stdio'.format(base,prefix,refcount)
        cmd = cmd.split()
        self.meteor_comm =  Popen(cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True, bufsize=1) 
      else:
        self.loss_ax = True 
        
      self.pos= True if conf['snrio'] in ['pos', 'e2epos','trainpos','traine2epos','slen'] else False
      self.length = True if conf['snrio'] in ['slen_cte'] else False
        
    @tf.function  # by changing decoder as an input, tf.function is not redefined. it is re arranged. so graph is changed and not accepted
    def train_step(self, f, loss_object,ep):
        img_tensor,mask,i3d_tensor,maski3d, target,vid_id,cap_id,pos = f
        decoder=self.decoder; encoder=self.encoder;
        tokenizer = self.tkn
        batch_size=target.shape[0]
        loss = 0; kl_loss = 0
        length = tf.math.logical_not(tf.math.equal(target, 0))
        length = tf.cast(length, dtype=tf.float32)
        length = tf.reduce_sum(length,axis=1)[:,tf.newaxis,tf.newaxis]
        
        #  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1) 
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)#TODO 
        # caused error, since last batch size is not 64
        
        # RL!
        sentences = []
        with tf.GradientTape(persistent=True) as tape:
#          tf.print(img_tensor[0], mask[0])
          features = encoder([img_tensor,mask],True)
          # initializing the hidden state for each batch
          # because the captions are not related from image to image
#            tf.print(features.shape) # [64, 1, 500]
          if self.length:
            features = tf.concat([features,length/20],-1)
          if self.pos :
#            features, generated_pos = features 
            if self.posencoder:
              z_mean, z_log_var, pos = self.posencoder(pos[:,0,...])
              kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
              kl_loss = tf.reduce_mean(kl_loss)
              kl_loss *= -0.5
              pos=pos[:,tf.newaxis,:]
            pos = self.pos_do(pos,True)
#            print(pos)
            combined_pos = pos # pos*(10-ep)+generated_pos*ep# in order to gradually increase feedback
#            features = tf.concat([features,self.normalize_pos(pos)],-1)
#            tf.print(features[0][0][:10], pos[0][0][-10:],generated_pos[0][0][-10:])
#            show(generated_pos[0][0][-10:])
#            show("pos",pos[0][0][-10:])
#            tf.print("pos ",pos[0][0][:20], summarize=-1)
#            tf.print("gen pos ", generated_pos[0][0][:20])
            features = tf.concat([features,combined_pos],-1) #self.normalize_pos(combined_pos) // *** pay attention to 1233
          if False:
            loss_pos = self.loss_object_pos(pos, generated_pos )
            tf.stop_gradient( generated_pos ) # in order not to contribute in loss! 

          if self.i3d:
            features = tf.concat([features,i3d_tensor],-1)
          if self.i3denc:
            i3d_tensor = self.i3dencoder([i3d_tensor,maski3d])
            features = tf.concat([features,i3d_tensor],-1)
          hidden = decoder.reset_state(batch_size=target.shape[0],features=features,training = True )
#              tf.print(self.normalize_pos(pos))
#            tf.print(te) # [64, 1, 501]
          assert(target.shape[1]!=20)
          for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden = decoder(dec_input, hidden, features,True)#
#            show(hidden[0][:10],0.001)
#            tf.print(hidden[0])
            # RL!
            sentences += [ tf.argmax(predictions,axis=-1) ]#.numpy()
            
            loss += loss_function(target[:, i], predictions,loss_object,self.loss_ax)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

            # increasing feedback in training --- avg target and prediction . 
          norm = tf.math.logical_not(tf.math.equal(target, 0))
          norm = tf.cast(norm, dtype=loss.dtype)
          norm = tf.reduce_sum(norm)
#            tf.print(norm, norm/BATCH_SIZE)
          loss = loss/norm *batch_size 
          loss2 = loss 
          loss = tf.reduce_mean(loss) 
          if self.posencoder:
            kl_loss *= 10
            loss+= kl_loss
          
          if self.pos and False :
            loss_pos *= 7- loss 

        s = tf.stack(sentences)
        s = tf.transpose(s)
        
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

#        tt = gradients[-4]
#        tf.print(tt.shape)
#        tf.print( tf.reduce_sum(tt[500:520], axis  = -1),summarize=-1)

#        for i,t in enumerate(trainable_variables):
#          tf.print(t.name , '  ', t.shape , end='  ')
#          if gradients[i]==None:
#            tf.print("none")
#            continue
#          else:
#            tf.print(gradients[i].shape)
            
#          tf.print(tf.reduce_sum(t))
#        for g in gradients:
#          tf.print(tf.reduce_sum(g))
#        tf.print(gradients)
        
        # total_loss = (loss / int(target.shape[1]))
        if self.RL!=0 and ep>3: #ep=0,1,2,...
          meteorCoeff = np.mean( loss2.numpy()* (0.5 - self.meteor3(s.numpy(), vid_id,ep)) )/loss.numpy()### meteor:1+5* [0,1] 
          
          # meteorCoeff = self.RL*(1-4*np.mean(self.meteor(s.numpy(), vid_id,ep))) 
          # tf.constant( 1-meteor(sentences, img_name,ep) ,dtype='float64')
          # if meteorCoeff is high, means that sentences are already good! so needn't to change the weights..
          # usu. 0.15-0.25; so 1-MS has low telorance. 
          # any change here will change learning rate and difference with RL=False. 
#          embed()
        else:
          meteorCoeff = 1 #tf.constant(1.0 ,dtype='float64')
        
        gradients = tape.gradient(loss, trainable_variables)

        # TODO important notice! layers that dont contribute to loss, have None gradient. 
        # RL! 
        gradients = [tf.math.scalar_mul(meteorCoeff, g) if g is not None else g for g in gradients ]
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        
        
        if self.pos and False :
          tr_var_pos = self.encoder.fc_pos.trainable_variables
#          tf.print([n.name for n in tr_var_pos])
          gradients = tape.gradient(loss_pos, tr_var_pos)
#          print('START')
##          print( "METEOR1    ",self.meteor(np.array(sentences).T, vid_id,ep))
#          print( "METEOR2    ",self.meteor2(np.array(sentences).T, vid_id,ep))
#          if ep>0:
#            meteorCoeff = 1-4*np.mean(self.meteor(np.array(sentences).T, vid_id,ep))
#          else:
#            meteorCoeff = 1 
#          gradients = [tf.math.scalar_mul(meteorCoeff, g) if g is not None else g for g in gradients ]
          self.opt_pos.apply_gradients(zip(gradients,tr_var_pos))
#          tf.print(loss_pos, end='')

          
        return loss ,s,kl_loss#, [gradients,pos,generated_pos]
        
    def meteor3(self,sentences, vid_id,ep):
      vid_id = vid_id.numpy()
      cat = self.meteor_comm
      
      l = []
      for vid,sent in zip(vid_id,sentences):
        refs = ' ||| '.join([s[:-1] for s in ref_dict[vid]])
        
        tests = ' '.join([  tokenizer.index_word[ wd ] for wd in sent if tokenizer.index_word[ wd ] != '<end>' ] )
        echo = 'SCORE ||| '+refs+' ||| '+tests
#        print(echo)
        print(echo, file=cat.stdin, flush=True)
        print('EVAL ||| '+cat.stdout.readline()[:-1], file=cat.stdin, flush=True)
        l+=[float(cat.stdout.readline()[:-1])]
#      print(l)
      return np.array(l )
      
        
    def meteor(self,sentences, vid_id,ep):
    #        embed()
        batch_size=vid_id.shape[0]
        vid_id = vid_id.numpy()
        
        file1 = open(refbatch,"w+") # refaddr and testaddr
        for vid in vid_id:
            file1.writelines(ref_dict[vid ])# ref_dict is text but train_dict is num
        file1.close()   
    
        # next FOR will generate sentences and put them in a file...
        file1 = open(testbatch,"w+") # refaddr and testaddr
        for sent in sentences:
            t = ' '.join([  tokenizer.index_word[ wd ] for wd in sent if tokenizer.index_word[ wd ] != '<end>' ] )
            file1.write(t+'\n')
    #        for image,cap in valid_output_dict.items():
        file1.close()

    #        file1 = open(checkaddr.format(ep),"w+") # refaddr and testaddr
    #        for image in valid_dict.keys():
    #            t = int(image[-8:-4])
    #            file1.write(str(t)+'\n')
    #            file1.write(valid_output_dict[image]+'\n')
    #            for tt in ref_dict[t]:
    #                file1.write('\t'+tt)
    #        file1.close()
        # embed()
                
        
        
        cmd = 'java -Xmx2G -jar {}meteor-*.jar {} {} -norm -writeAlignments -f {} -l en -r {}'.format(base,testbatch,refbatch,prefix,refcount)
        cmd
        # java -Xmx2G -jar /home/amirhossein/Desktop/implement/dataset/meteor-1.5/meteor-*.jar -norm -writeAlignments -f captioning -l en -r 2 -stdio
        # java -Xmx2G -jar /home/amirhossein/Desktop/implement/dataset/meteor-1.5/meteor-*.jar captions/testbatch2.txt captions/refbatch2.txt -norm -writeAlignments -f captioning -l en -r 2
        # stats = SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        # EVAL ||| stats
#        prefix = 'captioning'
#        refcount = 20
#        refbatch = 'captions/refbatch.txt'
#        testbatch= 'captions/testbatch.txt'
#        base = '/home/amirhossein/Desktop/implement/dataset/meteor-1.5/'

        stream = os.popen(cmd)
        output = stream.read()
    #        embed()
    #        print(output.split(sep = '\n'))
    #        out = output.split(sep = '\n')[-10:]
        segmentdic = {i:ii for i,ii in enumerate(output.split('\n'))}
        out = []
        for i in range(11, 11+batch_size):
            out+= [float(segmentdic[i].split()[-1])]
        return np.array(out)
        
    def meteor2(self,sentences, vid_id,ep):
      self.meteor_obj = Meteor()
      batch_size=vid_id.shape[0]
      vid_id = vid_id.numpy()
      
#      file1 = open(refbatch,"w+") # refaddr and testaddr
#      for vid in vid_id:
#          file1.writelines(ref_dict[vid ])# ref_dict is text but train_dict is num
#      file1.close()   
#  
#      # next FOR will generate sentences and put them in a file...
#      file1 = open(testbatch,"w+") # refaddr and testaddr
#      for sent in sentences:
#          t = ' '.join([  tokenizer.index_word[ wd ] for wd in sent if tokenizer.index_word[ wd ] != '<end>' ] )
#          file1.write(t+'\n')
#  #        for image,cap in valid_output_dict.items():
#      file1.close()
      
#      IDs = [str(i) for i in generated_dict.keys()]
#      samples = {str(k):[' '.join(v[0])] for k,v in generated_dict.items()}
#      gts = {str(k):[sent[:-1] for sent in ref_dict[k]] for k in generated_dict.keys()} # removing \n char
      samples = {}
      for i in range(batch_size):
        t = ' '.join([  tokenizer.index_word[ wd ] for wd in sentences[i] if tokenizer.index_word[ wd ] != '<end>' ] )
        samples[str(vid_id[i])] = [t]
      gts = {str(k):[sent[:-1] for sent in ref_dict[k]] for k in vid_id}
      res = self.meteor_obj.compute_score(gts,samples)
      print('gts\n', gts, '\n samples \n', samples)
      print('2:' , res, end=' ' )
      return res[1] # res: mean, score_list 
      

class testing():
    def __init__(self,decoder,encoder,loss_object,tkn,beam=1,cap_sc=None,i3dencoder = None):
        self.decoder=decoder; self.encoder=encoder;  self.loss_object=loss_object; self.i3dencoder = i3dencoder
        self.beam = beam ;
        self.i3d = 'i3d' in conf['features'] 
        self.i3denc = 'i3d_clips' in conf['features'] and 'i3d' not in conf['features']
        self.normalize_pos =tf.keras.layers.LayerNormalization(epsilon=1e-6) 
        self.caption_scorer = cap_sc
        self.tkn = tkn
        self.pos= True if conf['snrio'] in ['pos', 'e2epos','trainpos','traine2epos','slen'] else False
        self.length = True if conf['snrio'] in ['slen_cte'] else False
                  
    @tf.function 
    def test_step(self,img_tensor,mask, i3d_tensor,maski3d, target,pos, test_desired_length):
        batch_size=target.shape[0]
        tokenizer = self.tkn
        loss = 0#tf.Variable(0, dtype = tf.float32)
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)#TODO 
        features = self.encoder([img_tensor,mask])
        length = tf.ones(batch_size,dtype = tf.float32)*test_desired_length/20
        length = length[:,tf.newaxis,tf.newaxis]
#        features = tf.concat([features,length],axis = -1)
        if self.length:
            features = tf.concat([features,length],-1)
        if self.pos:
#          features, generated_pos = features 
              
          features = tf.concat([features,pos ],-1) # self.normalize_pos(pos)
#          features = tf.concat([features,self.normalize_pos(generated_pos)],-1) 
        if self.i3d:
            features = tf.concat([features, i3d_tensor],-1)
        if self.i3denc:
            i3d_tensor = self.i3dencoder([i3d_tensor,maski3d])
            features = tf.concat([features,i3d_tensor],-1)
            
        hidden = self.decoder.reset_state(batch_size=target.shape[0], features=features )
        result_sent = []#tf.zeros_like(target)
        # make new words, one by one
        for i in range(1, max_length): #target.shape[-1]# max sentence length  target==(batch_size,20,sentence length)
            assert(target.shape[-1]!=20)
            # passing the features through the decoder
            predictions, hidden = self.decoder(dec_input, hidden, features)#
    #        predicted_id = tf.nn.top_k(predictions[0], k=3).indices.numpy() *** must specify axis but not available
            predicted_id = tf.math.argmax(predictions,axis = 1)
            
            if conf['test'] in ['final']:
#              assert(target.shape[1]==20)
              for j in range(target.shape[1]): 
                  loss += loss_function(target[:, j,i], predictions,self.loss_object)
            
            result_sent.append(predicted_id)#result_sent[:,i].assign(predicted_id)
            dec_input = tf.expand_dims(predicted_id, 1)
        norm = tf.math.logical_not(tf.math.equal(target, 0))
        norm = tf.cast(norm, tf.float32)#dtype=loss.dtype
        norm = tf.reduce_sum(norm)
#            tf.print(norm, norm/BATCH_SIZE)
        loss = loss/norm *batch_size*20
            
        # total_loss = (loss / int(target.shape[1]))
        result_sent = tf.transpose(tf.stack(result_sent))
        
#        if not cap_sc:
        if conf['test'] in ['finalscorer']:
          meteors = self.caption_scorer(features, result_sent)
          assert(meteors.shape[1]==1)
          assert(len(meteors.shape)==2)
        else:
          meteors = np.zeros((batch_size, 1),dtype = 'int32')

        return [loss,result_sent, features,meteors ] #, total_loss
        
##        calc loss for 20 target sentences
##        generate one sentence 
##        hidden == bs,...
##        features == bs,...
##        resulst_sent == bs, max_sent
##        
##        generate 3 sent and then calc maximum, return 3
##        calc loss. after choosing the best sentence. (for the word before)
##        i'm not calculating loss for now...(it's not insightful)
#        beam = self.beam 
#        
#        features = self.encoder([img_tensor,mask])
#        batch_sentences = []
#        for b in range(target.shape[0]):
#            feature = features[b:b+1]
#            results = [[tokenizer.word_index['<start>']] for i in range(beam)]
#            scores = [1 for i in range(beam)]
#            hiddens = [self.decoder.reset_state(batch_size=1) for i in range(beam)]
#            for i in range(max_length):
#                # beam ^ 2 =9 possiblities
#                possible_res=[]; possible_sc = []; possible_hid = []
#                # for j,(p,hid) in enumerate(p_hid):
#                for j in range(beam): # loop in sentences (from last step)
#                    if tokenizer.index_word[results[j][-1]] == '<end>':
#                        possible_res += [ results[j]]*beam
#                        possible_sc += [scores[j]]*beam
#                        possible_hid += [hid]*beam
#                        continue
#                    p,hid = self.decoder(tf.expand_dims([results[j][-1]], 0),hiddens[j], feature)
#                    t = tf.nn.top_k(p, k=beam)
#                    # print(t.values.numpy())
#                    for k in range(beam): # loop in new words
#                        possible_res += [ results[j]+[t.indices[0][k].numpy()] ]
#                        possible_sc += [scores[j]*t.values[0][k].numpy()]#
#                        possible_hid += [hid]
#                
#                # the problem was that at the beginning, top-3 sentences are the same (<start>+first)
#                
##                topk = np.array(possible_sc).argsort()[::-1]
#                topk = tf.argsort(possible_sc,direction='DESCENDING')
#                results = [];scores = [];hiddens=[]
#                for j in topk:
#                    if possible_res[j] in results:
#                        continue
#                    if len(results)==beam:
#                        break
#                    results += [possible_res[j]]
#                    scores += [possible_sc[j]]
#                    hiddens += [possible_hid[j]]                   
#                if len(results)!=beam:
#                        tf.print("error , 411")    
#               
#            r = []
#            for sentence in results:
#                r+=[[]]
#                for word in sentence:
#                    r[-1]+=[tokenizer.index_word[word]]
#        
#            batch_sentences += [r]    
#        return 0,batch_sentences
            


#            loss = 0
#            results = np.zeros( (self.beam, target.shape[2]),dtype='int')
#            results[:,:,0] = tokenizer.word_index['<start>']
#            scores = [1 for i in range(beam)]
#            hiddens = self.decoder.reset_state(batch_size=target.shape[0])
#            hiddens = tf.repeat( tf.expand_dims(hiddens, axis=1) , axis=1, repeats=beam) # batch_size
#            
#            for i in range(1, target.shape[2]): # max sentence length  target==(batch_size,20,sentence length)
#                            
#                                        
#def evaluate(image,ep, encoder, decoder):
#   temp_input = tf.expand_dims(map_func(image,[0]*45)[0], 0)
#   features = encoder(temp_input)
#   dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
#   hidden = decoder.reset_state(batch_size=1) # TODO using more batch size!
#   result = []
#   for i in range(max_length):
#      predictions, hidden = decoder(dec_input,hidden, features)#
#      predicted_id = tf.nn.top_k(predictions[0], k=3).indices.numpy()
#      result.append(tokenizer.index_word[predicted_id[0]])
#      if tokenizer.index_word[predicted_id[0]] == '<end>':
#          return result#, attention_plot
#      dec_input = tf.expand_dims([predicted_id[0]], 0)
#   return result #r[0]#, attention_plot
