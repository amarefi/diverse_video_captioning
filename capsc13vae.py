"""
using valid set for VAE cross validation 

i think masking is not needed... # TODO  

https://keras.io/examples/generative/vae/
https://www.tensorflow.org/tutorials/text/text_generation

# there are 31 POSes


id list train is used for training data. 
test and valid data are filled with zero 

"""
from time import time 
import nltk 
import numpy as np
import pickle
from capsc13dataset import *
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# WARNING:tensorflow:Gradients do not exist for variables ['vae/dense_1/kernel:0', 'vae/dense_1/bias:0'] when minimizing the loss. TODO 

import tensorflow as tf 
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
print("memory growth enabled")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--restore", action="store_false", default=True)
parser.add_argument("--dataset", type=str,default = 'msrvtt')
parser.add_argument("--silent", action="store_false", default=True)
args = parser.parse_args()
training = args.restore 
dataset = args.dataset 
silent = args.silent
print('dataset',dataset)

#sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
#tokens = nltk.word_tokenize(sentence)
#print(tokens)
#tagged = nltk.pos_tag(tokens)
#print(tagged)

verbose = False
max_length = 32
# max_length of sentence = 32 # check capsc13dataset 
num_pos_inc = 4 # start, end,pad,unk tokens!!!

# pos_dic is for whole dataset (capsc13dataset)
def extract(dic):
  target_list = []
  cap_id_list = []
  vid_id_list = []  
  for k,vv in dic['dict'].items(): # [sentence,vid_id,cap_id,pos]
    for v in vv:
      target_list += [v[3]] 
      cap_id_list += [v[2]]
      vid_id_list += [k]    
  print(vv[:2])

  target_list = np.array( target_list); cap_id_list = np.array( cap_id_list); vid_id_list = np.array( vid_id_list)
  print('target_list.shape',target_list.shape)
    #print(cap_id_list)

  pos_ds = tf.data.Dataset.from_tensor_slices((target_list,cap_id_list ,vid_id_list ))#  +num_pos_inc it's done in capsc13pos
  pos_ds = pos_ds.shuffle(200000)
  pos_ds = pos_ds.batch(batch_size)

  return pos_ds
  
def load():  
#  try:
#    infile = open(filename,'rb')
#    data = pickle.load(infile)
#    infile.close()
#    print('loaded')
#  except:
            
  batch_size = 64
  trainds  = extract(Train) 
  validds = extract(Valid)
  testds = extract(Test)

#  data = {'trainds':trainds, 'testds':testds', 'validds':validds }
#    outfile = open(filename,'wb')
#    pickle.dump(data ,outfile)
#    outfile.close()
#    print('extracted')
  return trainds,validds,testds 

baseaddr = 'vidfeat/vae/'
#filename = baseaddr +dataset+'DATASET_'+str(latent_dim) # ** name msvdDATASET_50
batch_size = 64
#batch_size = 8
if dataset == 'msvd':
  dataset_config = msvd_config
  pos_dic, index_dic = msvd_caption(msvd_config)
  top_k = 8000; 
if dataset == 'msrvtt':
  msrvtt_config['cap_data'] = 'vidfeat/msrvtt/msrvtt_captions_valid10percent'
  dataset_config = msrvtt_config
  pos_dic, index_dic = msrvtt_caption(msrvtt_config,0.1)
  top_k = 12000

print(pos_dic)

trainds,validds,testds = load()
#print(data)
num_pos = len(pos_dic.keys()) + num_pos_inc

# ==========================================================

#print(data)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


#encoder_input = keras.Input(shape=(max_length))
#print(encoder_input.shape)
#x=layers.Embedding(input_dim=max_length, output_dim=embed_dim)(encoder_input)
#print(x.shape)
#x = layers.LSTM(
#  lstm_dim, return_state=False, name='encoder_lstm', return_sequences=False)(x)
##layers.Conv(5,3,activation="relu", strides=2, padding="same")
##x = layers.Dense( 
#print(x.shape)        
#x = layers.Dense(lstm_dim, activation="relu")(x)
#z_mean = layers.Dense(latent_dim, name="z_mean")(x) # with no activation TODO 
#z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
#z = Sampling()([z_mean, z_log_var])
#encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
#encoder.summary()

class Encoder(tf.keras.Model):
  def __init__(self, max_length, lstm_dim,latent_dim, embed_dim, num_pos ):
    super(Encoder, self).__init__()
    self.max_length = max_length; self.lstm_dim=lstm_dim; 
    self.embed_dim= embed_dim; self.num_pos = num_pos 
    self.embed = layers.Embedding(input_dim=num_pos, output_dim=embed_dim)
    self.lstm = layers.LSTM(
      lstm_dim, name='encoder_lstm', return_sequences=False, # return_state=True,
      activation='relu', recurrent_activation='sigmoid')
    self.fc1 = layers.Dense(lstm_dim, activation="sigmoid", name = "state")
    self.fc_mean = layers.Dense(latent_dim, name="z_mean")
    self.fc_var = layers.Dense(latent_dim, name="z_log_var")
  
  def call(self, data):
#    tf.print(data.shape)
    # batch, max_length (max: num_pos)
#    tf.print('1 ',data)
    mask = tf.math.logical_not(tf.math.equal(data, -4+num_pos_inc))
    
    data = self.embed(data)
#    tf.print('2 ',data)
#    tf.print(data.shape, mask.shape)
    # batch, max_length, embed_dim
    state = self.lstm(data, mask  = mask ) # this is output of lstm 
#    tf.print('3 ',state)
    state = self.fc1(state)
#    tf.print('4 ',state)
    z_mean = self.fc_mean(state)
    z_log_var = self.fc_var(state)
    z = Sampling()([z_mean, z_log_var])
    return [z_mean, z_log_var, z]

    
#latent_inputs = keras.Input(shape=(latent_dim,))
#x = layers.Dense(lstm_dim, activation="relu")(latent_inputs)
#lstm = layers.LSTM(
#  lstm_dim, return_state=False, name='decoder_lstm', return_sequences=True)
#decoder_output = lstm(..., initial_state = x)


    
class Decoder(tf.keras.Model):
  def __init__(self, max_length, lstm_dim, embed_dim,num_pos):
    super(Decoder, self).__init__()
    self.max_length  = max_length
    self.lstm_dim = lstm_dim; self.num_pos = num_pos; self.embed_dim = embed_dim
    # Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch. 
    self.embed = layers.Embedding(input_dim=num_pos , output_dim=embed_dim)
    self.gru = layers.GRU(lstm_dim,
                        return_sequences=True,
                        recurrent_initializer='glorot_uniform') # ,return_state=True,
    self.fc_latent = layers.Dense(lstm_dim, activation="relu")
    self.fc = tf.keras.layers.Dense(num_pos, activation="softmax")
    
  def call(self,latent  ):#, old
    init = self.fc_latent(latent )
    bs= latent.shape[0]
    self.dummy = tf.ones((bs, max_length-1, 1),dtype = tf.float32)
    
    output  = self.gru(self.dummy,initial_state = init)
    output = self.fc(output)
    # old: batch,1 (max: num_pos)
#    old = self.embed(old)
    # old: batch,1,embed_dim
    
#    self.gru.reset_states()
#    old = tf.ones((
#    for i in range(self.max_length)
#    new = self.gru(old )
    # new: batch,1,embed_dim
#    new = self.fc(new) 
    # new: batch, 1, num_pos 	
    return output #new
    
  def reset(self, latent):
    latent = self.fc_latent(latent)
#    tf.print(latent.shape)
    self.gru.reset_states(states = latent)
    bs= latent.shape[0]
    
#    self.gru(latent[:,tf.newaxis,:])
#    self.gru(
#    self.gru(initial_state = latent)
      
      
    
        
#def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
#  model = tf.keras.Sequential([
#    tf.keras.layers.Embedding(vocab_size, embedding_dim,
#                              batch_input_shape=[batch_size, None]),
#    tf.keras.layers.GRU(rnn_units,
#                        return_sequences=True,
#                        stateful=True,
#                        recurrent_initializer='glorot_uniform'),
#    tf.keras.layers.Dense(vocab_size)
#  ])
#  return model

#model = build_model(
#    vocab_size = len(vocab),
#    embedding_dim=embedding_dim,
#    rnn_units=rnn_units,
#    batch_size=BATCH_SIZE)


#def generate_text(model, start_string):
#  # Evaluation step (generating text using the learned model)

#  # Number of characters to generate
#  num_generate = 1000

#  # Converting our start string to numbers (vectorizing)
#  input_eval = [char2idx[s] for s in start_string]
#  input_eval = tf.expand_dims(input_eval, 0)

#  # Empty string to store our results
#  text_generated = []

#  # Low temperatures results in more predictable text.
#  # Higher temperatures results in more surprising text.
#  # Experiment to find the best setting.
#  temperature = 1.0

#  # Here batch size == 1
#  model.reset_states()
#  for i in range(num_generate):
#    predictions = model(input_eval)
#    # remove the batch dimension
#    predictions = tf.squeeze(predictions, 0)

#    # using a categorical distribution to predict the character returned by the model
#    predictions = predictions / temperature
#    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

#    # We pass the predicted character as the next input to the model
#    # along with the previous hidden state
#    input_eval = tf.expand_dims([predicted_id], 0)

#    text_generated.append(idx2char[predicted_id])

#  return (start_string + ''.join(text_generated))


#decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
#decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
#decoder.summary()
#decoder = Decoder(max_length, lstm_dim, embed_dim, num_pos)
#decoder.build()
#decoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')
        self.optimizer = keras.optimizers.Adam( learning_rate=0.0015) # learning_rate=0.005
        
    def test(self, data):
      batch_size = data.shape[0] 
      z_mean, z_log_var, z = self.encoder(data)
#        self.decoder.reset(z)
      output = self.decoder(z)
      reconstruction = tf.argmax(output,axis=-1) 
      
      mask = tf.math.logical_not(tf.math.equal(data[:,1:], -4+num_pos_inc))
      mask = tf.cast(mask, dtype=output.dtype)
      l = self.loss_object(data[:,1:], output )
      l *= mask
      reconstruction_loss = l
      reconstruction_loss = tf.reduce_mean( reconstruction_loss)
      kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
      kl_loss = tf.reduce_mean(kl_loss)
      kl_loss *= -0.5
      total_loss = reconstruction_loss + kl_loss*0.8 #0.1*
      # using teacher forcing 
#      old = tf.Variable([-1+num_pos_inc]*batch_size, dtype = tf.float32)[:,tf.newaxis] 
#      reconstruction = tf.zeros((batch_size,0),dtype= tf.int64)
#      self.decoder(old) # to solve batch size error 
#      self.decoder.reset(z)
#      reconstruction_loss =0 
#      for i in range(max_length-1):
#        new = self.decoder(old)
##        reconstruction_loss += self.loss_object(data[:,i+1], new )
#        
#        new = tf.argmax(new,axis=-1)
#        old = new
#        reconstruction = tf.concat([reconstruction,new],axis = -1)
      return {"reconstruction":reconstruction, "z":z, "z_mean":z_mean, 
        "kl_loss":kl_loss, "reconstruction_loss":reconstruction_loss,"loss": total_loss}
      

#    def train_step(self, data):
    def call(self, data):
#        tf.print(data.shape)
      if isinstance(data, tuple):
          data = data[0]
      batch_size = data.shape[0] 
      
      # it must be outside of tape??
#      old = tf.Variable([-1+num_pos_inc]*batch_size, dtype = tf.float32)[:,tf.newaxis] 
#      self.decoder(old) # to solve batch size error 
      with tf.GradientTape() as tape:
        z_mean, z_log_var, z = self.encoder(data)
#        self.decoder.reset(z)
        output = self.decoder(z)
        mask = tf.math.logical_not(tf.math.equal(data[:,1:], -4+num_pos_inc))
        mask = tf.cast(mask, dtype=output.dtype)
        l = self.loss_object(data[:,1:], output )
        l *= mask
        reconstruction_loss = l
        # using teacher forcing 
#        reconstruction = tf.zeros((batch_size,0),dtype= tf.int64)
#        self.decoder.reset(z)
#        reconstruction_loss =0 
#        for i in range(max_length-1):
#          new = self.decoder(old)
#          old = data[:,i+1][:,tf.newaxis] 
#          
#          mask = tf.math.logical_not(tf.math.equal(data[:,i+1], -3+num_pos_inc))
#          
#          l = self.loss_object(data[:,i+1], new )
#          mask = tf.cast(mask, dtype=l.dtype)
#          reconstruction_loss += mask*l
#              
#          new = tf.argmax(new,axis=-1)
#          reconstruction = tf.concat([reconstruction,new],axis = -1)
        reconstruction_loss = tf.reduce_mean( reconstruction_loss)
#            reconstruction = decoder(z)
#            reconstruction_loss = tf.reduce_mean(
#                keras.losses.binary_crossentropy(data, reconstruction)
#            )

#            reconstruction_loss *= 28 * 28
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss*0.8 #0.1*
#        total_loss = tf.reduce_mean(z)+tf.reduce_mean(z_mean)+tf.reduce_mean(z_log_var)
#        print(total_loss)
#        print(self.count_params())
#        print(dir(self))
#        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
#        print(members)
#        
#        print(self.input)
#        for t in self.trainable_weights: *** TODO imp. how does it know decoder as a member // maybe like above loop 
#          print(t.shape, t.name )
#        print(self.trainable_weights)
      
#      trainable_variables = encoder.trainable_variables + decoder.trainable_variables
#      grads = tape.gradient(total_loss, trainable_variables)
#      tf.print(total_loss)
      grads = tape.gradient(total_loss, self.trainable_weights)
#      W = [self.encoder.fc1.trainable_weights, self.encoder.embed.trainable_weights, self.decoder.embed.trainable_weights, self.decoder.fc.trainable_weights]
#      grads = tape.gradient(total_loss, W)
#        WARNING:tensorflow:Gradients do not exist for variables ['vae/dense/kernel:0', 'vae/dense/bias:0'] when minimizing the loss. TODO maybe because of very small gradients 
#      print(grads)
#        print(data.shape)
#      print('=')
#      for g in grads:
#        print('-')
##        for gg in tf.Variable(g):
#        print(g.shape, tf.reduce_mean(g))
#      tf.print('state trainable ',self.encoder.fc1.trainable)
#      for i,g in enumerate(grads):
#        tr = self.trainable_weights
#        tf.print('\n\n',tr[i].shape, tr[i].name )
#        tf.print(tr[i])
#        try:
#          tf.print(g.shape, tf.reduce_sum(g).numpy() , g)
#        except:
#          tf.print("NONNEEE")
#      print(grads)
      self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
      return {
          "loss": total_loss,
          "reconstruction_loss": reconstruction_loss,
          "kl_loss": kl_loss,
      }

#(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
#mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

#vae = VAE(encoder, decoder)
#vae.compile(optimizer=keras.optimizers.Adam())
#vae.fit(mnist_digits, epochs=30, batch_size=128)

latent_dim = [50,100,200]
embed_dim = 20
lstm_dim = 100
results = {}

for latent_dim in latent_dim:
  param = 'latentdim {}'.format(latent_dim)
  encoder = Encoder(max_length, lstm_dim,latent_dim , embed_dim, num_pos )
  print('\nEncoder max_length{}, lstm_dim{},latent_dim{} , embed_dim{}, num_pos{}'.\
    format(max_length, lstm_dim,latent_dim , embed_dim, num_pos))
  decoder = Decoder(max_length, lstm_dim, embed_dim, num_pos)
  print('Decoder max_length {}, lstm_dim {},embed_dim{}, num_pos {}'.format(max_length, lstm_dim,embed_dim, num_pos))
  print('embed_dim {}, latent_dim {}'.format(embed_dim, latent_dim))
  vae = VAE(encoder, decoder)
  results[param] = []


  from datetime import datetime
  
  now = datetime.now()
  date_time = now.strftime("%Y%m%d%H%M")
   ## Checkpoint
  #checkpoint_path = "./checkpoints/VAE/ckpt" # .format(date_time)
  #ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder,
  #                           vae = vae)
  #ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

  #start_epoch = 0
  #if ckpt_manager.latest_checkpoint:
  #  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

  #for (batch, (target, cap_id)) in enumerate(pos_ds):
  #  for i in [1,2]:
  #    e =encoder(target[i][tf.newaxis,...])
  #    print(target[i])
  #    print(e)
  #  break


  time_list= [0]
  epochs = 20
  if training:
    print(' -= training =-')
    for ep in range(epochs):
      time_list+=[time()] ; print('time:',time_list[-1]-time_list[-2])
      loss = 0;  rloss=0;klloss=0;
      print("")
      for (batch, (target, cap_id,vid_id)) in enumerate(trainds):
        if target.shape[0]!= batch_size:
    #      print('returning due to bad batch_size \n', target.shape[0])
          continue
        l = vae(target)
        loss += l['loss']
        rloss+=l['reconstruction_loss']
        klloss+=l['kl_loss']
    #    tf.print( l)
      
        if silent: print('\r ep {} b {}  loss {:.3f}   r-loss {:.3f}  kl-loss {:.6f}         {:.3f} {:.3f}'.format
          (ep, batch, loss.numpy()/batch, rloss.numpy()/batch, klloss.numpy()/batch, l['reconstruction_loss'],l['kl_loss']), end = '\r')
      print('\r ep {} b {}  loss {:.3f}   r-loss {:.3f}  kl-loss {:.6f}         {:.3f} {:.3f}'.format
          (ep, batch, loss.numpy()/batch, rloss.numpy()/batch, klloss.numpy()/batch, l['reconstruction_loss'],l['kl_loss']), end = '\r')
  #      if batch ==100: break
  #      break
  #  ckpt_manager.save()
  #  print(' -= saved! =-')
  #else:
  #  print(' -= restoring =-')
  #  ckpt.restore(ckpt_manager.latest_checkpoint)
  #  print('restored!')
      temp_result = {'b':batch,'tr_loss':loss.numpy()/batch, 'tr_rloss':rloss.numpy()/batch, 'tr_klloss':klloss.numpy()/batch}
         
  #ckpt_manager.save()
      print('\n -= testing =-')
      rloss=0;klloss=0;loss = 0
      for (batch, (target, cap_id,vid_id)) in enumerate(validds):
        if silent: print('\r {}'.format(batch), end = '\r')
        
        out = vae.test(target)
        target_out = out['reconstruction']
        rloss+=out['reconstruction_loss']
        klloss+=out['kl_loss']
        loss+=out['loss']
      rloss = rloss.numpy(); klloss = klloss.numpy(); loss=loss.numpy()
      print('param',param, 'epoch',ep,'b',batch,' ',loss/batch ,rloss/batch, klloss/batch)
      temp_result.update({'va_loss':loss/batch, 'va_rloss':rloss/batch, 'va_klloss':klloss/batch})
      results[param] +=[temp_result]

  print(results)          
  latent_data = {}     

  print("testing")
  for ds in [testds,trainds,validds]:
    for (batch, (target, cap_id,vid_id)) in enumerate(ds):
      print('\r {}'.format(batch), end = '\r')
      
      out = vae.test(target)
      target_out = out['reconstruction']
      
      for i in range(len(target)):
        latent_data[ cap_id[i].numpy()] = [vid_id[i].numpy(), out['z'][i].numpy(), out['z_mean'][i].numpy()]
        
      if batch==0:
        for i in range(5):
          print(cap_id[i].numpy(), ' ',vid_id[i].numpy(), ' ' , target[i].numpy())
          
          
          tar = target[i].numpy()
          tar_o = target_out[i].numpy()
          print(tar_o)
          
          pos= [index_dic[j-num_pos_inc ] for j in tar if j>3]#!=(-3+num_pos_inc)
          pos_o = [index_dic[j-num_pos_inc ] for j in tar_o if j>3]
          print('{} -> {} '.format(' '.join(pos), ' '.join(pos_o)))
          
      #  break 

  print(latent_data[10])
  baseaddr = 'vidfeat/'
  filename = baseaddr +dataset+'/'+dataset+'LATENT_'+str(latent_dim) # ** name msvdLATENT_50
  outfile = open(filename,'wb')
  pickle.dump(latent_data ,outfile)
  outfile.close()

#try:
#  print(' epochs {}    loss {:.3f}   r-loss {:.3f}  kl-loss {:.6f}  '.format
#    (epochs, loss.numpy(), rloss.numpy(), klloss.numpy()))
#except:
#  pass 





