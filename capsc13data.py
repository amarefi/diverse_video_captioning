#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
changes to make videos shorter:
   maxVidlen = 113/3
   ln 195
   and also setup for lstm encoder.
   disabling beam search. 

change to lstm: lines: 32,36,41,296 ,, main: 278,236,208,

* latentPOS : test and valid data are filled with zero 

"""
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
from collections import defaultdict 
import logging, os, json
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
#tf.get_logger().setLevel('ERROR')



from sklearn.model_selection import train_test_split
import numpy as np
from capsc13dataset import *




debug = False 
#BATCH_SIZE =64
max_length = 32 
dim = 2048
dim1 = 6; dim2= 300;
maxVidlen =38 # 38 #113  # msvd 225 but 113 is ok 
maxI3Dlen = 20
BUFFER_SIZE = 1000
nptype = '.npz'##
sampling_rate = [3]

#latentPOS = 'vidfeat/captionPOSlatent'
latentPOS_test_addr = 'vidfeat/captionPOSlatent_test'
POSaddr = 'vidfeat/captionPOS'

#id = [int(v['video_id'][5:]) for v in annotations['sentences'] if int(v['video_id'][5:])<7074]
#id = [trainfile +v['video_id'][5:]+'.npy' for v in annotations['sentences'] if int(v['video_id'][5:])<endvid]

# ??? Find the maximum length of any caption in our dataset 
def calc_max_length(tensor):
    t = [len(t) for t in tensor]
    if debug: print('  ', tokenizer.sequences_to_texts([tensor[np.argmax(t)]]) )
    t = max(t)
    if debug: print( 'longest sentence:', t, 'but i use max_length', max_length)
    
    return 
    


def caption_process(path,top_k = 12000):
  # Choose the top 5000 words from the vocabulary
  
  cap_list_train = Train['cap_list']
  cap_list_valid = Valid['cap_list']
  cap_list = cap_list_train+cap_list_valid
  #top_k = 12000
  tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
  tokenizer.fit_on_texts(cap_list)
  train_seqs = tokenizer.texts_to_sequences(cap_list)
  calc_max_length(cap_list)
  
  # tokenizer.word_index['<start>'] == 3 , <unk>:1 

  tokenizer.word_index['<pad>'] = 0
  tokenizer.index_word[0] = '<pad>'

  print(list(tokenizer.word_index.keys())[:10])
  print(len(tokenizer.word_index.keys()))
  
  
  # Create the tokenized vectors/ i think there is no diff with above
  train_seqs = tokenizer.texts_to_sequences(cap_list_train)

  # Pad each vector to the max_length of the captions
  # If you do not provide a max_length value, pad_sequences calculates it automatically
  cap_vector_train = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post', maxlen=max_length, truncating='post')
  # (59800, 61)
  
  train_seqs = tokenizer.texts_to_sequences(cap_list_valid)
  cap_vector_valid = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post', maxlen=max_length, truncating='post')
  
  Train['cap_vector'] = cap_vector_train
  Valid['cap_vector'] = cap_vector_valid
  
#  temp = [t for tt in Train['cap_list_single'] for t in tt]
#  temp = tokenizer.texts_to_sequences(temp)
#  temp = tf.keras.preprocessing.sequence.pad_sequences(temp, padding='post', maxlen=max_length, truncating='post')
#  Train['cap_vector_single'] = np.reshape(temp, (-1,20,temp.shape[-1]))
  Train['cap_vector_single'] = np.zeros((len(Train['cap_list_single']),20,32),dtype='int32')
  
#  temp = [t for tt in Valid['cap_list_single'] for t in tt]
#  temp = tokenizer.texts_to_sequences(temp)
#  temp = tf.keras.preprocessing.sequence.pad_sequences(temp, padding='post', maxlen=max_length, truncating='post')
  # dummy!
  Valid['cap_vector_single'] = np.zeros((len(Valid['cap_list_single']),20,32),dtype='int32') #np.reshape(temp, (-1,20,temp.shape[-1]))
  
  vocab_size = len(tokenizer.word_index) + 1
  print('vocab_size', vocab_size)
  
  
  
  
#  embeddings_index = {}
#  f = open(path)
#  for line in f:
#      values = line.split(' ')
#      word = values[0] ## The first entry is the word
#      coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
#      embeddings_index[word] = coefs
#  f.close()
#  print('tokenizer len',len(list(tokenizer.word_index.keys())))
#  glove_embed = []
#  co=0
#  #print(np.array(list(embeddings_index.values())).shape)
#  #print(np.mean(list(embeddings_index.values()),0), np.var(list(embeddings_index.values()),0))
##  print(np.max(Train['cap_vector'])) 11999
#  for wd in list(tokenizer.word_index.keys())[:top_k]:
#    if wd not in embeddings_index: 
#      temp =np.count_nonzero(Train['cap_vector'] == tokenizer.word_index[wd]) 
#      if temp>10:
#        print(wd,'not included. #',temp); 
#      glove_embed+= [np.random.normal(0,0.5,(200))]
#      co+=1
#    else:
#  #    print(wd)
#      glove_embed+= [embeddings_index[wd]]
#  del embeddings_index
#  print(co,'not in GloVe')
  glove_embed = ''

  return tokenizer, vocab_size , glove_embed #,np.array(glove_embed)




# Calculates the max_length, which is used to store the attention weights
#print('train')
#max_length = 40 # = calc_max_length(train_seqs)# 74 is max sentence length for train, 61 for test. 
#In [6]: h                                                                                                                    
#Out[6]: 
#(array([70757, 58156,  9327,  1636,   266,    37,    17,     3,     0, 1]),
# array([ 4., 11., 18., 25., 32., 39., 46., 53., 60., 67., 74.]))




# ============================================================


## Split the data into training and testing
#  Create training and validation sets using an 80-20 split
# also shuffling the data 
#id_list_train, id_list_valid, cap_vector_train, cap_vector_valid = train_test_split(id_list,
#                                                                  cap_vector,
#                                                                  test_size=0.05,
#                                                                  random_state=0)

# print('train and validation' ,len(id_list_train), len(cap_vector_train), len(id_list_valid), len(cap_vector_valid))


# ## Create a tf.data dataset for training
#  Our images and captions are ready! Next, let's create a tf.data dataset to use for training our model.


#vocab_size = len(tokenizer.word_index) + 1
#num_steps = len(img_name_train) // BATCH_SIZE
# # Shape of the vector extracted from InceptionV3 is (64, 2048)
# # These two variables represent that vector shape
# features_shape = 2048
# attention_features_shape = 64


if debug: 
  print('input vocab size', vocab_size, ' top_k', top_k, np.max(cap_vector)+1)
  print('train shape',cap_vector.shape, 'test shape',cap_vector_test.shape)
  print('\nA sample  ', tokenizer.sequences_to_texts([cap_vector[2]]),'\n',cap_vector[2] ,'\n'	)



# ------------------- load msrvtt pos
# pos for all sentences, save, use for model 
try:
  infile = open(POSaddr,'rb')
  latentPOS_test = pickle.load(infile)
  infile.close()   
except:
      
  outfile = open(POSaddr,'wb')
  pickle.dump(latentPOS_test,outfile)
  outfile.close()   
  print('POS saved')





# ------------------- load latentPOS data
latentPOS = {}
k = latentPOS.keys()
for i in range(20*10000):
  if i not in k:
    latentPOS[i] =[0,np.zeros((1), dtype = 'float32'), np.zeros((1), dtype = 'float32')]  
latent_dim = [0]

def load_latentPOS(path):
#  latent_dim = 50
  infile = open(path,'rb')
  temp  = pickle.load(infile)
  infile.close() 
  for k,v in temp.items():
    latentPOS[k]=v
  # all sentence poses are saved
  k = temp.keys()
  for i in range(20*10000):
    if i not in k:
      latentPOS[i] =[0,np.zeros((latent_dim[0]), dtype = 'float32'), np.zeros((latent_dim[0]), dtype = 'float32')]  


##print(latentPOS)
## store some sample latentPOS for testing 
#try:
#  infile = open(latentPOS_test_addr,'rb')
#  latentPOS_test = pickle.load(infile)
#  infile.close()   
#  for i,v in enumerate(latentPOS_test):
#    if debug: print(i, " : ", v['sentence'])
#  num_template_pos = len(latentPOS_test)
#  if debug: print(num_template_pos)
#except:
#  num_template_pos = 6
#  latentPOS_test = []
#  for i in range(num_template_pos):
#    d = {}
#    v = id_list_train[i]
#    print(v)
#    j = int(v[-1])
#    l = latentPOS[j]
#    print(l)
#    d['cap_id'] = j
#    d['vid_id'] = l[0]
#    d['z']= l[1]
#    d['z_mean']=l[2]
#    s = [ tokenizer.index_word[w] for w in cap_vector_train[i] if tokenizer.index_word[w] not in ['<pad>', '<start>', '<end>']]
#    s = ' '.join(s)
#    d['sentence']=s 
#    print(d)
#    latentPOS_test+=[d]
#      
#  outfile = open(latentPOS_test_addr,'wb')
#  pickle.dump(latentPOS_test,outfile)
#  outfile.close()   
#  print('latentPOS_test saved')

#zzz= []
#for k, v in latentPOS.items():
#  print('\r', k,end='\r')
#  zzz += [v[1]]
#zzz = np.array(zzz)
#print(zzz.shape, ' Mean and Var z')
#print(np.mean(zzz, axis= 0))
#print(np.var(zzz,axis=0))

#try:
#  pass
#except:
#  print("latentPOS loading error")
#  latentPOS = {i:[0,np.zeros((latent_dim), dtype = 'float32'), np.zeros((latent_dim), dtype = 'float32')] for i in range(20*10000)}
# fill undefined with zero.




# Load the numpy files
def map_func(paths, cap):
   img_name = paths[6]; 
    #infile = open(img_name,'rb') ### TODO npy is better
    #img_tensor = pickle.load(infile)
    #infile.close()   
    #return np.array(img_tensor), cap
    
   whole = np.load(img_name)
   img_tensor = whole['res_2048'][::sampling_rate[0]]#.decode('utf-8'))+'.npy') # decode made ERROR!! when testing
   if 'resnet_1000' in conf['features']:
#     img_name = paths[7]
     temp = whole['res_1000'][::sampling_rate[0]]
     img_tensor = np.concatenate([img_tensor, temp], axis = -1)
   if len(img_tensor)>maxVidlen: 
     img_tensor = img_tensor[:maxVidlen]
   t = np.zeros((maxVidlen, img_tensor.shape[-1]))
   t[:len(img_tensor)] = img_tensor
   mask = np.zeros((maxVidlen))
   mask[:len(img_tensor)] = 1
   
   if 'i3d' in  conf['features']:
#     i3d_name = paths[1]
     i3d_tensor = whole['i3d'][np.newaxis,:]
     mask2= np.array([1])
   elif 'i3d_clips' in  conf['features']:
#     i3d_name = paths[6]
     temp = whole['i3d_clips']
     if len(temp)>maxI3Dlen: 
       temp = temp[:maxI3Dlen]
     i3d_tensor = np.zeros((maxI3Dlen, temp.shape[-1] ))
     i3d_tensor[:len(temp)] = temp
     mask2 = np.zeros((maxI3Dlen))
     mask2[:len(i3d_tensor)] = 1
   else:
     i3d_tensor = np.zeros((1,400))
     mask2= np.array([1])
   
   name = img_name.split(b'/')[-1].split(b'.')[0]
   
#   print('latentPOS[int(paths[5])][1]',int(paths[5]),latentPOS[int(paths[5])][1])
   
   return t.astype('float32'),mask.astype('bool'),\
    i3d_tensor.astype('float32'), mask2.astype('bool'), cap, \
    int(paths[4] ),int(paths[5]), latentPOS[int(paths[5])][1][np.newaxis,...]

def map_func_trnsf(paths, cap):
   img_name = paths[2];  # now i just use fRCnn features. 
    #infile = open(img_name,'rb') ### TODO npy is better
    #img_tensor = pickle.load(infile)
    #infile.close()   
    #return np.array(img_tensor), cap
   img_tensor = np.load(img_name)['data'][::3]#.decode('utf-8'))+'.npy') # decode made ERROR!! when testing
   t = np.zeros((maxVidlen, dim1,dim2))
   t[:len(img_tensor)] = img_tensor
   mask = np.zeros((maxVidlen))
   mask[:len(img_tensor)] = 1
   i3d_name = paths[1]
   

   i3d_tensor = np.load(i3d_name)['data'][np.newaxis,:]
   name = img_name.split(b'/')[-1].split(b'.')[0]
   
   return t.astype('float32'),i3d_tensor.astype('float32'), cap.astype('int32'), mask.astype('bool'),\
      int(name ),int(paths[4]), latentPOS[int(paths[4])][1][np.newaxis,...] # 1:z # 2: z_mean


def map_func_single(img_name, cap):
   img_tensor = np.load(img_name)
   
   name = img_name.split(b'/')[-1].split(b'.')[0]
   return img_tensor, cap,np.array([True]),int(name )
   
# In[ ]:

## from_tensors may be useful later TODO
#trainds = tf.data.Dataset.from_tensor_slices((id_list_train, cap_vector_train))
## Use map to load the numpy files in parallel
#trainds = trainds.map(lambda item1, item2: tf.numpy_function(
#          map_func_single, [item1, item2], [tf.float32, tf.int32, tf.bool, tf.int64]), # float32
#          num_parallel_calls=tf.data.experimental.AUTOTUNE)# 32 1080
## Shuffle and batch
#trainds = trainds.batch(BATCH_SIZE)
##trainds = trainds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)# 20 1080

#trainad = 'vidfeat/bch_2048_train/'
## saving batched and shuffled data in order to reduce access time.
#for (batch, (img_tensor, target)) in enumerate(trainds):
#   print('\r','saving train batch', batch, end="\r")
#   np.savez(trainad+str(batch), img_tensor=img_tensor.numpy(), target=target.numpy())

#for i in range(100):
#   fl = np.load(trainad+str(i)+'.npz')
#   img_tensor = fl['img_tensor']; target = fl['target']
#   print('\r','loading train batch', i, end="\r")


#import time

#li  =[]
#s2 = time.time()
#for (batch, (img_tensor, target)) in enumerate(trainds):
#   t = time.time() - s2
#   li +=[t/BATCH_SIZE]
#   print ('\nTime taken for for batch{} {:.3f} sec    {:.3f}\n'.format(batch,t, np.mean(li)) , img_tensor.shape)  
#   time.sleep(0.2)
#   s2 = time.time()
#   
#   if batch==200:
#      break

#print(np.mean(li), np.mean(li)*64)
# 1080: auto: 0.82 ;; 32: 0.256;; 128: 0.8;; 32:0.43
# 2080 (32..,20):1.17
# 2080 vectorized:


#testds = tf.data.Dataset.from_tensor_slices((id_list_test, cap_vector_test))
## Use map to load the numpy files in parallel
#testds = testds.map(lambda item1, item2: tf.numpy_function(
#          map_func_single, [item1, item2], [tf.float32, tf.int32, tf.bool, tf.int64]), # float32
#          num_parallel_calls=tf.data.experimental.AUTOTUNE)
## Shuffle and batch
#testds = testds.batch(BATCH_SIZE)
#testds = testds.interleave(num_parallel_calls=tf.data.experimental.AUTOTUNE)
#testds = testds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# ds: 
#In [36]: a.shape                                                                                                             
#Out[36]: TensorShape([64, 113, 2048])
#In [37]: b.shape                                                                                                             
#Out[37]: TensorShape([64, 74])

def create_dataset(id_list, caption_vector_list,map_func,batch_size,postype = tf.float32):
   ds = tf.data.Dataset.from_tensor_slices((id_list, caption_vector_list))
   # Use map to load the numpy files in parallel
   ds = ds.map(lambda item1, item2: tf.numpy_function(
             map_func, [item1, item2], [tf.float32,tf.bool, tf.float32,tf.bool,  tf.int32, 
               tf.int64,tf.int64, postype]), # float32
             num_parallel_calls=tf.data.experimental.AUTOTUNE)
   # Shuffle and batch
#   ds = ds.shuffle(len(id_list)) needs a huge memory
   ds = ds.batch(batch_size)
   return ds



#   ds = tf.data.interleave(ds,num_parallel_calls=tf.data.experimental.AUTOTUNE)
#test_dict = {}
#for i,val in enumerate(img_name_test):
#    if val not in test_dict.keys():
#        test_dict[val]=[]
#    test_dict[val]+=[cap_test[i]]
# not needed. 
# to have a dictionary of file name and captions, just use ref_dict with appropriate range and this formula
# trainfile/testfile +annot['video_id'][5:]+'.npy'
 






