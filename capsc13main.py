#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention; 
resnet features for each 8 frames are extracted. 
attention model using them(maxVidlen = 113) 
or using just 20
// i used a void attention encoder, in order not to change syntax of old code (attention_Encoder_void)


c9main c9data c9model c9eval (version is 9)
c9model: parameters and model (from ... import *)

eval <- model
main <- eval, model, data

# in single(average of resnet features from video frames) : lstm_dim is fc_dic of encoder
set: maxVidlen=1; new Encoder_single_frame; npz:npy(2+1in eval); map_func:...; 





Resnet152v2 0.25fps(8frames each)
featre extraction using ResNet152-v2 from MSRVTT frames
.. layers[-1, -2] 1000,2048 ?
extraction method in preprocess2.py in server (seperating videos draft in preprocesstemp.py)

later: 
    compare outputlayer -1 & -2; frames: 8 & 16
refaddr
test train.json
divide data. 

======videos
test: 7010..9999
maxvidlen 113
(n, 2048) different n!!

train
maxvidlen 113 !! why?

======terminal
%run att10main.py server 10 
======





TODO
test layers and fps 
check todo in code
debug network and bprop 

use ipython embed() for debugging

for validation: 
    put params in lists2 and in FOR. add also in output names{}
for plotting, select the appr Checkpoint folder
"""

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# import matplotlib.pyplot as plt
#from tqdm import tqdm
import re
import numpy as np
import time
from glob import glob
# from PIL import Image
import pickle
ar = np.array
np.set_printoptions(precision=6)
import argparse
from IPython import embed
from datetime import datetime

#%%
parser = argparse.ArgumentParser()
parser.add_argument("where",help="directory") # 'home' or 'server'
parser.add_argument("epochs", type=int) # 'home' or 'server'
parser.add_argument("--lstmdim", type=int, default=128) # it's optional
parser.add_argument("--posepoch", type=int, default=7) # it's optional
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--silent", action="store_false", default=True)
parser.add_argument("--notrain", action="store_true", default=False)
#parser.add_argument("--pos", action="store_true", default=False)
parser.add_argument("--snrio", type=str,default = 'normal')
parser.add_argument("--dataset", type=str,default = 'msrvtt')
parser.add_argument("--lastmodel" ) 

args = parser.parse_args()
Posepoch = args.posepoch
dataset = args.dataset
print('dataset', dataset)
if args.cpu:
    print('using cpu not gpu', args.cpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#if args.pos:
#  scenario = 'pos'
#else:
#  scenario = ''
scenario = args.snrio
wh = args.where
silent = args.silent
notrain = args.notrain
lastmodel = args.lastmodel
if lastmodel == 'nosave' : 
  print('nosave')
  lastmodel = False 
  
print(wh)
if wh=='server':
#    test_annot_json = '../../dataset/msrvtt/test_videodatainfo.json' ##
#    train_annot_json = '../../dataset/msrvtt/train_val_videodatainfo.json'
    import tensorflow as tf
    
    if not args.cpu:
      gpu_devices = tf.config.experimental.list_physical_devices('GPU')
      tf.config.experimental.set_memory_growth(gpu_devices[0], True)
      print("memory growth enabled")

    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
#    base = '/home/amirhossein/Desktop/implement/dataset/meteor-1.5/'

elif wh=='home':
    test_annot_json = '../../../datasets/msrvtt/test_videodatainfo.json'
    import tensorflow as tf
    # tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
#    base = '/home/amir/Desktop/thesis/datasets/meteor-1.5/'

else:
    print('where?')
    exit(0)
EPOCHS = args.epochs
# if args.lstmdim:
lstm_dim = args.lstmdim
# else:
    # lstm_dim=128

print("loading modules")
from capsc13model import *
print("loading modules")
from capsc13data import *
print("loading modules")
from capsc13eval import *

conf['snrio'] = scenario
print(conf)
# snrio: normal slen pos e2epos trainpos traine2epos slen_cte
# test: final finalscorer model noise ref 
# base model: normal+final 
# pos and e2epos (dataset: model,noise,ref), final+scorer
# training_pos model: capscorer phase:False - test mode: noise // saved models can be tested with trainpos ..

#In [14]: target
#<tf.Tensor: id=31, shape=(64, 31), dtype=int32, numpy=
#In [15]: batch
#Out[15]: 0
#In [16]: img_tensor
#<tf.Tensor: id=30, shape=(64, 27, 200), dtype=float64, numpy=

# ===================================================

embedding_dim = 256;  lstm_dim = [500];  #units = [300]#300,450,600
fc2_type = ['softmax']#, 'linear'
layernorm = [False];  dropout = [0.4];  beam = [1];  RL = [0]
trnsf_box_enc = [True]
trnsf_score_enc = [0]
trnsf_frame_enc = [0]# True # Now a dummy variable for testing again!
pos_do = 0
if conf['snrio'] in ['normal']:
#  embedding_dim = 200
  trnsf_frame_enc = [1]  
#trnsf_frame_enc = [64]#  Now a dummy variable for batchsize!
capsc_dim = [200];  #lrate = 0.0003; sampling_rate=[2]
#lstm_dim = [400]
#units = [320]
#fc2_type = ['softmax']
if dataset == 'msvd':
  units = [150]#[50,150,300,450]#[300]
  lrate = 0.001; sampling_rate=[2]
  trnsf_score_enc = [1]#[0.0001,0.0003,0.001]

if dataset == 'msrvtt':
  units = [800]#[100,250,500,800]
  lrate = 0.0003;  sampling_rate=[3]
  trnsf_score_enc = [1] #[0.0001,0.001,0.0003]

# ===================================================

if conf['snrio'] in ['traine2epos', 'trainpos','slen']:
  conf['test'] = 'noise'
  print(conf)
#  pos_do = 0

if dataset == 'msvd':
  dataset_config = msvd_config
  pos_dic, index_dic = msvd_caption(msvd_config)
  top_k = 8000; 
if dataset == 'msrvtt':
  dataset_config = msrvtt_config
  msrvtt_config['cap_data']='vidfeat/msrvtt/msrvtt_captions_valid7percent'
  pos_dic, index_dic = msrvtt_caption(msrvtt_config,0.0709)
  top_k = 12000
print('dataset_config[cap_data]',dataset_config['cap_data'])

get_caption_list_train(dataset_config)# with shuffling
get_caption_list_train(dataset_config,True)
get_caption_list_test(dataset_config) # cap_list_test is empty

if conf['snrio'] in ['slen']:
  trnsf_frame_enc = [0] # pos_do
  latent_dim[0] =1
  print('sentence length ... making latentPOS')
  for dic in [Train['dict'],Valid['dict'] , Test['dict']]: # Train just needed. but to make dataformat int64
    for k,v in dic.items():
      for vv in v:
        temp = vv[0].split(' ')
        latentPOS[vv[2] ]= [0,np.array([len(temp)/20],dtype='float32')]
  print('slen latentPOS', temp, vv[2],len(temp))
if conf['snrio'] in ['e2epos','traine2epos']:
  print('e2e or train')
  latent_dim[0] = 50
  for dic in [Train['dict'], Valid['dict'] , Test['dict'] ]:
    for k,v in dic.items():
      for vv in v:
        latentPOS[vv[2]]= [0,vv[3]] # it's used in creating datasets # it's buggy!!! 
if conf['snrio'] in ['pos','trainpos']:
  latent_dim[0] = 50
  path = dataset_config['latentPOS_path']; print('latent pos path',path, latent_dim)
  load_latentPOS(path)
  print('latentPOS[1]',latentPOS[1])

print('#videos train:',len(Train['dict'].keys()),' test:',len(Test['dict'].keys()),' valid:',len(Valid['dict'].keys()), 'embed dim',
  embedding_dim )

tokenizer, vocab_size,glove_embed = caption_process( '../../dataset/glove.6B.200d.txt',top_k = top_k)

batch_size = 64
test_epoch = 0
if conf['snrio'] in ['pos','e2epos','trainpos','traine2epos','slen']: 
  testbatchsize = 1
  conf['features'] = ['i3d']
#  glove_embed = ''
  trnsf_frame_enc = [0]  
else:
  testbatchsize = 64

if conf['snrio'] in ['trainpos']:
  trnsf_frame_enc=[0,0.25,0.5]

if conf['snrio'] in ['e2epos','traine2epos']: 
  postype = tf.int64 
else: 
  postype = tf.float32
  
trainds = create_dataset(Train['id_list'], Train['cap_vector'],map_func,batch_size,postype)
validds = create_dataset(Valid['id_list_single'], Valid['cap_vector_single'],map_func,testbatchsize,postype)
trainds2 = create_dataset(Train['id_list_single'], Train['cap_vector_single'], map_func,1,postype)

#repeat_remainder_batch(Test, testbatchsize)
testds = create_dataset(Test['id_list_single'], Test['cap_vector_single'],map_func,testbatchsize,postype)
print('train and validation' ,len(Train['id_list']), len( Train['cap_vector']), len(Valid['id_list']), len(Valid['cap_vector']))


# ===================================================

L = [lstm_dim, units, fc2_type,layernorm,dropout,beam,RL,trnsf_box_enc,trnsf_score_enc,trnsf_frame_enc,capsc_dim ]
# L = [[1,2,3],[4,5,6],[7,8,9,10]]

outlist = ablation_list(L)
print('iter items', L)

bestresult={'15':0,'50':0,'85':0}


now = datetime.now()
date_time = now.strftime("%Y%m%d%H%M")
print(date_time)
#### 
for lstm_dim,units,fc2_type,layernorm,dropout,beam,RL,trnsf_box_enc,trnsf_score_enc,trnsf_frame_enc,capsc_dim in outlist:#,[128,1024],[256,512],[256,1024]  ,[128,512,3],[128,512,5]
#lstm_dim=512;units=448;fc2_type='softmax'
  print('lstm dim.{}, units.{}, fctype.{}, layernorm.{}, dropout.{}, beam.{}, RL.{}, Transformer: Box:{}, score{} frame{} encodings  capsc_dim {} '.format(lstm_dim,units,fc2_type,layernorm,dropout,beam,RL,trnsf_box_enc,trnsf_score_enc,trnsf_frame_enc,capsc_dim))
  param = '_{}__{}_{}_{}_{}_{:02}_{}_{}_{}_{}_{}_{}'.format(date_time,lstm_dim,units,fc2_type,layernorm,dropout*100,beam,RL,trnsf_box_enc,trnsf_score_enc,trnsf_frame_enc,capsc_dim)
  param0= param
  
  model = ['Trnsf','EncDec','Attn'][1]
  shuffle_each_ep= True

  feature_do = 0
#  if conf['snrio'] in ['normal']:
#    if trnsf_frame_enc==2:
#      conf['features'] = ['i3d']
#      print(conf)
#    if trnsf_frame_enc==3:
#      conf['features'] = ['i3d_clips']
#      print(conf)
#    if trnsf_frame_enc==4:
#      conf['features'] = ['resnet_1000','i3d']
#      print(conf)
  
    
#  if trnsf_score_enc==1:
#    glove_embed=''
#    embedding_dim = 256
#    print('not using glove - 256')
#  if trnsf_score_enc==2:
#    embedding_dim = 200
      
        
  # datasets with appropriate batchsize and mapfunc can be defined here 
#  if model == 'EncDec':
##    encoder = attention_Encoder_void(maxVidlen, dim,lstm_dim)
##    encoder = Encoder3mask(maxVidlen, dim,lstm_dim)
 #   encoder = Encoder4(maxVidlen, units,lstm_dim,0.4, use_pos) # units<-dim
#    if 'i3d_clips' in conf['features']: 
  i3dencoder = Encoder4(units,units, 0.4, False)
#    else:
#      i3dencoder = None
  if True :
    encoder = Encoder4(units,units, 0.4, False)#best 500,500,0.4
  else:
    print('SDP attention encoder')
    encoder = Encoder5_SDP(maxVidlen, 500,1000, 0.4, use_pos)#best 500,1000, 0.4
##    encoder = Encoder_single_frame(maxVidlen, dim,lstm_dim)
##    encoder = Encoder_single2(maxVidlen, dim,lstm_dim,0.4)
##    decoder = attention_Decoder(embedding_dim, units, top_k, fc2_type)
  #decoder = RNN_Decoder(embedding_dim, units, top_k, fc2_type,layernorm,0.4,0,'lstm_cell')#dropout, recurrent_dropout, cell_type
  
  decoder = RNN_Decoder (embedding_dim, 400, top_k, fc2_type , layernorm, 0.4,0,feature_do,'lstm_cell',glove_embed)# best 400,0.4,lstm_cell  
    
  caption_scorer = None 
  if conf['snrio'] in ['pos', 'e2epos']:
    conf['test']='finalscorer'
    caption_scorer = Caption_scorer( lstm_dim=capsc_dim , dropout=0.6)
    optimizer_sc = tf.keras.optimizers.Adam(learning_rate=0.001)#,amsgrad = True,epsilon=0.1

  if fc2_type=='softmax':
      loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=False, reduction='none')
  elif fc2_type== 'linear':    
      loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction='none')

  
#  optimizer = tf.keras.optimizers.Adam(learning_rate=0.005) #  default 0.001
#  lrate = trnsf_score_enc
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=lrate)


  # ## Checkpoint
  if lastmodel:
    checkpoint_path = "./checkpoints/train/{}/ckpt".format(lastmodel)
    try:
      ckpt = tf.train.Checkpoint(encoder=encoder, i3dencoder = i3dencoder,
                           decoder=decoder,
                           optimizer = optimizer)
      print('loading model with i3d encoder')
    except:
      ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
      print('not loading model with i3d encoder')
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print( ' model {} loaded '.format(lastmodel))
#    lastmodel = False
  else:
    if conf['snrio'] in ['slen', 'trainpos','traine2epos']:
      checkpoint_path1 = "./checkpoints/train/{}_15/ckpt".format(date_time)
      ckpt1 = tf.train.Checkpoint(encoder=encoder, i3dencoder = i3dencoder,
                                 decoder=decoder,
                                 optimizer = optimizer)
      ckpt_manager1 = tf.train.CheckpointManager(ckpt1, checkpoint_path1, max_to_keep=1)
      
      checkpoint_path2 = "./checkpoints/train/{}_50/ckpt".format(date_time)
      ckpt2 = tf.train.Checkpoint(encoder=encoder, i3dencoder = i3dencoder,
                                 decoder=decoder,
                                 optimizer = optimizer)
      ckpt_manager2 = tf.train.CheckpointManager(ckpt2, checkpoint_path2, max_to_keep=1)
      
      checkpoint_path3 = "./checkpoints/train/{}_85/ckpt".format(date_time)
      ckpt3 = tf.train.Checkpoint(encoder=encoder, i3dencoder = i3dencoder,
                                 decoder=decoder,
                                 optimizer = optimizer)
      ckpt_manager3 = tf.train.CheckpointManager(ckpt3, checkpoint_path3, max_to_keep=1)
  
    else:
      checkpoint_path = "./checkpoints/train/{}_{}/ckpt".format(date_time,trnsf_frame_enc)
      ckpt = tf.train.Checkpoint(encoder=encoder, i3dencoder = i3dencoder,
                                 decoder=decoder,
                                 optimizer = optimizer)
      ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
  
  
#    start_epoch = 0
#    if ckpt_manager.latest_checkpoint:
#      start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  
  pos_do = trnsf_frame_enc
  if model == 'EncDec':
    if conf['snrio'] in ['e2epos','traine2epos']:
      print('pos drop out', pos_do)
      num_pos = len(pos_dic.keys()) + num_pos_inc
      posencoder = PosEncoder(max_length, 50,20, num_pos,100,latent_dim[0]) #fc_dims,embed_dim, num_pos,lstm_dim,latent_dim50
      train_obj = training(decoder,encoder,optimizer,tokenizer,RL, posencoder=posencoder,i3dencoder = i3dencoder,pos_do = pos_do)
    else:
      print('pos drop out', pos_do)
      train_obj = training(decoder,encoder,optimizer,tokenizer,RL,i3dencoder = i3dencoder,pos_do = pos_do)
    test_obj = testing(decoder,encoder,loss_object,tokenizer,beam,cap_sc = caption_scorer,i3dencoder = i3dencoder)
  elif model == 'Trnsf':
    train_obj = training_trnsf(trnsf,optimizer, loss_object,RL)
    test_obj = testing_trnsf(trnsf, loss_object,beam)


  # turn list (containing 20 ids and sentences for each) to dict(vid: caption list)
 
 
#  current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
#  train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
#  train_log_dir2 = 'logs/gradient_tape/' + current_time + '/grads'
#  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#  train_summary_writer2 = tf.summary.create_file_writer(train_log_dir2)
##    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

  
  
  #logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
  split_num = 5
#    dataset_length = (len(id_list_train)-1)//batch_size
#    ds_length
  
  #for epoch in range(start_epoch, EPOCHS):
  max_meteor = 0
  epoch_data = {}
  test_list = [0]
  if conf['snrio'] in ['pos', 'e2epos','trainpos','traine2epos','slen']: # either of two works!
#      test_list = np.arange(num_template_pos)
    test_list = [0]
  if conf['snrio'] in ['slen_cte']:
    test_list = np.arange(10,18,3,dtype = 'float32')
  print(test_list)
#    for length in range(14,18,2):
  for length in test_list:
      epoch_data[length]=[]
  last_batch = 0; batch=0
  epoch_time=[]
  for epoch in range(EPOCHS):
    start = time()
    total_loss = 0
    tlist = []
    s2 = time()
#        for (batch, (img_tensor, target,mask,vid_id)) in enumerate(trainds.shard(num_shards=shard_num, index=epoch%shard_num)):
    t_loss = tf.Variable(0,dtype=tf.float32); kl_loss= tf.Variable(0,dtype=tf.float32);
    
    if shuffle_each_ep :
      temp = list(zip(Train['id_list'],Train['cap_vector']))
      random.shuffle(temp)
      id_list_train = [t[0] for t in temp]
      cap_vector_train = [t[1] for t in temp]
      trainds = create_dataset(id_list_train, cap_vector_train,map_func,batch_size,postype)
        
    show_sent = defaultdict(list)
    if not lastmodel:
      for batch,features  in enumerate(trainds): #img_tensor,mask,i3d_tensor,maski3d, target,vid_id,cap_id,pos
  #            if batch%split_num!=epoch%split_num :
  #                continue
        s3 = time() - s2; tlist+=[s3]
        if silent: 
          print ('\r','{}  ===== Time taken for for batch{} {:.3f} sec  mean{:.3f}    l {:.3f}  kl {:.5f}'.\
            format(epoch+1, batch,s3,np.mean(tlist), t_loss.numpy() , kl_loss.numpy()) , \
            features[0].shape, end='\r')  
        
        if features[0].shape[0]!=batch_size : break
        s2 = time()
        # last batch is of size 48, not 64 
        t_loss,osent,kl_loss = train_obj.train_step(features, loss_object,epoch)#img_tensor,i3d_tensor, target,mask,vid_id,pos
        total_loss += t_loss
        
  #            if epoch==0:
  #      with train_summary_writer.as_default():
  #        tf.summary.scalar('loss', t_loss, step=batch+epoch*last_batch)
  #      if batch%20==0:
  #        with train_summary_writer2.as_default():              
  #          tf.summary.histogram('grads_pos_0', grads[0][0], step=batch//20+epoch*last_batch)
  #          tf.summary.histogram('grads_pos_1', grads[0][1], step=batch//20+epoch*last_batch)
  #          tf.summary.histogram('grads_pos_2', grads[0][2], step=batch//20+epoch*last_batch)                                
  #          tf.summary.histogram('pos', grads[1], step=batch//20+epoch*last_batch)       
  #          tf.summary.histogram('genpos', grads[2], step=batch//20+epoch*last_batch)                           
  ##          tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        

        if batch==100 and notrain :break
#            print ('\nTime taken for for trainstep {:.3f} sec\n'.format(time.time() - s2))    

#        if epoch==EPOCHS-1 :
        vid_id = features[5].numpy()
        templist = [1000,1001,1003] if dataset=='msvd' else [6000,6001,6002,6003,6004]
        for i,v in enumerate(vid_id):
          if v in templist:
#            print('\r bing \r', end='')
            
            sent = osent[i].numpy()
            sent = ' '.join([  tokenizer.index_word[t] for t in sent if tokenizer.index_word[t] not in ['<start>', '<end>','<pad>']])# 
            show_sent[v] += [sent] 
      print('training encdec model samples')
      for k,v in show_sent.items():
        print(k, v)

    # training encoder/decoder done! now we have a fixed model 
    print('')
    last_batch = batch//20
    total_loss = total_loss/len(Train['id_list'])

    # now training the caption scorer 
    random_count = batch_size
    if conf['snrio'] in ['pos', 'e2epos'] :
      if epoch==0: caption_scorer.set_embed_layer( decoder.embedding )
      
      # calculating meteor for caption list 
      refaddr = 'captions/ref_cap_sc.txt'
      testaddr= 'captions/test_cap_sc.txt'
      checkaddr = 'captions/check_cap_sc.txt'
      resaddr = 'captions/result_cap_sc.txt'
      addr = [refaddr, testaddr, checkaddr,resaddr]
      
      
      cap_sc_data_addr = 'vidfeat/cap_sc/{}/'.format(dataset)
      folders = []
      for s in os.listdir(cap_sc_data_addr):
        try:
          folders +=[int(s)]
        except:
          pass
      print(folders)
      
      try:
        ds
        print('using dataset from last epoch')
      except:
        try:
          cap_sc_data_addr = cap_sc_data_addr + '{}'.format(np.max(folders)) # loading the last one! 
          print('cap_sc_data_addr',cap_sc_data_addr)
          infile = open(cap_sc_data_addr,'rb')
          cap_sc_list = pickle.load(infile)
          infile.close()   
        except:
          # making the dataset 
          cap_sc_list = []
            # using generated sentences with calculated meteor 
          print('\n using generated sentences with calculated meteor ')
          conf['test'] = 'model'
          _,t,_ = test_score(epoch+1, test_obj,addr,0,trainds , batch_size)
          cap_sc_list += [t for t in t.values() for t in t] # :)
          print('\n',len(cap_sc_list))
          print([t[1:] for t in cap_sc_list[-3:]]) # not showing features 
          del t
          
            # using random noise  # using trainds2 # random_count=batch_size to avoid error 
          print('\n using random noise  # using trainds2')
          conf['test'] = 'noise'
          _,temp,_ = test_score(epoch+1, test_obj,addr,0,trainds2 , random_count,random_count = random_count)
          cap_sc_list += [t for t in temp.values() for t in t]
    #      print(temp.keys())
          print('\n',len(cap_sc_list), '  some sentences with random pos')
          print([t[1:] for t in cap_sc_list[-150::10]])
          
            # using reference sentences with meteor = 1
          print('\n using reference sentences with meteor = 1')
          for (batch, (img_tensor,mask,i3d_tensor,maski3d, target,vid_id,cap_id,pos)) in enumerate(trainds):# trainds2->trainds
            if silent:print('\r', batch, end ='\r')
            for i in range(img_tensor.shape[0]):
              v = vid_id.numpy()[i]
              if v not in temp.keys():
                continue
              f = temp[v][0][0] # 1 video feature 
  #            print(target.shape)
              c = list(target[i].numpy()) # caption list 
#              for s in c:
              sent = ' '.join([  tokenizer.index_word[t] for t in c if tokenizer.index_word[t] not in ['<start>', '<end>','<pad>']])
              cap_sc_list += [[f, c[1:], sent, 1.0]] # removing <start> token from caption...
          print('\n',len(cap_sc_list))          
          print([t[1:] for t in cap_sc_list[-3:]])
          del temp
          cap_sc_data_addr = cap_sc_data_addr+'{}'.format(date_time)
          outfile = open(cap_sc_data_addr,'wb')
          pickle.dump(cap_sc_list ,outfile)
          outfile.close()   
    
    
    
    
        ds = caption_scorer_dataset(cap_sc_list)
        # training caption scorer 
        loss_sc =  tf.keras.losses.MeanSquaredError()#reduction='none'
      
      for e in range(Posepoch):
        capsc_loss = 0
        for (batch, (feature,caption,meteor)) in enumerate(ds):
          with tf.GradientTape() as tape:
            gmeteor = caption_scorer(feature, caption, training = True)
            loss = loss_sc(gmeteor,meteor)
#            loss *= 1.1-meteor 
#            loss = tf.reduce_sum(loss)
          capsc_loss += loss.numpy()
          if batch==10:
            print(gmeteor[:,0],'\n',  meteor,loss.numpy())
          if batch==100 and notrain :break
          if silent: print('\r ep{}'.format(e), batch, feature.shape, caption.shape, meteor.shape,'  ', loss.numpy(),  end ='\r')
          trainable_variables = caption_scorer.trainable_variables
          gradients = tape.gradient(loss, trainable_variables)
          optimizer_sc.apply_gradients(zip(gradients, trainable_variables))
        if batch==100 and notrain :
          print('training skipped')
          break
        print('capsc ep{}  totloss {}   meanl {}'.format(e, capsc_loss, capsc_loss/batch))

      conf['test'] = 'finalscorer'
    
      if epoch == EPOCHS-1:
        del ds; del cap_sc_list
#        for length in range(14,18,2):
      
    for length in test_list:
      if epoch<test_epoch:
        continue
      param = param0+'_{}'.format(length)
      print('\n',param)
      if debug: print(latentPOS_test[length]['sentence'])

      refaddr = 'captions/ref{}_{}.txt'.format(epoch+1,param)
      testaddr= 'captions/test{}_{}.txt'.format(epoch+1,param)
      checkaddr = 'captions/check{}_{}.txt'.format(epoch+1,param)
      resaddr = 'captions/result{}_{}.txt'.format(epoch+1,param)
      addr = [refaddr, testaddr, checkaddr,resaddr]
#      if conf['snrio'] in ['trainpos','traine2epos']:
#        val_meteor =0;val_data=0
#      else:
      print('\n VALIDATION ==================')
      val_data,_,_ = test_score(epoch+1, test_obj,addr,length,validds , batch_size, random_count = random_count)
      val_meteor = val_data['final']
      print('validation' , val_meteor)
#      val_meteor = 0
      # so that not delete its related test files 
      print('\n TEST ==================')
      ts,_,metdeb = test_score(epoch+1, test_obj,addr,length,testds , batch_size, random_count = random_count)
      print(ts)
      try: 
        epoch_data[length]+=[{'ep':epoch+1, 'sc':ts, 'total_train_loss':total_loss.numpy(),'val_meteor':val_meteor,'val_data':val_data}]
      except: 
        epoch_data[length]+=[{'ep':epoch+1, 'sc':ts, 'total_train_loss':total_loss,'val_meteor':val_meteor,'val_data':val_data}]
      tmp = epoch_data[length][-1]
      print("ep {} final {:.3f}     prec {:.3f} recall {:.3f} frag {:.3f}    \nsentence_len {:.3f}   trainloss {:.3f} testloss{:.3f}    validM{:.3f}"
          .format(tmp['ep'], ts['final'],ts['precision'],ts['recall'],ts['fragmentation'],
          ts['mean_len'],tmp['total_train_loss']*split_num, ts['tot_test_loss'],val_meteor))
      tmp = tmp['sc']['coco']
      print('coco:: Bleu_1 {:.3f},  Bleu_2 {:.3f},  Bleu_3 {:.3f},  Bleu_4 {:.3f}, \
        METEOR {:.3f},  ROUGE_L {:.3f},  CIDEr {:.3f} '.format(tmp['Bleu_1'], tmp['Bleu_2'], 
        tmp['Bleu_3'],tmp['Bleu_4'], tmp['METEOR'], tmp['ROUGE_L'],tmp['CIDEr']))
      # print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
      #                                      total_loss/num_steps))
#      if conf['snrio'] not in ['pos', 'e2epos','slen','trainpos','traine2epos']: break
    e_time = time() - start
    epoch_time +=[e_time]
    print ('\nTime taken for 1 epoch {:.3f} sec\n'.format(e_time))
  
    if not lastmodel :
      if conf['snrio'] in ['slen', 'trainpos','traine2epos']:
        s= val_data['coco']['selfcider']
        m = val_data['coco']['mbleu']
        me = val_data['coco']['METEOR']
        r = (s + m)*0.15*0.5+ me*(1-0.15)
        if r>bestresult['15']:
          print('checkpoint p15 posdo{} ep{}  meteor{:.3f} selfcider{:.3f} mbleu{:.3f} for valid '.format(pos_do,epoch+1,me,s,m))
          ckpt_manager1.save() 
          bestresult['15'] = r
        r = (s + m)*0.5*0.5+ me*(1-0.5)
        if r>bestresult['50']:
          print('checkpoint p50 posdo{} ep{}  meteor{:.3f} selfcider{:.3f} mbleu{:.3f} for valid '.format(pos_do,epoch+1,me,s,m))
          ckpt_manager2.save() 
          bestresult['50'] = r
        r = (s + m)*0.85*0.5+ me*(1-0.85)
        if r>bestresult['85']:
          print('checkpoint p85 posdo{} ep{}  meteor{:.3f} selfcider{:.3f} mbleu{:.3f} for valid '.format(pos_do,epoch+1,me,s,m))
          ckpt_manager3.save() 
          bestresult['85'] = r          
      else: 
        ckpt_manager.save()
 
  print(':: epoch time ', epoch_time, '  mean ', np.mean(epoch_time))
 
  del train_obj
  del test_obj
  del optimizer
  if model == 'EncDec':
      del encoder # *** these must be deleted too
      del decoder
  elif model == 'Trnsf':
      del trnsf
      
#    for length in range(14,18,2):
  for length in test_list:
    param = param0+'_{}'.format(length)
    filename = 'checkpoints/{}/result_{}'.format(date_time,param)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    outfile = open(filename,'wb')
    pickle.dump(epoch_data[length] ,outfile)
    outfile.close()
#    if conf['snrio'] not in ['pos', 'e2epos','slen','trainpos','traine2epos']: break

#    break


#l=[]  didnt work
#for (batch, (_,_,_,vid_id)) in enumerate(trainds.shard(num_shards=shard_num, index=5%shard_num)):
#   l+=list(vid_id.numpy())
#l2=[]
#for (batch, (_,_,_,vid_id)) in enumerate(trainds.shard(num_shards=shard_num, index=6%shard_num)):
#   l2+=list(vid_id.numpy())
#for temp in l:
#   if temp in l2:
#      print('error')

