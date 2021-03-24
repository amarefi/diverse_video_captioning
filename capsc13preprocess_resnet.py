#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python capsc13preprocess_resnet.py --dataset msvd --cpu

RGB instead of BGR 
crop then resize. 240:320=>240:20-300=>224:224
nohup python preprocess2.py --filename res_train_ --datasetaddr /home/amirhossein/Desktop/implement/dataset/msrvtt/msrvtt3/TrainValVideo/video{}.mp4 --s 1 --e 4000 &

msrvtt 240:320
msvd : different frame size

featre extraction using ResNet152-v2 from MSRVTT frames
.. layers[-1, -2]
.. framerate: 0.25 fps. test with 8 (test with 16 or 32 also)
input to resnet: 224,224,3)
zero padding batches(tensorflow needs constant batch size, 
  but video lengths are different. so use a batch of 32 frames and more batches for longer videos 

at last seperating features of each video. useful for tf dataset...

test: 
python preprocess2.py --filename RGB_test_res_ --datasetaddr /home/amirhossein/Desktop/implement/dataset/msrvtt/TestVideo/video{}.mp4 --s 7010 --e 9999
train:
python preprocess2.py --filename RGB_train_res_ --datasetaddr /home/amirhossein/Desktop/implement/dataset/msrvtt/msrvtt3/TrainValVideo/video{}.mp4 --s 0 --e 7009
3000


train 0..7009
test  7010..9999

=========== data[1] = featlist
>>> len(data[1])
3105

>>> data[1][1][0].shape
(50, 2048)
>>> data[1][1][1].shape
(50, 1000)

===========each video format
list( (2048,) )










# added for lab.
# config GPU=0 (not respond) - os hide gpu
# tf.resize_image tf.image.resize_images tf.image.resize
# code for showing memory usage of variables.


# PROBLEMS!
# P1: the problem(too slow) was that function loaded and removes model from GPU that takes a long time. use batches instead
#    maybe this problem was also with C3D...
#    i'll put frames in a buffer of size 100
# P2: memory killed. just use 50%
# ... now just using CPU. (runs multithreaded) didnt work
# ... as others noted, memory is not released after sess.close. https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution --- didnt work. (input doesnt pass properly or some problem w tf)
# ... feed network once. model compile or tf run... but now lets can buffer be 1000
# ..... i'm suspicious about 'steps' in predict...

"""
import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
ar = np.array
import cv2,pickle,os,argparse
from glob import glob 
# from keras.models import Sequential, Model
#from tqdm import tqdm
from tensorflow.keras.applications import ResNet152V2 as ResNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--dataset", type=str,required = True)
#parser.add_argument("--datasetaddr", type=str,required = True)
#parser.add_argument("--s", type=int,required = True)
#parser.add_argument("--e", type=int,required = True)

args = parser.parse_args()
dataset = args.dataset 
if args.cpu:
    print('using cpu not gpu', args.cpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
deb = 0

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dsrate = 8
bufsize = 32
RGB = True # convert BGR to RGB 


import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

    
# ============================ check frame sizes  // all 230,320,3
#baseaddr = '/home/amirhossein/Desktop/implement/dataset/msrvtt/video{}.mp4'
# for co in range(startvid, endvid+1):#here
#     print('video ', co)
#     inp =baseaddr.format(co)
#     capture = cv2.VideoCapture(inp)
#     num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     if not capture.isOpened:
#         print('Unable to open: ' + inp)
#         exit(0)
# #    i=0
# #    for i in range(0, num_frame):
# #        ret, frame = capture.read()
#     ret,frame = capture.read()
#     print(frame.shape, num_frame)

# ===========================
model = ResNet(weights='imagenet',include_top=True)
image_features_extract_model = tf.keras.Model(inputs = model.inputs, outputs = [model.layers[-2].output,model.layers[-1].output])


del model
print('loaded model')
print(image_features_extract_model.inputs)

# ===========================
done = 0
  
def crop_center_square(frame):
  _,y, x,_ = frame.shape
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[:,start_y:start_y+min_dim,start_x:start_x+min_dim]

def prep(img):
    
    if deb:print('check uint8:', img[0][0,:10,:2])
    print(img.shape)
    img = crop_center_square(img)
    print(img.shape)
    img = tf.image.resize(img, (224,224))  # img[:,:,20:300]
    print(img.shape)
    img = preprocess_input(img)
    if deb:print('after preprocess: ' ,img[0][0,:10])
    #print('np' ,img.shape)
    return img


def ext(frame):

    # Feel free to change batch_size according to your system configuration
#    image_dataset = tf.data.Dataset.from_tensor_slices(frame)
#    image_dataset = image_dataset.map(
#      prep, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)
#    
#    for img, path in image_dataset:
#        print(path)
#        print(img)
#        batch_features = image_features_extract_model(img)
#      batch_features = tf.reshape(batch_features,
#                                  (batch_features.shape[0], -1, batch_features.shape[3]))
#    
#      for bf, p in zip(batch_features, path):
#        path_of_feature = p.numpy().decode("utf-8")
#        np.save(path_of_feature, bf.numpy())
    sh = frame.shape 
    img = prep(frame)#(1,)+.reshape(frame.shape)
#    print('\r', 'start pred', end = '\r')
    # fc = image_features_extract_model(tf.reshape(img, (bufsize,224,224,3))) # .predict
    
   # image_features_extract_model.compile()
    if deb:print('fc calculated 1')
    
    print('done',sh,img.shape )
    fcnpy = image_features_extract_model.predict(img , steps =1)# tf.reshape(img, (bufsize,224,224,3))
    if deb:print('==== out', fc[0].shape, fcnpy[1].shape)
#    print('\r', 'end pred', end = '\r')
        
    
    return fcnpy


def batch(t):
#  print('shape',t.shape)
  num_frame,y,x,c = t.shape
  l1000 = []; l2048=[];
  buf = np.zeros(( (num_frame//bufsize+1)*bufsize,y,x,c))
  buf[:num_frame] = t
  for i in range(num_frame//bufsize+1):
    a,b = ext( buf[i*bufsize:(i+1)*bufsize] )
    l2048+=[a]
    l1000+=[b]
#  print(a.shape,len(l2048),l2048[0].shape,num_frame//bufsize+1)
  l2048 = np.concatenate(l2048,axis=0)[:num_frame]
  l1000 = np.concatenate(l1000,axis=0)[:num_frame]
  assert(l2048.shape==(num_frame,2048))
  return l2048,l1000
  
# ===============================

if dataset == 'msvd':
#  msvd_config = {'csv_path':'../../dataset/msvd/MSR Video Description Corpus.csv',
#    'video_path':'../../dataset/msvd/*.avi','vid_data': 'vidfeat/msvd/res_2048/', 'cap_data':'vidfeat/msvd/msvd_captions' }
  video_path = '../../dataset/msvd/*.avi'
  video_resnet= 'vidfeat/msvd/res_'
  
  video_list = [ [p, p.split('/')[-1][:-4]] for p in glob(video_path)] # 1970 videos are used. descriptions are for 2089 videos 

if dataset == 'msrvtt':
  video_path = '../../dataset/msrvtt/*.mp4'
#  video_i3d = 'vidfeat/msrvtt/i3d/'
  video_resnet= 'vidfeat/msrvtt/res_'
  video_list = [ [p, p.split('/')[-1][:-4]] for p in glob(video_path)] # 1970 videos are used. descriptions are for 2089 videos 
  
os.makedirs(video_resnet+'1000/', exist_ok=True)
os.makedirs(video_resnet+'2048/', exist_ok=True) ### TODO

for t in video_list:
  addr,vid = t
  print('video ', vid, end ='   ')
#  if video_resnet+'1000/'+vid+'.npz' in glob(video_resnet+'1000/*.npz'):
#    print('found')
#    continue
    
  capture = cv2.VideoCapture(addr)
  num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  if not capture.isOpened:
      print('Unable to open: ' + inp)
      exit(0)
  buf = []
  for i in range(0, num_frame):
    ret, frame = capture.read()
#                if i%8==0:
#                    plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
#                    plt.show()
    if i==0:
      print(frame.shape, num_frame)

    if i%dsrate==0:
      if RGB:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      buf += [frame]
      
      if deb:
        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                 key= lambda x: -x[1])[:10]:
            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    if frame is None:
      break
  l2048,l1000 = batch(ar(buf))
  np.savez_compressed(video_resnet+'1000/'+vid+'.npz', data = l1000)
#  np.savez_compressed(video_resnet+'2048/'+vid+'.npz', data = l2048)

#==========================================
#==========================================
#==========================================
          
#data = {}
#buf = []
#bufco = 0
#featlist = []
#featdic = {}
#for co in range(startvid, endvid+1):#here
#    # set # of videos
#    # set path
#    # change showclip(ur)
#    # first and last window is shown(beside peaks)
#    # name of pickle file
#    # make sure images are uint8
#    #==========================================
#    print('video ', co)
#    inp =baseaddr.format(co)
#    capture = cv2.VideoCapture(inp)
#    num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#    if not capture.isOpened:
#        print('Unable to open: ' + inp)
#        exit(0)
#    i=0
#    featdic[co]=0
#    for i in range(0, num_frame):
#        ret, frame = capture.read()
##                if i%8==0:
##                    plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
##                    plt.show()
#        if i==0:
#            print(frame.shape, num_frame)

#        if i%dsrate==0:
#            if RGB:
#                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        #    i+=1
#    #        frame_rs = resize(frame[:, 20:220], (112, 112), anti_aliasing=True)
##            frame_rs = resize(frame[:, sc:ec], (112, 112), anti_aliasing=True)
##            frame_rs = cv2.resize(frame, (112, 112), interpolation = cv2.INTER_AREA)
#            
#            #featlist += [ext(frame)]
#            featdic[co]+=1
#            buf +=[frame]
#            bufco +=1
#            if bufco == bufsize:
#            
##                x = threading.Thread(target=ext, args=(ar(buf),))
##                x.start()
##                x.join()
##                featlist += [ext(ar(buf))]
#                
#                featlist += [ext(ar(buf))]
#                if deb:
#                    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
#                             key= lambda x: -x[1])[:10]:
#                        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
#    
#                buf = []
#                bufco = 0
#        if frame is None:
#            break
#    #============================================
##    l = ar(l)
##    
##    tlength = 16
###    stride = 4 # was 2 meaning 8 frame steps. # 2 when stride=4
##    if len(l)<16 :
##        print("error... not enough window")
##        continue
##    ns = (len(l)-tlength)// stride +1
##    
##    fclist = []
##    for i in range(ns):
##        fc = ext(l[i*stride:i*stride+16])
##        fclist += [ fc ]
#        
#    #data.update({co:featlist})
#    # if co%100==0:
#    #      outfile = open(filename+'{}'.format(co),'wb')
#    #      pickle.dump([featdic, featlist],outfile)
#    #      outfile.close()
#    
#    
#if bufco !=0:
#   buf2 = np.zeros((bufsize,240,320,3))
#   buf2[:bufco] = np.array(buf)
#   featlist+=[ext(buf2)]

#outfile = open(filename,'wb')
#pickle.dump([featdic, featlist],outfile)
#outfile.close()
#    
#del image_features_extract_model


## filename = 'vidfeat/msr_test_res_'
#os.makedirs(filename+'1000/', exist_ok=True)
#os.makedirs(filename+'2048/', exist_ok=True) ### TODO

## outfile = open(filename,'rb')
## features = pickle.load(outfile)
## outfile.close()
## featlist  = features[1]
## featdic  = features[0]

## startvid = 7010
#keywords = []
##decode_predictions
##average then top20; then decode
#maxVidLen= max(list(featdic.values())) # 113 

#for layer, name in [[0,"2048"],[1,"1000"]]:
#    n=startvid; i=0; temp=[]; co=0
#    for j,batch in enumerate(featlist):
#        for k,v in enumerate(batch[layer]):
#            i+=1
#            co+=1
#            temp+=[v]
#            if n in featdic:### TODO
#                if i==featdic[n] :
#                    #outfile = open(filename+name+'/'+str(n),'wb')
#                    #pickle.dump(temp, outfile)
#                    
##                    z = np.zeros((maxVidLen, len(v)))
##                    z[:i] = temp
##                    np.save( filename+name+'/'+str(n)+'.npy', z)
#                    np.savez_compressed(filename+name+'/'+str(n)+'.npz', data = ar(temp))

#                    #outfile.close()
#                    n+=1
#                    # print(n, ' ', len(temp), temp[0].shape, ' ', j,k)
#                    i=0
#                    temp = []
#                    print('co' , co, sum(list(featdic.values())[:n]))
#                    # print("counter", co, sum(list(featdic.values())))
#    print('layer', name)
#    print("num of videos", n, '=', len(featdic))   # they mustnt be equal!
#    print("counter check", co, sum(list(featdic.values())))
#    # i think at last batch, empty temp is created and co is incremented. not important
#print('maxvidlen', maxVidLen)
