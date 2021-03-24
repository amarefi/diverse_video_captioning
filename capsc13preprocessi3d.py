#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extracting i3d features. 
? why this model uses the whole video as input??? not 64 frames



train 0..7009
test  7010..9999
nohup python preprocess4i3d.py --filename I3D_test --datasetaddr /home/amirhossein/Desktop/implement/dataset/msrvtt/TestVideo/video{}.mp4 --s 7010 --e 9999 &
nohup python preprocess4i3d.py --filename I3D_train --datasetaddr /home/amirhossein/Desktop/implement/dataset/msrvtt/msrvtt3/TrainValVideo/video{}.mp4 --s 0 --e 7009 --cpu &

old notes 
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
import tensorflow_hub as hub
#from tensorflow_docs.vis import embed
from urllib import request  # requires python3

# import matplotlib.pyplot as plt
import numpy as np
ar = np.array
# from keras.models import Sequential, Model
import pickle,os; from glob import glob
#from tqdm import tqdm
import random,re,imageio,cv2

from tensorflow.keras.applications import ResNet152V2 as ResNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", default=False)
parser.add_argument("--dataset", type=str,required = True)
#parser.add_argument("--datasetaddr", type=str,required = True)
#parser.add_argument("--s", type=int,required = True)
#parser.add_argument("--e", type=int,required = True)

args = parser.parse_args()

if args.cpu:
  print('using cpu not gpu', args.cpu)
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
  gpu_devices = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(gpu_devices[0], True)
  print("memory growth enabled")
  
deb = 0
dataset = args.dataset 
##os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

## baseaddr = '/home/amirhossein/Desktop/implement/dataset/msrvtt/video{}.mp4'
#baseaddr = args.datasetaddr
##baseaddr = '/home/amir/Desktop/thesis/datasets/msrvtt/test_videos/TestVideo/video{}.mp4'
## filename = 'msr299test7075_'
#filename = 'vidfeat/'+args.filename
#startvid = args.s; endvid = args.e; 
dsrate = 8
bufsize = 50
RGB = True 


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
#model = ResNet(weights='imagenet',include_top=True)
#image_features_extract_model = tf.keras.Model(inputs = model.inputs, outputs = [model.layers[-2].output,model.layers[-1].output])


#del model
#print('loaded model')
#print(image_features_extract_model.inputs)

# ===========================
done = 0

#def prep(img):
#    
#    if deb:print('check uint8:', img[0][0,:10,:2])
#    
#    img = tf.image.resize(img[:,:,20:300], (224,224)) 
#    img = preprocess_input(img)
#    if deb:print('after preprocess: ' ,img[0][0,:10])
#    #print('np' ,img.shape)
#    return img


#def ext(frame):

#    # Feel free to change batch_size according to your system configuration
##    image_dataset = tf.data.Dataset.from_tensor_slices(frame)
##    image_dataset = image_dataset.map(
##      prep, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)
##    
##    for img, path in image_dataset:
##        print(path)
##        print(img)
##        batch_features = image_features_extract_model(img)
##      batch_features = tf.reshape(batch_features,
##                                  (batch_features.shape[0], -1, batch_features.shape[3]))
##    
##      for bf, p in zip(batch_features, path):
##        path_of_feature = p.numpy().decode("utf-8")
##        np.save(path_of_feature, bf.numpy())
#    sh = frame.shape 
#    img = prep(frame.reshape(frame.shape))#(1,)+
##    print('\r', 'start pred', end = '\r')
#    # fc = image_features_extract_model(tf.reshape(img, (bufsize,224,224,3))) # .predict
#    
#   # image_features_extract_model.compile()
#    if deb:print('fc calculated 1')
#    fcnpy = image_features_extract_model.predict( tf.reshape(img, (bufsize,224,224,3)), steps =1)
#    if deb:print('==== out', fc[0].shape, fcnpy[1].shape)
##    print('\r', 'end pred', end = '\r')
#        
##    global done 
##    done+=1
##    print('\r', 'done', done, end = '\r')
#    
#    # eager execution is in TF.2
##    config = tf.ConfigProto(
##        device_count = {'GPU': 0,'CPU': 6 }, inter_op_parallelism_threads = 6, intra_op_parallelism_threads = 6
##    )
##    sess = tf.Session(config=config)
##    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
##    gpu_options.per_process_gpu_memory_fraction = 0.4
##    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess :
##    with tf.Session() as sess:
##        sess.run(tf.global_variables_initializer())
##        fcnpy =  fc.eval()
#    print('done',sh,img.shape )
#    
#    return fcnpy

def fetch_ucf_video(video):
  """Fetchs a video and cache into local filesystem."""
  cache_path = os.path.join(_CACHE_DIR, video)
  if not os.path.exists(cache_path):
    urlpath = request.urljoin(UCF_ROOT, video)
    print("Fetching %s => %s" % (urlpath, cache_path))
    data = request.urlopen(urlpath, context=unverified_context).read()
    open(cache_path, "wb").write(data)
  return cache_path

# Utilities to open video files using CV2
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]] # BGR to RGB 
      frames.append(frame)
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

def to_gif(images):
  converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
  imageio.mimsave('./animation.gif', converted_images, fps=25)
#  return embed.embed_file('./animation.gif')


# ===============================

#video_path = "v_CricketShot_g04_c02.avi"
#sample_video = load_video(video_path)

print('loading')
#i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']
i3d = hub.load("/home/amirhossein/Desktop/implement/i3d-kinetics-400_1")
#print([i.shape  for i in i3d.variables])
print(i3d.signatures)
i3d = i3d.signatures['default']

## Get the kinetics-400 action labels from the GitHub repository.
#KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
labelspath = '/home/amirhossein/Desktop/implement/i3d-kinetics-400_1/labels.txt'
#with request.urlopen(KINETICS_URL) as obj:
with open(labelspath, "r+") as obj:
  labels = [line.strip() for line in obj.readlines()]
print("Found %d labels." % len(labels))


def predict(sample_video):
  # Add a batch axis to the to the sample video.
  model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

  logits = i3d(model_input)['default'][0] #TODO what is this!?
#  print(logits)
  probabilities = tf.nn.softmax(logits)

#  print("Top 5 actions:")
#  for i in np.argsort(probabilities)[::-1][:5]:
#    print(f"  {labels[i]:22}: {probabilities[i] * 100:5.2f}%")
  return probabilities

#predict(sample_video)


# filename = 'vidfeat/msr_test_res_'
#os.makedirs(filename+'/', exist_ok=True)

#if __name__ == "__main__":
if dataset == 'msvd':
#  msvd_config = {'csv_path':'../../dataset/msvd/MSR Video Description Corpus.csv',
#    'video_path':'../../dataset/msvd/*.avi','vid_data': 'vidfeat/msvd/res_2048/', 'cap_data':'vidfeat/msvd/msvd_captions' }
  video_path = '../../dataset/msvd/*.avi'
  video_i3d= 'vidfeat/msvd/i3d_clips/'
  
  video_list = [ [p, p.split('/')[-1][:-4]] for p in glob(video_path)] # 1970 videos are used. descriptions are for 2089 videos 

if dataset == 'msrvtt':
  video_path = '../../dataset/msrvtt/*.mp4'
  video_i3d = 'vidfeat/msrvtt/i3d_clips/'
  video_list = [ [p, p.split('/')[-1][:-4]] for p in glob(video_path)] # 1970 videos are used. descriptions are for 2089 videos 
  
os.makedirs(video_i3d, exist_ok=True)

for t in video_list:
  addr,vid = t
  frames = load_video(addr)
  print('video ', vid, frames.shape, end = '')

  i=0
  pr=[]
  while True:
    pr += [predict(frames[i:i+64])[tf.newaxis,...]]
    i+=32
    if len(frames)<i+64:
      break
#  print(pr)
  pr = tf.concat(pr,axis=0)
  print(pr.shape)
  
#  pr = predict(frames[:32]) all works! :2 produced nan!
#  print(pr)
#  
#  pr = predict(frames[:64])
#  print(pr)
#  pr = predict(frames[-64:])
#  print(pr)
#  print(pr)
  np.savez_compressed(video_i3d+vid+'.npz', data = pr)

#if dataset == 'msrvtt':
#  for co in range(startvid, endvid+1):
#    print('video ', co)
#    path =baseaddr.format(co)
#    frames = load_video(path)
#    pr = predict(frames)
#  #  print(pr)
#    np.savez_compressed(filename+'/'+str(co)+'.npz', data = pr)



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
