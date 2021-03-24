import pandas, pickle,random ,json,re
from collections import defaultdict 
from glob import glob
import numpy as np
from capsc13pos import *

conf = {'snrio':'normal', 'test':'final','features':['i3d']}
ref_dict =defaultdict(list)
Test={}; Valid={}; Train = {}

max_length = 32

def msvd_caption(conf):
  csv_path = conf['csv_path'] # for csv and videos 
  video_path = conf['video_path']
  video_list = [ p.split('/')[-1] for p in glob(video_path)] # 1970 videos are used. descriptions are for 2089 videos 
  cap_data = conf['cap_data']
  try:
    infile = open(cap_data,'rb')
    t = pickle.load(infile)
    infile.close()
    video_train,video_valid, video_test, pos_dic, index_dic = t   

    
  except:
    print('len vid list' , len(video_list))
    msvd = pandas.read_csv(csv_path)
  #  msvd = msvd.loc[msvd['Language']=='English']
    print('total #videos' , len(set(msvd['VideoID'])))
    video_dict = defaultdict(list)
    co=0 # unique number for each caption in dataset 
    sid = 0
    
    counter=[0]
    pos_dic = {}; index_dic = {}; # max_length of sentence = 32
    
    for index, row in msvd.iterrows():
      if row['Language']!='English' : continue
      k = row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])
      if k+'.avi' not in video_list: continue
      d = row['Description']
      if type(d) !=str: continue # some descriptions are empty
      if d[-1]=='.': d= d[:-1] # removing dot(.)
      if '\n' in d:
        d = ''.join(e for e in d if e is not '\n')
      d = re.sub('[^A-Za-z0-9\'\s]+', '', d).lower()

      pos = make_pos_list(d, counter, pos_dic, index_dic)
      v = [d,k,co,pos] # row['VideoID'] ,row['Start'],row['End']
      co+=1
      video_dict[k ] +=[v]
  #    try:
  #      if row['Start']!=video_dict[k][0][2]:print(row['Start'],video_dict[k][0][2])
  #    except:
  #      pass
    print(len(video_dict.keys()))
    video_train = {}; video_valid={}; video_test={}
    for i,(k,v) in enumerate(video_dict.items()):
      if i<1200:
        video_train[i]=v
      if i>=1200 and i<1300:
        video_valid[i]=v
      if i>=1300:
        video_test[i]=v
    
    outfile = open(cap_data,'wb')
    pickle.dump( [video_train,video_valid, video_test, pos_dic, index_dic] ,outfile)
    outfile.close()  
    
  for dic in [video_train,video_valid, video_test]:
    for k,v in dic.items():
      for vv in v:
        ref_dict[k] += [ vv[0]+'\n' ]
      
  Train['dict'], Valid['dict'], Test['dict'] = video_train,video_valid, video_test
  return pos_dic, index_dic



def msrvtt_caption(conf,valid_size = 0.01):
  #with open('annotations_trainval2014/annotations/captions_train2014.json', 'r') as f:
  train_annot_json = conf['train_annot_json']
  test_annot_json = conf['test_annot_json']
  
  cap_data = conf['cap_data']
  try:
    pickle 
    infile = open(cap_data,'rb')
    t = pickle.load(infile)
    infile.close()
    train_dict, valid_dict, test_dict, pos_dic, index_dic = t   

  except:  
    with open(train_annot_json, 'r') as f:
      annotations = json.load(f)
        
    # cap_list = [... , '<start> ' + caption + ' <end>', ...]
    # refdict = { ..., 8010:[ ..., 'caption'+'\n' ,...] , ...}
    ref = defaultdict(list)
    co = 0
    
    counter=[0]
    pos_dic = {}; index_dic = {}; # max_length of sentence = 32
  
    for annot in annotations['sentences']:
      k = int(annot['video_id'][5:])
      pos = make_pos_list(annot['caption'], counter, pos_dic, index_dic)
      v = [ annot['caption'], annot['video_id'][5:],co,pos]
      ref[k] += [v]
      co+=1
    
    train_dict={}; valid_dict={}; test_dict={};
    # split valid and train 
    # this can be done later in iteration loops TODO
    #random.setstate()
    sample = random.sample(range(7010), int(7010*valid_size) ) # a 3% validation 
    print('sample  ', sample[:10])
    for s in sample:
      valid_dict[s] = ref[s]
    sample2 = [s for s in range(7010) if s not in sample]
    for s in sample2:
      train_dict[s] = ref[s]

    # test data 
    with open(test_annot_json, 'r') as f: 
        annotations = json.load(f)
    test_dict = defaultdict(list)
    
    for annot in annotations['sentences']:
      k = int(annot['video_id'][5:])
      pos = make_pos_list(annot['caption'], counter, pos_dic, index_dic)
      v = [ annot['caption'], annot['video_id'][5:],co,pos]  
      test_dict[k] += [v]
      co+=1
        
    outfile = open(cap_data,'wb')
    pickle.dump( [train_dict, valid_dict, test_dict, pos_dic, index_dic] ,outfile)
    outfile.close()  
  
  for dic in [train_dict, valid_dict, test_dict]:
    for k,v in dic.items():
      for vv in v:
        ref_dict[k] += [ vv[0]+'\n' ]

  Train['dict'], Valid['dict'], Test['dict'] =  train_dict, valid_dict, test_dict
  return  pos_dic, index_dic
  
def calc_video_len(dic,c):
  m=[]
  for k,v in dic.items():
    i = v[0]
    path = c['video_resnet']+i[1]+'.npz'
    sh = np.load(path)['data'].shape
    assert(sh[-1]==2048)
    m+=[sh[0]]
#    if m<sh[0]:
#      m = sh[0]
#      print(m)
  print('max video length (downsampled frames):', max(m) )
  print(np.histogram(m))
  

def get_caption_list_train(c,single=False ):
  ret = []
  for dic in [Train,Valid]:
    id_list = []; cap_list = []
    for k, v in dic['dict'].items():
      if single:
          i = v[0]
          t = [c['video_resnet']+i[1]+'.npz', c['video_i3d']+i[1]+'.npz', '', str(i[1]),str(k), str(i[2])\
            , c['whole_features']+i[1]+'.npz'] # res,i3d,'',vidname,vid,captionid
          id_list += [t] 
          cap_list += [['<start> ' +i[0] +' <end>' for i in v]] # dummy!     
      else:
        for i in v:
          t = [c['video_resnet']+i[1]+'.npz', c['video_i3d']+i[1]+'.npz', '', str(i[1]),str(k), str(i[2])\
            , c['whole_features']+i[1]+'.npz'] # res,i3d,'',vidname,vid,captionid
          id_list += [t] 
          cap_list += ['<start> ' +i[0] +' <end>' ]
#    print('shuffling')
    temp = list(zip(id_list,cap_list))
    random.shuffle(temp)
    id_list = [t[0] for t in temp]
    cap_list = [t[1] for t in temp]
#    print('shuffled')
    if single:
      dic['id_list_single' ] =id_list
      dic['cap_list_single'] = cap_list 
    else: 
      dic['id_list' ] =id_list
      dic['cap_list'] = cap_list 
    
def get_caption_list_test(c ):
  # ValueError: Can't convert non-rectangular Python sequence to Tensor.
  id_list = []; cap_list = []
  dic = Test['dict']
  for k, v in dic.items():
#    for i in v:
    i = v[0] 
    t = [c['video_resnet']+i[1]+'.npz', c['video_i3d']+i[1]+'.npz', '',  
      str(i[1]),str(k), str(i[2]) , c['whole_features']+i[1]+'.npz'] # res,i3d,'',vidname,vid,captionid
    id_list += [t] 
#      cap_list[-1] += ['<start> ' +i[0] +' <end>' ]
    cap_list +=[[[0]*max_length]]
  Test['id_list_single'] = id_list 
  Test['cap_vector_single'] = cap_list 
  # cap_list empty.. id_list for videos not sentences .. use test_dict for evaluation
  
  
def repeat_remainder_batch(dic, testbatchsize):
  rem = len(dic['id_list'])%testbatchsize
  dic['id_list'] += [dic['id_list'][-1]]*(testbatchsize-rem)
  dic['cap_list'] += [dic['cap_list'][-1]]*(testbatchsize-rem)
  assert(len(dic['id_list'])%testbatchsize==0)


def caption_scorer_dataset(cap_sc_list):
  print('shuffling')
  random.shuffle( cap_sc_list ) 
  print('shuffled')
  cap_sc_list = [t+[i] for i,t in enumerate(cap_sc_list)]
  f = [tf.constant(t[0],dtype = tf.float32) for t in cap_sc_list]; #f = np.array(f)
  c = [tf.constant(t[1],dtype = tf.int32) for t in cap_sc_list]; #c = np.array(c)
  s = [tf.constant(t[2],dtype = tf.string) for t in cap_sc_list];
  m = [tf.constant(t[3],dtype = tf.float32) for t in cap_sc_list]; #m = np.array(m)
  del cap_sc_list
  print('some samples of caption scorer dataset \n' ,\
     '* '.join([t.numpy().decode()+' '+str(tt.numpy()) for t,tt in zip(s[:11], m[:11])]),'\n')
  print('sliced')
  assert(len(f)== len(c)); assert(len(c)== len(m))
  assert(len(set([tuple(t.shape) for t in c]))==1)
  ds = tf.data.Dataset.from_tensor_slices((f,c,m))
  ds = ds.batch(64)

  print('dataset ready!')
  return ds
#  del f; del c; del m;
  
  
    
#    id_list += [[trainfile +annot['video_id'][5:]+nptype,i3dtrainfile +\
#      annot['video_id'][5:]+nptype,frcnn_trainfile +annot['video_id'][5:]+nptype,\
#      annot['video_id'][5:], str(co)]]
#    co +=1
          
#  # now test and train captions are in refdict and cap_list.    
#  test_seqs = tokenizer.texts_to_sequences(cap_list_test)
#  print('\ntest')
#  calc_max_length(test_seqs)
#  cap_vector_test = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post', maxlen=max_length, truncating='post')
#  # cap_vector_test and id_list_test are used to make dataset

#  # now i want to group the sentences of each video, so that 
#  #  (img_tensor, target) will contain all sentences related to video
#  # id_list_test is in order, so reshaping is enough. (first i check this!)
#  #for i,idt in enumerate(id_list_test):
#  #    if idt!=id_list_test[i-i%20]:print('error \n\n\n\n') # this is removed due to path[4]
#  id_list_test = id_list_test[::20]
#  cap_vector_test = cap_vector_test.reshape(-1,20,cap_vector_test.shape[-1])




    
#  = msvd_caption(c)
#  # ref_dict, id_list 
#  ref_dict = {}; ref_dict.update(tr); ref_dict.update(va);ref_dict.update(te)
#  
#  temp = tr.values()
#  id_list_train = []
#  cap_list_train = ['<start> ' +i[0] +' <end>' for i in temp]
#  for i in temp:
#    t = [c['video_resnet']+i[1]+'.npz', c['video_i3d']+i[1]+'.npz', '', i[1], i[2]]
#    id_list_train += [t] 

#  temp = va.values()
#  id_list_valid = []
#  cap_list_valid = ['<start> ' +i[0] +' <end>' for i in temp]
#  for i in temp:
#    t = [c['video_resnet']+i[1]+'.npz', c['video_i3d']+i[1]+'.npz', '', i[1], i[2]]
#    id_list_valid += [t] 

#  temp = te.values()
#  id_list_test = []
#  cap_list_test = ['<start> ' +i[0] +' <end>' for i in temp]
#  for i in temp:
#    t = [c['video_resnet']+i[1]+'.npz', c['video_i3d']+i[1]+'.npz', '', i[1], i[2]]
#    id_list_test += [t] 
#    
#  return id_list_train, cap_list_train, id_list_valid, cap_list_valid, id_list_test, cap_list_test, ref_dict
  
  
  
#import random 
#def get_validation(id_list, cap_vector):
#  # this can be done later in iteration loops TODO
#  #random.setstate()
#  sample = random.sample(range(len(id_list)//20 ), len(id_list)//20//30 ) # a 3% validation 
#  print('sample  ', sample[:10])
#  id_list_valid=[]; id_list_train = []; cap_vector_valid = []; cap_vector_train=[]
#  for i,idl in enumerate(id_list):
#    if i//20 in sample:
#      id_list_valid += [idl]  
#      cap_vector_valid += [cap_vector[i]]
#    else:
#      id_list_train += [idl]  
#      cap_vector_train += [cap_vector[i]]    
#      
#  cap_vector_train = np.array(cap_vector_train)
#  cap_vector_valid = np.array(cap_vector_valid)
#  #print(len(id_list_train), len(id_list_valid))
#  id_list_valid = id_list_valid[::20]
#  cap_vector_valid = cap_vector_valid.reshape(-1,20, cap_vector_valid.shape[-1])
#  
#  id_list_train2 = id_list_train[::20]
#  cap_vector_train2 = cap_vector_train.reshape(-1,20, cap_vector_train.shape[-1])
#  
#  sample = random.sample(range(len(id_list_train)), len(id_list_train))
#  id_list_train =  [id_list_train[i] for i in sample]
#  cap_vector_train = np.array( [cap_vector_train[i] for i in sample]  )
#  
#  sample = random.sample(range(len(id_list_valid)), len(id_list_valid))
#  id_list_valid =  [id_list_valid[i] for i in sample]
#  cap_vector_valid = np.array( [cap_vector_valid[i] for i in sample]  )
#  
#  sample = random.sample(range(len(id_list_train2)), len(id_list_train2))
#  id_list_train2 =  [id_list_train2[i] for i in sample]
#  cap_vector_train2 = np.array( [cap_vector_train2[i] for i in sample]  )
#  
#  return id_list_train,id_list_valid,id_list_train2, cap_vector_train, cap_vector_valid,cap_vector_train2

    

msvd_config = {'csv_path':'../../dataset/msvd/MSR Video Description Corpus.csv',
  'video_path':'../../dataset/msvd/*.avi',
  'video_resnet': 'vidfeat/msvd/res_2048/','video_resnet_1000': 'vidfeat/msvd/res_1000/',
  'video_i3d': 'vidfeat/msvd/i3d/', 'video_i3d_clips': 'vidfeat/msvd/i3d_clips/', 
  'cap_data':'vidfeat/msvd/msvd_captions',
  'whole_features': 'vidfeat/msvd/whole_features/',
  'latentPOS_path':'vidfeat/msvd/msvdLATENT_50' }
  
msrvtt_config = {'test_annot_json':'../../dataset/msrvtt/test_videodatainfo.json',
  'train_annot_json': '../../dataset/msrvtt/train_val_videodatainfo.json',
  'video_path':'../../dataset/msrvtt/*.avi',
  'video_resnet': 'vidfeat/msrvtt/res_2048/','video_resnet_1000': 'vidfeat/msrvtt/res_1000/video',
  'video_i3d': 'vidfeat/msrvtt/i3d/',  'video_i3d_clips': 'vidfeat/msrvtt/i3d_clips/video', 
  'cap_data':'vidfeat/msrvtt/msrvtt_captions_valid7percent',
  'whole_features': 'vidfeat/msrvtt/whole_features/' ,
  'latentPOS_path':'vidfeat/captionPOSlatent'}#msrvttLATENT_50

  
if __name__ == "__main__":
  dataset  = 'msrvtt'
  if dataset == 'msvd':
    dataset_config = msvd_config
    msvd_caption(msvd_config)
  if dataset == 'msrvtt':
    dataset_config = msrvtt_config
    msrvtt_caption(msrvtt_config)
  
#  id_list_train, cap_list_train,id_list_valid,cap_list_valid = get_caption_list_train(train_dict, valid_dict,dataset_config)# with shuffling
#  train_dict.update(valid_dict)
#  calc_video_len(train_dict,dataset_config)


#  a,b,c = msvd_caption(path,video_list)










