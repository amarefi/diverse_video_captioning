import numpy as np 
from glob import glob 

names = glob('vidfeat/msvd/res_2048/*.npz')
names = [n.split('/')[-1][:-4] for n in names]
base  = 'vidfeat/msvd/'
dest = 'vidfeat/msvd/whole_features/'
#print(names)
for name in names:
  res_2048 = np.load(base+'res_2048/'+name+'.npz')['data']
  res_1000 = np.load(base+'res_1000/'+name+'.npz')['data']
  i3d = np.load(base+'i3d/'+name+'.npz')['data']
  i3d_clips = np.load(base+'i3d_clips/'+name+'.npz')['data']
  np.savez_compressed(dest+name, res_1000 = res_1000, res_2048=res_2048, i3d=i3d, i3d_clips = i3d_clips)



#  
#def write_file_meteor(refaddr,testaddr, refdic,gendic, part,num,odict,topX, scoredict=None):
##    print('thread {} started'.format(part))
#    keylist = sorted(gendic.keys())[part::num]
##    print(len(keylist))
#    part = str(part) 
#    
#    meteor  = Meteor()
#    file1 = open(refaddr+part,"w+") # refaddr or testaddr
#    file2 = open(testaddr+part,"w+") # refaddr and testaddr
#    for image in keylist: # it was ref_dict (but needs conditioning)
#      if topX==0: 
#        sentences = gendic[image]
#      else:
#        sc = scoredict[image]
#        sc = np.argsort(sc)[:-(topX+1):-1]
#        sentences = [gendic[image][i] for i in sc]
#        
#      for sent in sentences:
#        file1.writelines(refdic[image]) # ref_dict is text but valid_dict is num
#        file2.write(' '.join(sent)+'\n') # excluding <start> <end>
#    file1.close()
#    file2.close()
#    
##    for image,cap in generated_dict.items():
##        if image<7010:
##            continue
#    
#    # base = '/home/amir/Desktop/thesis/datasets/meteor-1.5/'
#    # ref = '{}example/xray/myref'.format(base)
#    # test= '{}example/xray/mytest'.format(base)
#    # prefix = 'captioning'
#    # refcount = 1
#    cmd = 'java -Xmx2G -jar {}meteor-*.jar {} {} -norm -writeAlignments -f {} -l en -r {}'.format(base,testaddr+part,refaddr+part,prefix,refcount)
#    stream = os.popen(cmd)
#    output = stream.read()
#    
#    i=11
#    score = defaultdict(list) 
#    out = output.split(sep = '\n')
#    if topX==0:
#      for image in keylist:
#        for sent in gendic[image]:    
#          score[image] += [float(out[i].split()[-1])]
#          i+=1
#    out2 = out 
#    out = out[-10:]
#    out = report(out)
#    if topX!=0:
#      print(' TOPX ',topX,' ', out['final'])
#      return 
#    odict[int(part)] = [ out ,score,out2 ]
##    print('thread {} ended'.format(part))
##    print(out)
##    return output 


#def threading_meteor(gendic,topX=0,num_thread = 5, scoredict=None):
#  odict = {}
#  threads = list()
#  t1 = time()
#  for i in range(num_thread):
#    x = threading.Thread(target=write_file_meteor2, args=(gendic,i,num_thread,odict,0, scoredict,))
#    threads.append(x)
#    x.start()
#  for i in range(num_thread):
#    threads[i].join()
#  t2 = time()
#  print('TH Meteor: scoring ', t2-t1)  
#     
#  cmd = 'java -Xmx2G -jar {}meteor-1.5.jar -norm -writeAlignments -f {} -l en -stdio'.format(base,prefix)
#  cmd = cmd.split()
#  cat =  Popen(cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True, bufsize=1) 
#  
#  # test 
#  print('hi', file=cat.stdin, flush=True) 
#  print('coco Out' , cat.stdout.readline())
#  print('coco Out' , cat.stdout.readline())
#  
#  eval_dic={}
#  for i in range(num_thread):
#    eval_dic.update(odict[i])
#    
#  eval_line = 'EVAL'
#  for k,v in eval_dic.items():
#    for e in v:
#      eval_line += ' ||| {}'.format(e)
##  print(eval_line)
#  if 'Error: specify SCORE or EVAL' in eval_line:
#    print('there is error!')
#  print(eval_line +'\n', file=cat.stdin, flush=True) #
#  met_scores = defaultdict(list)
#  for k,v in eval_dic.items():
#    for e in v:
#      try:
#        temp = cat.stdout.readline()[:-1] 
#        met_scores[k]+=[float(temp)]
#      except:
#        print(k,temp)
#  final_score= float(cat.stdout.readline()[:-1])
#  
#  t3 = time()
#  print('TH Meteor: eval ', t3-t2)  

#  return final_score, met_scores     
#  
#  
#def write_file_meteor2(gendic, part,num,odict,topX, scoredict=None):
#    cmd = 'java -Xmx2G -jar {}meteor-1.5.jar -norm -writeAlignments -f {} -l en -stdio'.format(base,prefix)
#    cmd = cmd.split()
#    cat =  Popen(cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True, bufsize=1) 
#        
#    eval_dic = defaultdict(list)
##    print('thread {} started'.format(part))
#    keylist = sorted(gendic.keys())[part::num]
##    print(len(keylist))
#    part = str(part) 
#    
#    for image in keylist: # it was ref_dict (but needs conditioning)
#      if topX==0: 
#        sentences = gendic[image]
#      else:
#        sc = scoredict[image]
#        sc = np.argsort(sc)[:-(topX+1):-1]
#        sentences = [gendic[image][i] for i in sc]
#        
#      for sent in sentences:
#        refs = ' ||| '.join([s[:-1] for s in ref_dict[image]])
#        sent = ' '.join(sent)
#        echo = 'SCORE ||| '+refs+' ||| '+sent
#  #        print(echo)
#        print(echo, file=cat.stdin, flush=True)
#        temp = cat.stdout.readline()[:-1]
#        if temp == '' or 'Error' in temp:
#          print('error in scoring')
#        eval_dic[image] += [temp ]
#        
#        
#    cat.terminate()
#    odict[int(part)] = eval_dic  
#    
