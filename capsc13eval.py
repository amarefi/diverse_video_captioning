#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""
from capsc13model import *
from capsc13data import *
from pycocoevalcap.diversity_eval import *

import os
base = '/home/amirhossein/Desktop/implement/dataset/meteor-1.5/'
# addresses for METEOR calculation
import cocoeval
from time import time; import threading
r=np.round

def report(out):
   if not out[-2].split()[0]=='Final':
      print("error, reporting meteor output")
      return

   final = float(out[-2].split()[2])
   frag = float(out[-4].split()[2])
   prec = float(out[-8].split()[1]);recall = float(out[-7].split()[1])
   return {'final':final, 'precision':prec,'recall':recall,'fragmentation':frag}

  # remove a word and add a padding at the end 
def remove_word(sent, length):
  i = np.random.randint(length-1)
  for j in range(i, length-1):
    sent[j] = sent[j+1]
  

def topsc(ts,gs):
  l = []
  

  l+= [ np.mean(ts), np.mean(gs)]
  t10 = np.sort(ts)[:-11:-1]
#        print(t10)
  # in order to test meteor model, first sort by gen score then calc their real score 
  g10 = np.argsort(gs)[:-11:-1]
#        print(g10)
  g10 = [ts[i] for i in g10]
#        print(g10,np.mean(t10), np.mean(g10)'\n\n')
  l+= [np.mean(t10), np.mean(g10)]
  
  t20 = np.sort(ts)[:-21:-1]
  g20 = np.argsort(gs)[:-21:-1]
  g20 = [ts[i] for i in g20]
  l+= [np.mean(t20), np.mean(g20)]

  t5 = np.sort(ts)[:-6:-1]
  g5 = np.argsort(gs)[:-6:-1]
  g5 = [ts[i] for i in g5]
  l+= [np.mean(t5), np.mean(g5)]
    
  t1 = np.sort(ts)[:-2:-1]
  g1 = np.argsort(gs)[:-2:-1]
  g1 = [ts[i] for i in g1]
  l+= [np.mean(t1), np.mean(g1)]
  
  return l


def calculate_diversity(generated_dict,methods, topX=0, scoredict=None ):
  ntest_sentence = topX

#  cand = {k:v for k,v in ref_dict.items() if k<10}
  #print('Candidates ', cand)
  scores = {}
  
  if topX==0:
    cand = {}
    for k,val in generated_dict.items():
      sentences = [' '.join(v) for v in val]
      cand[k] = sentences
    ntest_sentence = len(val)
    print('ntest_sentence = len(val)',ntest_sentence)
    if 'selfcider' in methods:
      selfcider = SelfCider(ntest_sentence)
      scores['selfcider'] = selfcider.evaluate(ref_dict, cand)
    if 'mbleu' in methods:
      mbleu = mBLEU(ntest_sentence)
      scores['mbleu'] = mbleu.evaluate(ref_dict,cand)
  else:
    cand = {}
    for k,val in generated_dict.items():
      sc = scoredict[k]
      sc = np.argsort(sc)[:-(topX+1):-1]
      sentences = [' '.join(val[i]) for i in sc]
      cand[k] = sentences
    if 'selfcider' in methods:
      selfcider = SelfCider(ntest_sentence)
      scores['selfcider'] = selfcider.evaluate(ref_dict, cand)
    if 'mbleu' in methods:
      mbleu = mBLEU(ntest_sentence)
      scores['mbleu'] = mbleu.evaluate(ref_dict,cand)
  return scores 

def calculate_scores(generated_dict,meteor=True, topX=0, scoredict=None ):
  IDs = []; samples = {}; gts={}
  for k,val in generated_dict.items():
    if topX==0: 
      sentences = val
    else:
      sc = scoredict[k]
#      print(sc,val)
      sc = np.argsort(sc)[:-(topX+1):-1]
      sentences = [val[i] for i in sc]
    for i,v in enumerate(sentences):
      ID = str(k)+'_'+str(i)
      IDs += [ID] 
      samples[ID]=[' '.join(v)]  
      gts[ID] = [sent[:-1] for sent in ref_dict[k]] # removing \n char
  scorer = cocoeval.COCOScorer() 
  if meteor:
    score,_ = scorer.score(gts, samples, IDs,['meteor']) # gts, samples, IDs
    return score, scorer.imgToEval
  
  else:
    scores,_ = scorer.score(gts, samples, IDs,['bleu','rouge','cider']) # gts, samples, IDs    
    print('COCO server',scores)
    return scores 
    
def complete_batch(tens, BATCH_SIZE, typ = tf.float32):
  bs = tens.shape[0]
  print(bs, BATCH_SIZE)
  temp = tf.zeros((BATCH_SIZE,)+tens.shape[1:],dtype = typ)
#  temp[:bs].assign( tens)
#  tens.assign(temp) # = temp
  return tf.concat([tens, temp], axis=0)[:BATCH_SIZE]
          
def test_score(ep, test_obj, addr,test_desired_length,testds, BATCH_SIZE, random_count=None):
    tokenizer = test_obj.tkn
    total_loss = 0#tf.Variable(0)
#    test_step(img_tensor, target,mask)
    
    # # lists for caption scorer 
    cap_sc_data = defaultdict(list) 
    
    # generated caption with test videos. number:caption
    generated_dict =  defaultdict(list) 
    score_dict = defaultdict(list) 
    res_len=[]

    for (batch, (img_tensor,mask, i3d_tensor,maski3d, target,vid_id,cap_id,pos)) in enumerate(testds):
#        print(img_tensor)
        # last batch is of size 48, not 64 
        # each batch has generated one test sent
#        pos = tf.random.normal((bs,1,latent_dim))
#        pos = np.repeat(pos, bs, axis=0)
#        pos = latentPOS_test[ test_desired_length ]['z_mean'][np.newaxis, np.newaxis,...]
#        if ((not gt_pos and cap_sc) or not cap_sc) and repeat : # and cap_sc
#        print(conf)
        if conf['test'] in ['noise','finalscorer']:
          pos = tf.random.normal((random_count,1,latent_dim[0]))
          if conf['snrio'] in ['slen']:
#            pos = tf.random.uniform((random_count,1,latent_dim[0]), 5,18,tf.int32) /20
#            pos = tf.cast(pos, dtype  =tf.float32)
            pos = tf.range(42,42+64)/6/20 # 1:64 7:18
            pos = tf.cast(pos,tf.float32)[...,tf.newaxis, tf.newaxis]
          img_tensor = tf.repeat(img_tensor, axis = 0, repeats = random_count )
          i3d_tensor = tf.repeat(i3d_tensor, axis = 0, repeats = random_count )
          mask = tf.repeat(mask, axis = 0, repeats = random_count )   
          vid_id = tf.repeat(vid_id, axis = 0, repeats = random_count )   
          target = tf.repeat(target, axis = 0, repeats = random_count )  
        
        bs = img_tensor.shape[0]
#        if bs!= BATCH_SIZE : #or batch==50 # TODO make error! solved at making dataset itself
#          # last batch has fewer samples
#          img_tensor = complete_batch(img_tensor, BATCH_SIZE)
#          mask = complete_batch(mask, BATCH_SIZE, tf.bool)
#          i3d_tensor = complete_batch(i3d_tensor, BATCH_SIZE)
#          vid_id = complete_batch(vid_id, BATCH_SIZE, tf.int64)
#          target = complete_batch(target, BATCH_SIZE, tf.int32)
#        print(target.shape)
                              
#          print(' *** test score break at batch ',bs, BATCH_SIZE, batch)
#          break 
        # if not cap_sc:
          # pos = tf.random.normal((random_count,1,latent_dim))
          # img_tensor = tf.repeat(img_tensor, axis = 0, repeats = random_count )
          # i3d_tensor = tf.repeat(i3d_tensor, axis = 0, repeats = random_count )
          # mask = tf.repeat(mask, axis = 0, repeats = random_count )   
          # vid_id = tf.repeat(vid_id, axis = 0, repeats = random_count )   
#        print('pos.shape',pos.shape)
        t_loss,generated_sent,features,meteors = test_obj.test_step\
          (img_tensor,mask, i3d_tensor,maski3d, target,pos, test_desired_length)
        
        meteors = meteors.numpy()
        vid_id = vid_id.numpy()
        generated_sent = generated_sent.numpy()
        features = features.numpy()
        
        print('\r', batch,'     ',bs, ' ' ,generated_sent.shape[0],  ' ',generated_sent.shape,pos.shape, img_tensor.shape,  end= '\r')
#        if not bs==generated_sent.shape[0]:
#        print(generated_sent.shape, img_tensor.shape,'\n\n')
        assert( bs==generated_sent.shape[0])

        # each video in batch
        temp_res_len = []
        for i in range(bs):#generated_sent.shape[0]
#            if conf['test'] in ['noise'] and conf['snrio'] in ['pos','e2epos']:
#              remove_word( generated_sent[i] , generated_sent.shape[1])
#              if i>20: break 
            sentence = []
            v = vid_id[i]
            for j in range(generated_sent.shape[1]):
                word = tokenizer.index_word[ generated_sent[i][j] ]
                if word=='<end>': # <end> is not appended
                    for k in range(j+1,generated_sent.shape[1]):
                      generated_sent[i][k]= 0# use <pad> after <end> token 
                    break
                sentence += [word ]
            generated_dict[v]+=[sentence]
            if conf['test'] in ['finalscorer']:
              
              score_dict[v] += [ meteors[i][0]]
            temp_res_len+=[len(sentence)-1] # <start> is excluded
            
#            printt = [ref_dict[v][j] for j in range(3,6)]
#            printt = ' // '.join(printt)
#            if (batch*64//BATCH_SIZE ==10 or batch*64//BATCH_SIZE ==11) and i<4 and not cap_sc: 
#              print(' '.join(generated_dict[v][0]),
#                '.:', score_dict[v][0] ,':. \n \t',printt ,' ', v)
              
            if conf['test'] in ['noise','model'] and conf['snrio'] in ['pos','e2epos']:
              cap_sc_data[v] +=[[ features[i],generated_sent[i], ' '.join(sentence)]] 
            else:
              cap_sc_data[v] += [[]]
        
        # removing duplicate sentence and returning 30 sent.
        num_sent = 30 
#        if conf['snrio'] in ['pos', 'e2epos','trainpos','traine2epos','slen'] and conf['test'] in ['finalscorer']:
        if conf['test'] in ['noise']:#,'finalscorer'
        # 'noise': needs more manipulation for: generated_dict,score_dict,cap_sc_data,res_len(!generated_sent)
          temp_generated_dict = []; temp_cap_sc_data=[]; temp_score_dict=[]; temp_res_len2=[]
#          print(score_dict, generated_dict)
          for i,gen in enumerate(generated_dict[v]): # all elements in batch is for same video 
            if gen not in temp_generated_dict or bs-i<=num_sent-len(temp_generated_dict):
#              print(temp_generated_dict)
              temp_generated_dict.append(gen)
              temp_cap_sc_data += [cap_sc_data[v][i]]
              temp_res_len2 += [temp_res_len[i]]
              if conf['test'] in ['finalscorer']:
                temp_score_dict += [score_dict[v][i]]
          generated_dict[v] = temp_generated_dict[:30]
          score_dict[v ] = temp_score_dict[:30]
          cap_sc_data[v] = temp_cap_sc_data[:30]
          res_len += temp_res_len2[:30]    
#          print('len(generated_dict[v])',len(generated_dict[v]),' ')
          assert(len(generated_dict[v])==num_sent)      
        else:
          res_len += temp_res_len
          
        
        total_loss += t_loss
    print('**features.shape:',features.shape, 'len(generated_dict[v])',len(generated_dict[vid_id[0]]),
      len(generated_dict[vid_id[1]]),len(generated_dict[vid_id[2]]),len(generated_dict[vid_id[3]]))     
        
    print('#test videos', len(generated_dict.keys()),'   ')
    # print samples 
    if conf['test'] in ['noise','final','finalscorer']: 
      repeat = 1;temp = ''
      for v in list(generated_dict.keys())[:7]:
        if len(generated_dict[v]) >4:
          repeat = 5
        if conf['test'] in ['finalscorer']:
          print1 = [' '.join(generated_dict[v][j])+' '+str(r(score_dict[v][j],3)) for j in range(repeat)]
#        print(ref_dict[v], generated_dict[v])
        else:
          print1 = [' '.join(generated_dict[v][j])+temp for j in range(repeat)]
        print1 = ' __ '.join(print1)
        print2 = [ref_dict[v][j][:-1] for j in range(4)]
        print2 = " __ ".join(print2)
        
        print(' +', print1 ,'\n=', print2, ' ', v,'\n')

     
#        print ('\r','epoch{} b{} total_loss{:.3f}  '.format(ep,batch, total_loss) , end='\r')  
#        if batch==10:break
         
#        # beam 
#        for i,sentences in enumerate(generated_sent): next time, exclude <start> <end>
#            generated_dict[vid_id[i]] = sentences
#            res_len+=[len(sentences[0])-1] # <start> is excluded
#            if batch ==10: 
#                for tt in generated_dict[vid_id[i]]:
#                    print(' '.join(tt[1:-1]),end=' .:. ')
#                print('')
#            
#        print ('\r','epoch{} b{} total_loss{:.3f}  '.format(ep,batch, total_loss.numpy()) , end='\r')  
    total_loss = 0# total_loss/len(id_list_test)/20  TODO

##        if imagenum==7020:break
    
    print('\n')           
    refaddr = addr[0]    
    testaddr = addr[1]    
    checkaddr = addr[2]
    resaddr = addr[3]
    
    # using coco evaluation modules     

    scores={}
#    print(" using coco evaluation modules B{:.3f},{:.3f},{:.3f},{:.3f} M{:.3f} R{:.3f} C{:.3f} ".format())
    # calculating final meteor score for topX(by model) 
    if conf['test'] in ['finalscorer']:
#      temp, _= calculate_scores(generated_dict,meteor=True, topX=1, scoredict=score_dict )
#      print('top1', temp)
#      temp, _= calculate_scores(generated_dict,meteor=True, topX=5, scoredict=score_dict )
#      print('top5', temp)
      temp, _= calculate_scores(generated_dict,meteor=True, topX=30, scoredict=score_dict )
      scores.update(temp)
#      for k,v in temp.items():
#        scores.update({k+'_top30',v})
        
      temp= calculate_scores(generated_dict,meteor=False, topX=30, scoredict=score_dict )
      scores.update(temp)
#      print('top30', temp,'\n')
      temp = calculate_diversity(generated_dict,['selfcider','mbleu'], topX=30, scoredict=score_dict )
      scores.update(temp)
      print('scores for TOP30',scores)
#      calculate_diversity(generated_dict,['selfcider','mbleu'], topX=10, scoredict=score_dict )
    if conf['snrio'] in ['trainpos','traine2epos','slen']:#'pos','e2epos',
      temp = calculate_diversity(generated_dict ,['selfcider','mbleu'])
      scores.update(temp)
      print('whole diversity:', temp)
      

    # now compare generated_dict with ref_dict
    if conf['test'] not in ['finalscorer']:
      met_score, met_scores = calculate_scores(generated_dict )
      scores.update(met_score)
  #    print('debug' , final_score)
      temp = calculate_scores(generated_dict,meteor = False )
      scores.update(temp)
  #    scores.update({'CIDEr':0})
  
    
#    if cap_sc: cap_sc_data[image] = [[feat, .. , meteor],[][]] or [[meteor],[],[]] (not cap_sc)
    if conf['test'] in ['model','ref','noise']: # ,'finalscorer'
      for image in generated_dict.keys():
        for j,sent in enumerate(generated_dict[image]):    
          cap_sc_data[image][j] += [met_scores[str(image)+'_'+str(j)]['METEOR']]

      print('meteor mean =' , np.mean([t['METEOR'] for t in met_scores.values() ]))
    met_list = defaultdict(list) 
    if conf['test'] in ['finalscorer'] and False:
#        if not cap_sc:
      file1 = open(checkaddr.format(ep),"w+") # refaddr and testaddr
      capsc_loss = 0
      capsc_loss_co = 0
      for image in generated_dict.keys():
  #        if image<7010:
  #            continue
        gen_score = score_dict[image] 
        true_score = [ t[-1] for t in cap_sc_data[image]]
        
        for g,t in zip(gen_score, true_score) :
          capsc_loss+=(g-t)**2
          capsc_loss_co+=1
#        print('gen score {} and \n true score {} '.format( gen_score, true_score), end = ' ')
#        v = met_list[image] ...
        met_list[image] = topsc(true_score, gen_score)
        
        


#        if len(set(true_score))!=1:
#          print(generated_dict[image])
        
        file1.write(str(image)+'\n')
        sd = np.argsort(gen_score)
#        print(sd)
        for i,sent in enumerate(generated_dict[image]):
            file1.write(' '.join(sent)+' .:.T{:.3f},, G{:.3f} \n'.format(true_score[i],gen_score[i]))
        for tt in ref_dict[image]:
            file1.write('\t'+tt)
      file1.close()
      # embed()
             
#      print(met_list)
      met_list =np.array( list(met_list.values()) )
      print('met list' , met_list[:10])
      met_list = np.mean(met_list, axis= 0)
      print('Meteor mean: {:.3f} ,( GenMet mean: {:.3f} ),\n bestX true, \
      bestX (rank using model):  1: {:.3f},{:.3f}   5: {:.3f},{:.3f}   10: {:.3f},{:.3f}   20: {:.3f},{:.3f}'
        .format(met_list[0],met_list[1],met_list[8],met_list[9],met_list[6],met_list[7],met_list[2],met_list[3], met_list[4],met_list[5]))
      print( 'Caption_scorer Test Loss {:.3f}    mean {:.3f}'.format(capsc_loss, capsc_loss/capsc_loss_co))
    
#    file1 = open(resaddr,"w+")
#    file1.writelines(out)
#    file1.close()
#    out = out[-10:]
    try:
      t = total_loss.numpy()
    except :
      print('Deb: total loss not tensor')
      t = total_loss
    dic = {'coco':scores,'mean_len':np.mean(res_len), 'tot_test_loss':t,'test_met_list':met_list}
    out = {'final':scores['METEOR'], 'precision':0,'recall':0,'fragmentation':0}
    dic.update(out )#report(out)
    return dic,cap_sc_data,[] #metdeb
    
    
def ablation_list(L):

  def flatten(B):    # function needed for code below;
      A = []
      for i in B:
          if type(i) == list: A.extend(i)
          else: A.append(i)
      return A
  outlist =[]; templist =[[]]
  for sublist in L:
      outlist = templist; templist = [[]]
      for sitem in sublist:
          for oitem in outlist:
              newitem = [oitem]
              if newitem == [[]]: newitem = [sitem]
              else: newitem = [newitem[0], sitem]
              templist.append(flatten(newitem))
  outlist = list(filter(lambda x: len(x)==len(L), templist))  # remove some partial lists that also creep in;
  return outlist
