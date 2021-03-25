# diverse video captioning using LSTMs and POS

introduction
=======
In this thesis, we propose a method for describong short videos with diverse sentences. we use encoder-decoder framework as baseline, and use a method for embedding the sentence syntax in a vector to train the model.  
This syntex vector is either sentence length (1d vector) or POS vector(encoded with a variational auto-encoder-VAE).  
We use MSRVTT and MSVD datasets to test our method. Pretrained Resnet(+LSTM) and I3D networks are used for static and dynamic feature extraction    
code is available and requrements & tutorials are given below. 

requirements
========
+ python 3
+ tensorflow 2.1
+ Microsoft COCO evaluation library [link](https://github.com/tylin/coco-caption) (use cocoeval.py and pycocoevalcap folder in the same address as capsc13XX.py files)
+ sklearn and other common packages

how to use the code
========
e.g. : python capsc13main.py server 1 --snrio trainpos --lastmodel 202101281638_15  
arguments:  
**where**,help="directory" # 'home' or 'server'  
**epochs**,type=int, # number of training epochs (for encoder-decoder model)  
optional arguments:  
**--posepoch**, type=int, default=7 # (epochs for caption_scorer model)  
**--cpu**, action="store_true", default=False # do the processing without GPU   
**--silent**, action="store_false", default=True # dont print extra info (e.g. epoch number)  
**--notrain**, action="store_true", default=False # do the processing just for 100 batches (used for debugging)   
**--snrio**, type=str,default = 'normal'  
\# code scenario:  
+ normal(encoder-decoder)  
+ slen (train the model using sentence length-as one of decoder inputs, test with a range of sentence lengths)
+ pos (train the model using POS vector-as one of decoder inputs, test with random normal POS vectors)
+ e2epos (train the model using POS vector, end to end-as one of decoder inputs)
+ trainpos 
+ traine2epos 
+ slen_cte   

**--dataset**, type=str,default = 'msrvtt' # msrvtt or msvd  
**--lastmodel** name of last saved model or '' or 'nosave'  

setup variables in code
======

\# video, feature vectors(1 file for each video) and captions addresses  
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
  'latentPOS_path':'vidfeat/captionPOSlatent'}  

max_length = 32 # maximum sentence length

file structure
======
![](/images/struct.png)


flow chart
=======
![](/images/flow.png)

variable examples
=====
In [4]: Train['dict'][1] # train video 1: sentence, video number, caption number, tokenized sentence                                                                                                                              
Out[4]: 
[['in a kitchen a woman adds different ingredients into the pot and stirs it',  
  '1',  
  110460,  
  array([ 3,  8,  4,  5,  4,  5,  7, 15,  6,  8,  4,  5, 11,  7, 12,  2,  0,  
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])],...]  
          
In [5]: Train['id_list'][1] # feature vector files. (shuffled)                                                                                                                           
Out[5]:  
['vidfeat/msrvtt/res_2048/6620.npz',  
 'vidfeat/msrvtt/i3d/6620.npz',  
 '',  
 '6620',  
 '6620',  
 '126198',  
 'vidfeat/msrvtt/whole_features/6620.npz']            
 
In [7]: Train['cap_vector'][1]                                                                                                                        
Out[7]:  
array([   3,   22,    7,   84, 2567,   17, 1777,   15,   80,  973, 3470,
          4,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  
      dtype=int32)  

 

code structure 
========
here 

method description
========
now



abstract
========

In recent years, the simultaneous analysis of image and text by artificial intelligence has gained much attention. Video description is one of the topics used to help the blind, automate video content analysis, and more. This issue is usually examined in the context of supervised learning and is very similar to image description and machine translation.  
The proposed solutions to this problem are mainly in the framework of encoder-decoder and attention-based neural networks. Selection of various pre-trained networks to extract 2D and 3D visual features (description of objects and actions in the image), various hierarchical structures and different teaching methods (based on reinforcement learning and so on) are among the solutions of researchers in this field. The research background will be reviewed in detail.  
The more similar the sentences produced by the models to human sentences, the more desirable it is. Creating diversity in the produced sentences is one of the ways to make the sentences more natural, which is examined in this research. To this end, solutions based on GAN and VAE networks have been proposed so far.   
The proposed solution of this research is to use part of speech (POS) to train the model. In the training phase, the POS sequence of sentences is summarized by AutoEncoder (VAE) to enforce compression and normal distribution. Sentences diversity is modelled in this way, according to POS vector as one the inputs. Due to the fact that these POS vectors have a normal distribution, in the test phase, a random vector with a normal distribution is used to produce sentences with various formats. Another solution is to use sentence length in training the model, which, like the above, produces a variety of sentences.   
Available video description datasets usually have several different descriptions for each video. This poses a challenge in teaching the neural network model, because the simple model cannot model several different outputs at the same time, reducing the variety of sentences and producing simple sentences. The proposed solution has the advantage that in addition to video features, it also uses POS features during training, thus improving training.   
The results of experiments show that the proposed solutions have the ability to produce sentences with various formats with an acceptable accuracy and comparable diversity with state of the art.   









citation
========

If you use this code base in your work, please cite

\@article{DiverseVideoCaptioningPOS,  
  title={Diverse Video Captioning using RNNs and POS},  
  author={Amirhossein Arefipour and Hamid Behrouzi and Hoda Mohammadzade},  
  year={2021},  
  publisher={Sharif University of Technology, Iran}  
}

contact
=======

For questions about our paper or code, please contact [[A.
Arefipour]{.underline}](mailto:arefipour.amirhossein@ee.sharif.ir)


