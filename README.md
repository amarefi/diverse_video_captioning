# diverse video captioning using LSTMs and POS

introduction
=======
In this thesis, we propose a method for describong short videos with diverse sentences. we use encoder-decoder framework as baseline, and use a method for embedding the sentence syntax in a vector to train the model.  
This syntex vector is either sentence length (1d vector) or POS vector(encoded with a variational auto-encoder-VAE).  
We use MSRVTT and MSVD datasets to test our method.   
code is available and requrements & tutorials are given below. 

requirements
========
+ python 3
+ tensorflow 2.1
+ Microsoft COCO evaluation library [link](https://github.com/tylin/coco-caption) (use cocoeval.py and pycocoevalcap folder in the same address as capsc13XX.py files)
+ 

how to use the code
========
hello 

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


