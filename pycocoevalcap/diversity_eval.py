__author__ = 'Qignzhong'
# https://github.com/qingzwang/DiversityMetrics/blob/master/pycocoevalcap/diversity_eval.py
# this source used another version of cider_scorer . but i change it according to the old one 

import os
import json
import pickle
import numpy as np

from collections import defaultdict
#from tokenizer.ptbtokenizer import PTBTokenizer
#from bleu.bleu import Bleu
#from meteor.meteor import Meteor
#from rouge.rouge import Rouge
#from cider.cider import Cider
#from spice.spice import Spice
#from cider_scorer2 import Cider
from .cider.cider_scorer import CiderScorer
from .bleu.bleu import Bleu
import nltk


class SelfCider:
    def __init__(self,num=20): # pathToData, candName, num=10, dfMode = "coco-val-df"
        """
        Reference file: list of dict('image_id': image_id, 'caption': caption).
        Candidate file: list of dict('image_id': image_id, 'caption': caption).
        :params: refName : name of the file containing references
        :params: candName: name of the file containing cnadidates
        """
        self.eval = {}
        # self._refName = refName
#        self._candName = candName
#        self._pathToData = pathToData
#        self._dfMode = dfMode
        self._num = num
#        if self._dfMode != 'corpus':
#            with open('./data/coco-train2014-df.p', 'r') as f:
#                self._df_file = pickle.load(f)

    def evaluate(self, references, candidates ):
#        """
#        Load the sentences from json files
#        """
#        def readJson():
#            path_to_cand_file = os.path.join(self._pathToData, self._candName)
#            cand_list = json.loads(open(path_to_cand_file, 'r').read())

#            res = defaultdict(list)

#            for id_cap in cand_list:
#                res[id_cap['image_id']].extend(id_cap['captions'])

#            return res

#        print('Loading Data...')
#        res = readJson()
#        # res = {
#        #     '0': [
#        #         'a zebra standing in the forest',
#        #         'a zebra standing near a tree in a field',
#        #         'a zebra standing on a lush dry grass field',
#        #         'a zebra standing on all four legs near trees and bushes with hills in the far distance',
#        #         'a zebra is standing in the grass near a tree'
#        #     ]
#        # }
#        # self._num=5

        refs = [sentences for k,sentences in references.items() ]
        cider = CiderScorer()
        for ss in refs:
          cider.cook_append(None, refs = ss)
        cider.compute_doc_freq()
        doc_freq = cider.document_frequency
        ref_len = np.log(float(len(cider.crefs))) # 

        print('ref_len', ref_len)
        print(list(doc_freq.items())[:5])
        
        ratio = {}
        avg_diversity = 0
        for im_id in candidates.keys():
            print( '\r', im_id, len(ratio), '\r',end='')
#            print ('number of images: {}\n'.format(len(ratio)))
            cov = np.zeros([self._num, self._num])
            for i in range(self._num):
                for j in range(i, self._num):
                    new_gts = []#{}
                    new_res = []
                    # new_res[im_id] = [{'caption': res[im_id][i]}]
                    # new_gts[im_id] = [{'caption': res[im_id][j]}]
                    new_res = candidates[im_id][i] #[im_id]
                    new_gts = [candidates[im_id][j]]
                    # new_res[im_id] = ['a group of people are playing football on a grass covered field']
                    # new_gts[im_id] = ['a group of people are playing football on a grass covered field',
                    #                   'a group of people are watching a football match']
                    # new_gts[im_id] = gt
                    # =================================================
                    # Set up scorers
                    # =================================================
                    # print 'tokenization...'
                    # tokenizer = PTBTokenizer()
                    # new_gts = tokenizer.tokenize(new_gts)
                    # new_res = tokenizer.tokenize(new_res)

                    # =================================================
                    # Set up scorers
                    # =================================================
#                    print ('setting up scorers...')
#                    scorers = [
#                        # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#                        # (Meteor(), "METEOR"),
#                        # (Rouge(), "ROUGE_L"),
#                        # TODO (Cider(self._dfMode, self._df_file), "CIDEr"),
#                        (Cider(),"CIDEr")
#                        # (Spice(), "SPICE")
#                    ]

                    # =================================================
                    # Compute scores
                    # =================================================
#                    for scorer, method in scorers:
# #                       print ('computing {} score...'.format(scorer.method()))
#                        score, scores = scorer.compute_score(new_gts, new_res)
                    cider = CiderScorer(new_res,new_gts ) 
                    cider.document_frequency = doc_freq
                    score = cider.compute_cider(ref_len )
                    score = np.mean(score )
                    
                    cov[i, j] = score
                    cov[j, i] = cov[i, j]
#                    print(score, end= ' ')
            # np.save('log_att_x0_c1_d0.npy', cov)
            u, s, v = np.linalg.svd(cov)
            s_sqrt = np.sqrt(s)
            r = max(s_sqrt) / s_sqrt.sum()
#            print('ratio={:.5}\n'.format(-np.log10(r) / np.log10(self._num)))
            ratio[im_id] = -np.log10(r) / np.log10(self._num)
            avg_diversity += -np.log10(r) / np.log10(self._num)
            if len(ratio) == 5000:
                print('errrorrr diversity_eval 140')  # seems unuseful
                break
        print('Average diversity: {:.5}'.format(avg_diversity / len(ratio)))
        self.eval = ratio
        return avg_diversity / len(ratio)

    def setEval(self, score, method):
        self.eval[method] = score












class mBLEU:
    def __init__(self, num=10):
        """
        Reference file: list of dict('image_id': image_id, 'caption': caption).
        Candidate file: list of dict('image_id': image_id, 'caption': caption).
        :params: refName : name of the file containing references
        :params: candName: name of the file containing cnadidates
        """
        self.eval = {}
        self._num = num

    def evaluate(self,ref,res):
        ratio = {}
        avg_diversity = 0
        for im_id in res.keys():
#            print ('number of images: {}\n'.format(len(ratio)))
            final_score = []
            for i in range(self._num):
                new_gts = {}
                new_res = {}
                new_res[im_id] = [res[im_id][i]]
                new_gts[im_id] = [res[im_id][j] for j in range(self._num) if j != i]
                # =================================================
                # Set up scorers
                # =================================================
#                print 'setting up scorers...'
                scorers = [
                    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                    # (Meteor(), "METEOR"),
                    # (Rouge(), "ROUGE_L"),
                    # (Cider(self._dfMode, self._df_file), "CIDEr"),
                    # (Spice(), "SPICE")
                ]

                # =================================================
                # Compute scores
                # =================================================
                for scorer, method in scorers:
#                    print 'computing %s score...'%(scorer.method())
                    score, scores = scorer.compute_score(gts=new_gts, res=new_res)
                final_score.append(score)
#                print('score',score)# 4 numbers
            mbleus = np.array(final_score).sum(0) / self._num
            ratio[im_id] = list(mbleus)
            avg_diversity += sum(mbleus) / 4
#            if len(ratio) == 5000:
#                break
#        print('Average diversity: {:.5f}'.format(1-avg_diversity / len(ratio)))
        self.eval = ratio
        return 1-avg_diversity / len(ratio)

    def setEval(self, score, method):
        self.eval[method] = score
        
