from __future__ import print_function, division
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import mxnet as mx
from sigr.evaluation_db1multistream_outputpreds import CrossValEvaluation as CV, Exp
from sigr.data import Preprocess, Dataset
from sigr import Context

one_fold_intra_subject_eval = CV(crossval_type='one-fold-intra-subject', batch_size=1000)
intra_session_eval = CV(crossval_type='intra-session', batch_size=1000)

print('NinaPro DB1 Multistream')
print('===========')


semg_row = []
semg_col = []
num_ch = []             
for i in range(10):
    num_ch.append(1)    
    semg_row.append(1)             
    semg_col.append(20)
window=20
num_raw_semg_row=1
num_raw_semg_col=10
feature_name = 'ch_multistream'
fusion_type = 'fuse_5'

with Context(parallel=True, level='DEBUG'):
    acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('ninapro-db1-sigimg-fast'),
             dataset_args=dict(preprocess=Preprocess.parse('ninapro-lowpass')),
             Mod=dict(num_gesture=52,
                      context=[mx.gpu(0)],
                      multi_stream = True,
                      num_stream=len(semg_col),
                      symbol_kargs=dict(dropout=0, zscore=False, num_pixel=2, num_stream=len(semg_col), fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64),
                      params='.cache/ninapro-db1-ch_multistream-20-1-one-fold-intra-subject-fold-%d-v1.0.0.1/model-0028.params'))],
        folds=np.arange(27),
        windows=np.arange(1, 5),
        window=window,
        num_semg_row = num_raw_semg_row,
        num_semg_col = num_raw_semg_col,
        feature_name = feature_name,
        balance=True)
    acc = acc.mean(axis=(0, 1))
    
    
    kk = one_fold_intra_subject_eval.output_softmax_preds(
        [Exp(dataset=Dataset.from_name('ninapro-db1-sigimg-fast'),
             dataset_args=dict(preprocess=Preprocess.parse('ninapro-lowpass')),
             Mod=dict(num_gesture=52,
                      context=[mx.gpu(0)],
                      multi_stream = True,
                      num_stream=len(semg_col),
                      symbol_kargs=dict(dropout=0, zscore=False, num_pixel=2, num_stream=len(semg_col), fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64),
                      params='.cache/ninapro-db1-ch_multistream-20-1-one-fold-intra-subject-fold-%d-v1.0.0.1/model-0028.params'))],
        folds=np.arange(27),
        windows=np.arange(1, 5),
        window=window,
        num_semg_row = num_raw_semg_row,
        num_semg_col = num_raw_semg_col,
        feature_name = feature_name,
        balance=True)
        
    print('Single frame accuracy: %f' % acc[0])       
#    print('5 frames (50 ms) majority voting accuracy: %f' % acc[4])
#    print('10 frames (100 ms) majority voting accuracy: %f' % acc[9])
#    print('15 frames (150 ms) majority voting accuracy: %f' % acc[14])
#    print('20 frames (200 ms) majority voting accuracy: %f' % acc[19])
#    print('25 frames (250 ms) majority voting accuracy: %f' % acc[24])
    

#with Context(parallel=True, level='DEBUG'):
#    acc = one_fold_intra_subject_eval.accuracies(
#        [Exp(dataset=Dataset.from_name('ninapro-db1-sigimg-fast'), vote=-1,
#             dataset_args=dict(preprocess=Preprocess.parse('ninapro-lowpass')),
#             Mod=dict(num_gesture=52,
#                      context=[mx.gpu(0)],
#                      symbol_kargs=dict(dropout=0, num_semg_row=semg_row, num_semg_col=semg_col, num_filter=64),
#                      params='.cache/ninapro-db1-sigimg-1-1-one-fold-intra-subject-fold-%d-v1.0.0.6/model-0028.params'))],
#        folds=np.arange(27))
#    print('Per-trial majority voting accuracy: %f' % acc.mean())
