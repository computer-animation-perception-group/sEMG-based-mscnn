from __future__ import print_function, division
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import mxnet as mx
import scipy.io as sio
from sigr.evaluation_dbcmultistream import CrossValEvaluation as CV, Exp
from sigr.data import Preprocess, Dataset
from sigr import Context

one_fold_intra_subject_eval = CV(crossval_type='one-fold-intra-subject', batch_size=1000)
intra_session_eval = CV(crossval_type='intra-session', batch_size=1000)

print('CSL Multistream')
print('===========')


#semg_row = []
#semg_col = []
#num_ch = []             
#for i in range(10):
#    num_ch.append(1)    
#    semg_row.append(1)             
#    semg_col.append(20)
#window=20
#num_raw_semg_row=1
#num_raw_semg_col=10
#feature_name = 'ch_multistream'
#fusion_type = 'fuse_2'
#
#with Context(parallel=True, level='DEBUG'):
#    acc = one_fold_intra_subject_eval.vote_accuracy_curves(
#        [Exp(dataset=Dataset.from_name('ninapro-db1-sigimg-fast'),
#             dataset_args=dict(preprocess=Preprocess.parse('ninapro-lowpass')),
#             Mod=dict(num_gesture=52,
#                      context=[mx.gpu(0)],
#                      multi_stream = True,
#                      num_stream=len(semg_col),
#                      symbol_kargs=dict(dropout=0, num_stream=len(semg_col), fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64),
#                      params='.cache/ninapro-db1-ch_multistream-20-1-one-fold-intra-subject-fold-%d-v1.0.0.1/model-0028.params'))],
#        folds=np.arange(27),
#        windows=np.arange(1, 5),
#        window=window,
#        num_semg_row = num_raw_semg_row,
#        num_semg_col = num_raw_semg_col,
#        feature_name = feature_name,
#        balance=True)
#    acc = acc.mean(axis=(0, 1))
#    print('Single frame accuracy: %f' % acc[0])
##    print('5 frames (50 ms) majority voting accuracy: %f' % acc[4])
##    print('10 frames (100 ms) majority voting accuracy: %f' % acc[9])
##    print('15 frames (150 ms) majority voting accuracy: %f' % acc[14])
##    print('20 frames (200 ms) majority voting accuracy: %f' % acc[19])
##    print('25 frames (250 ms) majority voting accuracy: %f' % acc[24])
    




#semg_row = []
#semg_col = []
#num_ch = []             
#for i in range(2):
#    num_ch.append(1)    
#    semg_row.append(1)             
#    semg_col.append(10)
#window=1
#num_raw_semg_row=1
#num_raw_semg_col=10
#feature_name = 'singleframe_multistream'
#fusion_type = 'multistream_multistruct_fuse_1'
#
#with Context(parallel=True, level='DEBUG'):
#    acc = one_fold_intra_subject_eval.vote_accuracy_curves(
#        [Exp(dataset=Dataset.from_name('ninapro-db1-sigimg-fast'),
#             dataset_args=dict(preprocess=Preprocess.parse('ninapro-lowpass')),
#             Mod=dict(num_gesture=52,
#                      context=[mx.gpu(0)],
#                      multi_stream = True,
#                      num_stream=len(semg_col),
#                      symbol_kargs=dict(dropout=0, num_stream=len(semg_col), fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64),
#                      params='.cache/TEST-ninapro-db1-singleframe_multistream-1-1-one-fold-intra-subject-fold-%d/model-0028.params'))],
#        folds=np.arange(27),
#        windows=np.arange(1, 501),
#        window=window,
#        num_semg_row = num_raw_semg_row,
#        num_semg_col = num_raw_semg_col,
#        feature_name = feature_name,
#        balance=True)
#    acc = acc.mean(axis=(0, 1))
#    print('Single frame accuracy: %f' % acc[0])
#    print('5 frames (50 ms) majority voting accuracy: %f' % acc[4])
#    print('10 frames (100 ms) majority voting accuracy: %f' % acc[9])
#    print('15 frames (150 ms) majority voting accuracy: %f' % acc[14])
#    print('20 frames (200 ms) majority voting accuracy: %f' % acc[19])
#    print('25 frames (250 ms) majority voting accuracy: %f' % acc[24])


semg_row = []
semg_col = []
num_ch = []             
for i in range(3):
    num_ch.append(1)    
    semg_row.append(8)             
    semg_col.append(7)
window=1
num_raw_semg_row=24
num_raw_semg_col=7
feature_name = 'piece_multistream'
fusion_type = 'fuse_5'

with Context(parallel=True, level='DEBUG'):
    acc = intra_session_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('csliter'),
             dataset_args=dict(preprocess=Preprocess.parse('(csl-cut,abs,ninapro-lowpass)')),
             Mod=dict(num_gesture=27,
                      adabn=True,
                      num_adabn_epoch=10,
                      context=[mx.gpu(1)],
                      multi_stream = True,
                      num_stream=len(semg_col),
                      symbol_kargs=dict(dropout=0, zscore=False, num_pixel=2, num_stream=len(semg_col), fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64),
                      params='.cache/CSL-piece_multistream-1-1-one-fold-intra-session-%d/model-0010.params'))],
        folds=np.arange(250),
        windows=np.arange(1, 2049),
        window=window,
        num_semg_row = num_raw_semg_row,
        num_semg_col = num_raw_semg_col,
        feature_name = feature_name,
        balance=True)
    acc = acc.mean(axis=(0, 1))
    print('Single frame accuracy: %f' % acc[0])
    print('307 frames majority voting accuracy: %f' % acc[306])
    print('350 frames majority voting accuracy: %f' % acc[349])
    print('614 frames majority voting accuracy: %f' % acc[613])

save_root = "/home/weiwentao/public-2/wwt/voting_acc/"    
if os.path.isdir(save_root) is False:
       os.makedirs(save_root) 
out_acc = os.path.join(save_root, 'csl_multistream.v.0.1.0.mat')   
sio.savemat(out_acc, {'acc': acc})   


with Context(parallel=True, level='DEBUG'):
    acc = intra_session_eval.accuracies(
        [Exp(dataset=Dataset.from_name('csliter'), vote=-1,
             dataset_args=dict(preprocess=Preprocess.parse('(csl-cut,abs,ninapro-lowpass)')),
             Mod=dict(num_gesture=27,
                      adabn=True,
                      num_adabn_epoch=10,
                      context=[mx.gpu(1)],
                      multi_stream = True,
                      num_stream=len(semg_col),
                      symbol_kargs=dict(dropout=0, zscore=False, num_pixel=2, num_stream=len(semg_col), fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64),
                      params='.cache/CSL-piece_multistream-1-1-one-fold-intra-session-%d/model-0010.params'))],
        folds=np.arange(250),
        window=window,
        num_semg_row = num_raw_semg_row,
        num_semg_col = num_raw_semg_col,
        feature_name = feature_name)
    print('Per-trial majority voting accuracy: %f' % acc.mean())










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
