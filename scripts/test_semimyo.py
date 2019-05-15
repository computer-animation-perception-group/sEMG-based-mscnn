from __future__ import print_function, division
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import mxnet as mx
from sigr.evaluation_semimyo import CrossValEvaluation as CV, Exp
from sigr.data import Preprocess, Dataset
from sigr import Context


one_fold_intra_subject_eval = CV(crossval_type='one-fold-intra-subject', batch_size=1000)

with Context(parallel=True, level='DEBUG'):
    acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('ninapro-db1-raw/semg-glove'),
             dataset_args=dict(preprocess=Preprocess.parse('{ninapro-lowpass-parallel,identity,identity}')),
             Mod=dict(for_training=False,
                      context=[mx.gpu(1)],
                      symbol_kargs=dict(
                          dropout=0,
                          num_gesture=52,
                          num_glove=22,
                          num_semg_row=1,
                          num_semg_col=10,
                          glove_loss_weight=0.01,
                          num_glove_layer=2,
                          num_glove_hidden=128
                      ),
                      params='.cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-%d-v20161127.2/model-0028.params'))],
        folds=np.arange(27),
        windows=[1],
        balance=True)
    acc = acc.mean(axis=(0, 1))
    print('Single frame accuracy: %f' % acc[0])

#  with Context(parallel=True, level='DEBUG'):
    #  acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1-raw/semg-glove'),
             #  dataset_args=dict(preprocess=Preprocess.parse('{ninapro-lowpass-parallel,identity,identity}')),
             #  Mod=dict(for_training=False,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(
                          #  dropout=0,
                          #  num_gesture=52,
                          #  num_glove=22,
                          #  num_semg_row=1,
                          #  num_semg_col=10,
                          #  glove_loss_weight=0,
                          #  num_glove_layer=2,
                          #  num_glove_hidden=128
                      #  ),
                      #  params='.cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-%d-v20161127.1/model-0028.params'))],
        #  folds=np.arange(27),
        #  windows=[1],
        #  balance=True)
    #  acc = acc.mean(axis=(0, 1))
    #  print('Single frame accuracy: %f' % acc[0])

#  with Context(parallel=True, level='DEBUG'):
    #  acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1-raw/semg-glove'),
             #  dataset_args=dict(preprocess=Preprocess.parse('{ninapro-lowpass-parallel,identity,identity}')),
             #  Mod=dict(for_training=False,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(
                          #  dropout=0,
                          #  num_gesture=52,
                          #  num_glove=22,
                          #  num_semg_row=1,
                          #  num_semg_col=10,
                          #  glove_loss_weight=0.1,
                          #  num_glove_layer=2,
                          #  num_glove_hidden=128
                      #  ),
                      #  params='.cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-%d-v20161127.3/model-0028.params'))],
        #  folds=np.arange(27),
        #  windows=[1],
        #  balance=True)
    #  acc = acc.mean(axis=(0, 1))
    #  print('Single frame accuracy: %f' % acc[0])

#  with Context(parallel=True, level='DEBUG'):
    #  acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1-raw/semg-glove'),
             #  dataset_args=dict(preprocess=Preprocess.parse('{ninapro-lowpass-parallel,identity,identity}')),
             #  Mod=dict(for_training=False,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(
                          #  dropout=0,
                          #  num_gesture=52,
                          #  num_glove=22,
                          #  num_semg_row=1,
                          #  num_semg_col=10,
                          #  glove_loss_weight=0.01,
                          #  num_glove_layer=2,
                          #  num_glove_hidden=256
                      #  ),
                      #  params='.cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-%d-v20161127.4/model-0028.params'))],
        #  folds=np.arange(27),
        #  windows=[1],
        #  balance=True)
    #  acc = acc.mean(axis=(0, 1))
    #  print('Single frame accuracy: %f' % acc[0])

#  with Context(parallel=True, level='DEBUG'):
    #  acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1-raw/semg-glove'),
             #  dataset_args=dict(preprocess=Preprocess.parse('{ninapro-lowpass-parallel,identity,identity}')),
             #  Mod=dict(for_training=False,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(
                          #  dropout=0,
                          #  num_gesture=52,
                          #  num_glove=22,
                          #  num_semg_row=1,
                          #  num_semg_col=10,
                          #  glove_loss_weight=0.01,
                          #  num_glove_layer=4,
                          #  num_glove_hidden=128
                      #  ),
                      #  params='.cache/semimyo-ninapro-db1-raw-semg-glove-one-fold-intra-subject-%d-v20161127.5/model-0028.params'))],
        #  folds=np.arange(27),
        #  windows=[1],
        #  balance=True)
    #  acc = acc.mean(axis=(0, 1))
    #  print('Single frame accuracy: %f' % acc[0])
