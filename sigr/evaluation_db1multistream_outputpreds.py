from __future__ import division
import os
import numpy as np
import scipy.io as sio
from functools import partial
from .parse_log import parse_log
from . import utils
from . import module_multistream
from logbook import Logger
from copy import deepcopy
import mxnet as mx


Exp = utils.Bunch

logger = Logger(__name__)

#feature_name = args.feature_name,
#window=args.window,
#num_semg_row = args.num_semg_row,
#num_semg_col = args.num_semg_col


@utils.cached(ignore=['context'])
def _crossval_predict_aux(self, Mod, get_crossval_val, fold, context, 
                          feature_name,
                          window,
                          num_semg_row,
                          num_semg_col,
                          dataset_args=None):
    Mod = deepcopy(Mod)
    Mod.update(context=context)
    mod = module_multistream.RuntimeModule(**Mod)
    Val = partial(
        get_crossval_val,
        fold=fold,
        batch_size=self.batch_size,
        window=window,
        feature_name=feature_name,
        num_semg_row=num_semg_row,
        num_semg_col=num_semg_col,
        **(dataset_args or {})
    )
#    print Val.name
    return mod.predict(utils.LazyProxy(Val))


@utils.cached(ignore=['context'])
def _crossval_predict_proba_aux(self, Mod, get_crossval_val, fold, context,  
                                feature_name,
                                window,
                                num_semg_row,
                                num_semg_col,
                                dataset_args=None):
    Mod = deepcopy(Mod)
    Mod.update(context=context)
    mod = module_multistream.RuntimeModule(**Mod)
    Val = partial(
        get_crossval_val,
        fold=fold,
        batch_size=self.batch_size,
        window=window,
        feature_name=feature_name,
        num_semg_row=num_semg_row,
        num_semg_col=num_semg_col,
        **(dataset_args or {})
    )
    return mod.predict_proba(utils.LazyProxy(Val))

def _crossval_predict_proba_generate_softmax_aux(self, Mod, get_crossval_val, fold, context, 
                                                 feature_name,
                                                 window,
                                                 num_semg_row,
                                                 num_semg_col,
                                                 dataset_args=None):
    Mod = deepcopy(Mod)
    Mod.update(context=context)
    mod = module_multistream.RuntimeModule(**Mod)
    
    print dataset_args
    Train,Val=get_crossval_val(
        fold=fold,
        batch_size=self.batch_size,
        window=window,
        feature_name=feature_name,
        num_semg_row=num_semg_row,
        num_semg_col=num_semg_col,
        **(dataset_args or {})
    )    
 

#    Train = Data[0]
#    Val = Data[1]

#    print Train.shape
#    print Val.shape    
    
#    Val = get_crossval_val(
#        fold=fold,
#        batch_size=self.batch_size,
#        window=mod.num_channel,
#        **(dataset_args or {})
#    )
    pred_prob_val, true_val, segment_val = mod.predict_proba(Val)
    
    pred_prob_train, true_train, segment_train = mod.predict_proba(Train)
    
#    return pred_prob_train, true_train, segment_train, Train, pred_prob_val, true_val, segment_val, Val
    
    return pred_prob_train, true_train, segment_train, pred_prob_val, true_val, segment_val
    
def _crossval_predict_proba_generate_softmax(self, **kargs):
#    proba = kargs.pop('proba', False)
    fold = int(kargs.pop('fold'))
    Mod = kargs.pop('Mod')
    Mod = deepcopy(Mod)
    Mod.update(params=self.format_params(Mod['params'], fold))
    context = Mod.pop('context', [mx.gpu(0)])
    #  import pickle
    #  d = kargs.copy()
    #  d.update(Mod=Mod, fold=fold)
    #  print(pickle.dumps(d))

    #  Ensure load from disk.
    #  Otherwise following cached methods like vote will have two caches,
    #  one for the first computation,
    #  and the other for the cached one.
#    func = _crossval_predict_proba_generate_softmax_aux
    pred_prob_train, true_train, segment_train, pred_prob_val, true_val, segment_val = _crossval_predict_proba_generate_softmax_aux(self, Mod=Mod, fold=fold, context=context, **kargs)
#    print pred_prob.shape
#    print pred_prob[1,:]
#    gesture = Val.gesture
#    subject = Val.subject
#    segment = Val.segment    
#    assert len(np.unique(subject)) == 1
# 
#    print segment.shape    
#    
    assert (len(true_train)==len(segment_train)==pred_prob_train.shape[0])    
    assert (len(true_val)==len(segment_val)==pred_prob_val.shape[0])

    save_root = "/home/weiwentao/public-2/wwt/PRL_chwise_multistream/ninapro-db1-ch_multistream-20-1-one-fold-intra-subject-fold-%d" % fold
    
    if os.path.isdir(save_root) is False:
           os.makedirs(save_root) 
    
    out_dir_train_prob = os.path.join(save_root, 'train_prob.mat')
    out_dir_train_true = os.path.join(save_root, 'train_true.mat')
    out_dir_train_segment = os.path.join(save_root, 'train_segment.mat')
    
    out_dir_test_prob = os.path.join(save_root, 'test_prob.mat')
    out_dir_test_true = os.path.join(save_root, 'test_true.mat')
    out_dir_test_segment = os.path.join(save_root, 'test_segment.mat')
    
    sio.savemat(out_dir_train_prob, {'data': pred_prob_train})
    sio.savemat(out_dir_train_true, {'data': true_train})
    sio.savemat(out_dir_train_segment, {'data': segment_train})    
    
    sio.savemat(out_dir_test_prob, {'data': pred_prob_val})
    sio.savemat(out_dir_test_true, {'data': true_val})    
    sio.savemat(out_dir_test_segment, {'data': segment_val})    

#    num_fea = 0
#
#    segment_unique = np.unique(segment)      
#    
#    
#    for i in range(len(segment_unique)):
#        itemindex = np.argwhere(segment==segment_unique[i])
#        itemindex = itemindex.reshape([itemindex.shape[0]])
#        seg_softmax = pred_prob[itemindex,:] 
##        seg_subject = subject[itemindex] 
#        seg_gesture = gesture[itemindex]   
#        
#        assert len(np.unique(seg_gesture)) == 1
#        
#        trial = segment_unique[i] - ((seg_gesture[0])*10) 
#        print  ("Saving Subject %d Gesture %d Trial %d soft max output..." % (fold, seg_gesture[0]+1, trial))
#        
#        out_dir = os.path.join(save_root,'{0:03d}',
#                                         '{1:03d}').format(int(fold), int(seg_gesture[0]+1))             
#        
#        if os.path.isdir(out_dir) is False:
#           os.makedirs(out_dir)                    
#        
#        out_path = os.path.join(out_dir,'{0:03d}_{1:03d}_{2:03d}.mat').format(int(fold), int(seg_gesture[0]+1), int(trial))    
#        sio.savemat(out_path, {'data': seg_softmax})   
#        num_fea = num_fea + seg_softmax.shape[0]
#            
#    assert (len(gesture)==len(subject)==len(segment)==pred_prob.shape[0]==num_fea)


def _crossval_predict(self, **kargs):
    proba = kargs.pop('proba', False)
    fold = int(kargs.pop('fold'))
    Mod = kargs.pop('Mod')
    Mod = deepcopy(Mod)
    Mod.update(params=self.format_params(Mod['params'], fold))
    context = Mod.pop('context', [mx.gpu(0)])
    window = kargs.pop('window')
    feature_name = kargs.pop('feature_name')
    num_semg_row=kargs.pop('num_semg_row')
    num_semg_col=kargs.pop('num_semg_col')
    #  import pickle
    #  d = kargs.copy()
    #  d.update(Mod=Mod, fold=fold)
    #  print(pickle.dumps(d))

    #  Ensure load from disk.
    #  Otherwise following cached methods like vote will have two caches,
    #  one for the first computation,
    #  and the other for the cached one.
    func = _crossval_predict_aux if not proba else _crossval_predict_proba_aux
    return func.call_and_shelve(self, Mod=Mod, fold=fold, context=context, 
                                                          window=window, 
                                                          feature_name=feature_name,
                                                          num_semg_row=num_semg_row,
                                                          num_semg_col=num_semg_col,
                                                          **kargs).get()


class Evaluation(object):

    def __init__(self, batch_size=None):
        self.batch_size = batch_size


class CrossValEvaluation(Evaluation):

    def __init__(self, **kargs):
        self.crossval_type = kargs.pop('crossval_type')
        super(CrossValEvaluation, self).__init__(**kargs)

    def get_crossval_val_func(self, dataset):
        return getattr(dataset, 'get_%s_val' % self.crossval_type.replace('-', '_'))
        
    def get_crossval_forsoftmaxgenerating_func(self, dataset):
        return getattr(dataset, 'get_%s_forsoftmaxgenerating' % self.crossval_type.replace('-', '_'))    

    def format_params(self, params, fold):
        try:
            return params % fold
        except:
            return params

    def transform(self, Mod, dataset, fold, dataset_args=None):
        get_crossval_val = self.get_crossval_val_func(dataset)
        pred, true, _ = _crossval_predict(
            self,
            proba=True,
            Mod=Mod,
            get_crossval_val=get_crossval_val,
            fold=fold,
            dataset_args=dataset_args)
        return pred, true

    def accuracy_mod(self, Mod, dataset, fold,
                     vote=False,
                     dataset_args=None,
                     balance=False):
        get_crossval_val = self.get_crossval_val_func(dataset)
        pred, true, segment = _crossval_predict(
            self,
            Mod=Mod,
            get_crossval_val=get_crossval_val,
            fold=fold,
            dataset_args=dataset_args)
        if vote:
            from .vote import vote as do
            return do(true, pred, segment, vote, balance)
        return (true == pred).sum() / true.size

    def accuracy_exp(self, exp, fold):
        if hasattr(exp, 'Mod') and hasattr(exp, 'dataset'):
            return self.accuracy_mod(Mod=exp.Mod,
                                     dataset=exp.dataset,
                                     fold=fold,
                                     vote=exp.get('vote', False),
                                     dataset_args=exp.get('dataset_args'))
        else:
            try:
                return parse_log(os.path.join(exp.root % fold, 'log')).val.iloc[-1]
            except:
                return np.nan

    def accuracy(self, **kargs):
        if 'exp' in kargs:
            return self.accuracy_exp(**kargs)
        elif 'Mod' in kargs:
            return self.accuracy_mod(**kargs)
        else:
            assert False

    def accuracies(self, exps, folds):
        acc = []
        for exp in exps:
            for fold in folds:
                acc.append(self.accuracy(exp=exp, fold=fold))
        return np.array(acc).reshape(len(exps), len(folds))

    def compare(self, exps, fold):
        acc = []
        for exp in exps:
            if hasattr(exp, 'Mod') and hasattr(exp, 'dataset'):
                acc.append(self.accuracy(Mod=exp.Mod,
                                         dataset=exp.dataset,
                                         fold=fold,
                                         vote=exp.get('vote', False),
                                         dataset_args=exp.get('dataset_args')))
            else:
                try:
                    acc.append(parse_log(os.path.join(exp.root % fold, 'log')).val.iloc[-1])
                except:
                    acc.append(np.nan)
        return acc

    def vote_accuracy_curves(self, exps, folds, windows, feature_name, window, num_semg_row, num_semg_col, balance=False):
        acc = []
        for exp in exps:
            for fold in folds:
                acc.append(self.vote_accuracy_curve(
                    Mod=exp.Mod,
                    dataset=exp.dataset,
                    fold=int(fold),
                    windows=windows,
                    feature_name=feature_name,
                    window=window,
                    num_semg_row=num_semg_row,
                    num_semg_col=num_semg_col,
                    dataset_args=exp.get('dataset_args'),
                    balance=balance))
        return np.array(acc).reshape(len(exps), len(folds), len(windows))

    def vote_accuracy_curve(self, Mod, dataset, fold, windows, feature_name, window, num_semg_row, num_semg_col,
                            dataset_args=None,
                            balance=False):
        get_crossval_val = self.get_crossval_val_func(dataset)
        pred, true, segment = _crossval_predict(
            self,
            Mod=Mod,
            get_crossval_val=get_crossval_val,
            fold=fold,
            feature_name=feature_name,
            window = window,
            num_semg_row = num_semg_row,
            num_semg_col = num_semg_col,
            dataset_args=dataset_args)
        from .vote import get_vote_accuracy_curve as do
        return do(true, pred, segment, windows, balance)[1]



    def output_softmax_preds(self, exps, folds, windows, feature_name, window, num_semg_row, num_semg_col,
                             balance=False):
        for exp in exps:
            for fold in folds:
                self.output_softmax_pred(
                                                Mod=exp.Mod,
                                                dataset=exp.dataset,
                                                fold=int(fold),
                                                windows=windows,
                                                feature_name=feature_name,
                                                window=window,
                                                num_semg_row=num_semg_row,
                                                num_semg_col=num_semg_col,
                                                dataset_args=exp.get('dataset_args'),
                                                balance=balance)
        return None

    def output_softmax_pred(self, Mod, dataset, fold, windows, feature_name, window, num_semg_row, num_semg_col,
                            dataset_args=None,
                            balance=False):
  
        get_crossval_forsoftmaxgenerating = self.get_crossval_forsoftmaxgenerating_func(dataset)
        
        _crossval_predict_proba_generate_softmax(
                self,
                Mod=Mod,
                get_crossval_val=get_crossval_forsoftmaxgenerating,
                fold=fold,
                feature_name=feature_name,
                window=window,
                num_semg_row = num_semg_row,
                num_semg_col= num_semg_col,
                dataset_args=dataset_args)  


def get_crossval_accuracies(crossval_type, exps, folds, batch_size=1000):
    acc = []
    evaluation = CrossValEvaluation(
        crossval_type=crossval_type,
        batch_size=batch_size
    )
    for fold in folds:
        acc.append(evaluation.compare(exps, fold))
    return acc


