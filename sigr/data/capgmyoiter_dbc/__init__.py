from __future__ import division
from nose.tools import assert_equal
from functools import partial
from logbook import Logger
from ...utils import cached
import os
import joblib as jb
import mxnet as mx
import numpy as np
import scipy.io as sio
import scipy.stats as sstat
from itertools import product
from collections import OrderedDict
#from lru import LRU
from .. import Combo, Trial
from ... import Context, constant
from nose.tools import assert_is_not_none
from piecewise_multistream_iter import CapgMyoPieceWiseMultiStreamData
from piecewise_multistream_iter_v2 import CapgMyoPieceWiseMultiStreamData_V2
from blockwise_multistream_iter import CapgMyoBlockWiseMultiStreamData
from blockwise_multistream_iter_v2 import CapgMyoBlockWiseMultiStreamData_V2
from piecesigimg_multistream_iter import CapgMyoPieceSigImgMultiStreamData
from piecewisetwoaxis_multistream_iter import CapgMyoTwoAxisPieceWiseMultiStreamData
from piecewise_plus_rawimg_multistream_iter import CapgMyoPiecePlusRawImgMultiStreamData
from single_frame_multistream_iter import CapgMyoSingleFrameMultiStreamData

from .. import Dataset as Base

#from dualCHstream_iter import CSLDualCHStreamData

TRIALS = list(range(1, 11))
NUM_TRIAL = len(TRIALS)
NUM_SEMG_ROW = 16
NUM_SEMG_COL = 8
FRAMERATE = 1000
PREPROCESS_KARGS = dict(
    framerate=FRAMERATE,
    num_semg_row=NUM_SEMG_ROW,
    num_semg_col=NUM_SEMG_COL
)

logger = Logger(__name__)



class Dataset(Base):

    name = 'capgmyoiter_dbc'
   
    num_semg_row = NUM_SEMG_ROW
    num_semg_col = NUM_SEMG_COL
       
   
    root = ('.cache/dbc')
          
#    subjects = list(range(27))
#    gestures = list(range(53))
#    trials = list(range(10))
          
    subjects = list(range(1, 11))
    gestures = list(range(1, 13))
    trials = TRIALS    

    def __init__(self, root):
        self.root = root
        
    @classmethod
    def parse(cls, text):
        if text == 'capgmyoiter_dbc':
           return cls(root = '.cache/dbc')
#        if cls is not Dataset and text == cls.name:
#            return cls(root=getattr(cls, 'root', '/home/weiwentao/public/duyu/misc/csl'))    
        
    def get_trial_func(self, *args,  **kargs):
        return GetTrial(*args, **kargs)    

#    @classmethod
#    def get_preprocess_kargs(cls):
#        return dict(
#            framerate=cls.framerate,
#            num_semg_row=cls.num_semg_row,
#            num_semg_col=cls.num_semg_col
#        )

#    def get_trial_func(self, *args, **kargs):
#        return GetTrial(*args, **kargs)

    def get_dataiter(self, get_trial, combos, feature_name, window=1, adabn=False, mean=None, scale=None, **kargs):
        
        print ("Use %s, semg row = %d semg col = %d window=%d" % (feature_name,
                                                                  self.num_semg_row,
                                                                  self.num_semg_col,
                                                                  window))             
        def data_scale(data):
            if mean is not None:
                data = data - mean
            if scale is not None:
                data = data * scale
            return data 
        
       
        combos = list(combos)
                
        data = []
        gesture = []
        subject = []
        segment = []            
  
        
        for combo in combos:
            trial = get_trial(self.root, combo=combo)
            data.append(data_scale(trial.data))
            gesture.append(trial.gesture)
            subject.append(trial.subject)
            segment.append(np.repeat(len(segment), len(data[-1])))  
        logger.debug('MAT loaded')
        
        if not data:
           logger.warn('Empty data')
           return
        
        index = []
        n = 0
        for seg in data:
            index.append(np.arange(n, n + len(seg) - window + 1))
#            index.append(np.arange(n, n + len(seg)))
            n += len(seg)
        index = np.hstack(index)
        logger.debug('Index made')
        
        logger.debug('Segments: {}', len(data))
        logger.debug('First segment shape: {}', data[0].shape)
        data = np.vstack(data).reshape(-1, 1, self.num_semg_row, self.num_semg_col)
        logger.debug('Data stacked')
        
        gesture = get_index(np.hstack(gesture))
        subject = get_index(np.hstack(subject))
        print np.unique(subject)
        segment = np.hstack(segment)
        
        label = []
        

        label.append(('gesture_softmax_label', gesture))
        

        
        logger.debug('Make data iter')
          
        if feature_name == 'rawimg_dy':  
                return CAPGMYODYImageData(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )
        elif feature_name == 'piece_multistream':
             return CapgMyoPieceWiseMultiStreamData(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )
        elif feature_name == 'piece_multistream_v2':
             return CapgMyoPieceWiseMultiStreamData_V2(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )        
                
        elif feature_name == 'piece_multistream_twoaxis':
             return CapgMyoTwoAxisPieceWiseMultiStreamData(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )      
        elif feature_name == 'piece_plusrawimg_multistream':
             return CapgMyoPiecePlusRawImgMultiStreamData(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )                       
        elif feature_name == 'piecesigimg_multistream':
             return CapgMyoPieceSigImgMultiStreamData(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )
                
        elif feature_name == 'singleframe_multistream':
             return CapgMyoSingleFrameMultiStreamData(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                ) 
        elif feature_name == 'block_multistream':
              return CapgMyoBlockWiseMultiStreamData(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )               
        elif feature_name == 'block_multistream_v2':
              return CapgMyoBlockWiseMultiStreamData_V2(
                        data=OrderedDict([('data', data)]),
                        label=OrderedDict(label),
                        gesture=gesture.copy(),
                        subject=subject.copy(),
                        segment=segment.copy(),
                        index=index,
                        adabn=adabn,
                        window=window,
                        num_gesture=gesture.max() + 1,
                        num_subject=subject.max() + 1,
                        **kargs
                )          
    
                
                
#        elif feature_name == 'rawimg':
#               return CSLImageData(
#                        data=OrderedDict([('data', data)]),
#                        label=OrderedDict(label),
#                        gesture=gesture.copy(),
#                        subject=subject.copy(),
#                        segment=segment.copy(),
#                        index=index,
#                        adabn=adabn,
#                        window=window,
#                        num_gesture=gesture.max() + 1,
#                        num_subject=subject.max() + 1,
#                        **kargs)
                
#        elif feature_name == 'dualCHstream':
#                return CSLDualCHStreamData(
#                        data=OrderedDict([('data', data)]),
#                        label=OrderedDict(label),
#                        gesture=gesture.copy(),
#                        subject=subject.copy(),
#                        segment=segment.copy(),
#                        index=index,
#                        adabn=adabn,
#                        window=window,
#                        num_gesture=gesture.max() + 1,
#                        num_subject=subject.max() + 1,
#                        **kargs
#                )
             
    def get_one_fold_intra_subject_trials(self):
        return self.trials[::2], self.trials[1::2]
    
    def get_inter_subject_data(self, fold, batch_size, preprocess,
                               adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train = load(
            combos=self.get_combos(product([i for i in self.subjects if i != subject],
                                           self.gestures, self.trials)),
            adabn=adabn,
            mini_batch_size=10 if minibatch else 1,
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures, self.trials)),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val
    
    def get_inter_subject_val(self, fold, batch_size,  feature_name, window, num_semg_row, num_semg_col, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        val = load(
            combos=self.get_combos(product([subject], self.gestures, self.trials)),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return val
    
    def get_intra_subject_data(self, fold, batch_size, preprocess,
                               adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold // self.num_trial]
        trial = self.trials[fold % self.num_trial]
        train = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in self.trials if i != trial])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures, [trial])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val
    
    def get_intra_subject_val(self, fold, batch_size, feature_name, window, num_semg_row, num_semg_col, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold // self.num_trial]
        trial = self.trials[fold % self.num_trial]
        val = load(
            combos=self.get_combos(product([subject], self.gestures, [trial])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return val
    
    def get_universal_intra_subject_data(self, fold, batch_size, preprocess,
                                         adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs): 
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        trial = self.trials[fold]
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in self.trials if i != trial])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures, [trial])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val
    
    def get_one_fold_intra_subject_val(self, fold, batch_size, feature_name, window, num_semg_row, num_semg_col, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)             
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        _, val_trials = self.get_one_fold_intra_subject_trials()
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return val
        
   
        
    
    def get_one_fold_intra_subject_data(self, fold, batch_size, preprocess,
                                        adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs):
           
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        train = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        print 'Data loading finished!'    
        return train, val
        
    def get_one_fold_intra_subject_forsoftmaxgenerating(self, fold, batch_size, preprocess,
                                                        feature_name, window, num_semg_row, num_semg_col, **kargs):
           
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        print preprocess
        Train = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in train_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        Val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
               
        print 'Data loading finished!'    
        return Train, Val  
    
    def get_universal_one_fold_intra_subject_data(self, fold, batch_size, preprocess,
                                                  adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, **kargs):
        assert_equal(fold, 0)
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name = feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val
        


class CAPGMYODYImageData(mx.io.NDArrayIter):

    def __init__(self, *args, **kargs):
        print 'Initialization Data Iter!'
        self.random_shift_vertical = kargs.pop('random_shift_vertical', 0)
        self.random_shift_horizontal = kargs.pop('random_shift_horizontal', 0)
        self.random_shift_fill = kargs.pop('random_shift_fill', constant.RANDOM_SHIFT_FILL)
        self.amplitude_weighting = kargs.pop('amplitude_weighting', False)
        self.amplitude_weighting_sort = kargs.pop('amplitude_weighting_sort', False)
        self.framerate = kargs.pop('framerate', 1000)
        self.downsample = kargs.pop('downsample', None)
        self.shuffle = kargs.pop('shuffle', False)
        self.adabn = kargs.pop('adabn', False)
        self._gesture = kargs.pop('gesture')
        self._subject = kargs.pop('subject')
        self._segment = kargs.pop('segment')
        self._index_orig = kargs.pop('index')
        self._index = np.copy(self._index_orig)
        self.num_gesture = kargs.pop('num_gesture')
        self.num_subject = kargs.pop('num_subject')
        self.mini_batch_size = kargs.pop('mini_batch_size', kargs.get('batch_size'))
        self.random_state = kargs.pop('random_state', np.random)
        self.balance_gesture = kargs.pop('balance_gesture', 0)
        self.num_channel = 1    
        self.window = kargs.pop('window', 0)
        self.num_semg_col = kargs.pop('num_semg_col', 0)
        self.num_semg_row = kargs.pop('num_semg_row', 0)
        super(CAPGMYODYImageData, self).__init__(*args, **kargs)

        self.num_channel = self.window
        self.num_data = len(self._index)
        self.data_orig = self.data
        self.reset()
        

    @property
    def num_sample(self):
        return self.num_data

    @property
    def gesture(self):
        return self._gesture[self._index]

    @property
    def subject(self):
        return self._subject[self._index]

    @property
    def segment(self):
        return self._segment[self._index]

    @property
    def provide_data(self):
         res = [(k, tuple([self.batch_size, self.num_channel] + list(v.shape[2:]))) for k, v in self.data]
#         res = [(k, tuple([self.batch_size, self.num_channel] + list([self.window, len(genIndex(self.num_semg_col*self.num_semg_row))-1]))) for k, v in self.data]
         print res         
         return res

    def _expand_index(self, index):
        return np.hstack([np.arange(i, i + self.window) for i in index])

    def _reshape_data(self, data):
        return data.reshape(-1, self.window, *data.shape[2:])

#    def _get_sigimg(self, data):
#        
#        from ... import Context
#        import joblib as jb
#        res = []
#         
#        for amp in Context.parallel(jb.delayed(_get_sigimg_aux)(sample) for sample in data):
#            res.append(amp[np.newaxis, ...])
#        return np.concatenate(res, axis=0)

    def _get_segments(self, a, index):
        b = mx.nd.empty((len(index), self.window) + a.shape[2:], dtype=a.dtype)  
        for i, j in enumerate(index):           
            b[i] = a[j:j + self.window].reshape(self.window, *a.shape[2:])
        return b

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only"""
        assert(self.cursor < self.num_data), "DataIter needs reset."
                  
        if data_source is self.data and self.window > 1:  
            if self.cursor + self.batch_size <= self.num_data:
                #  res = [self._reshape_data(x[1][self._expand_index(self._index[self.cursor:self.cursor+self.batch_size])]) for x in data_source]
                res = [self._get_segments(x[1], self._index[self.cursor:self.cursor+self.batch_size]) for x in data_source]
            else:
                pad = self.batch_size - self.num_data + self.cursor
                res = [(np.concatenate((self._reshape_data(x[1][self._expand_index(self._index[self.cursor:])]),
                                        self._reshape_data(x[1][self._expand_index(self._index[:pad])])), axis=0)) for x in data_source]
        else:
            if self.cursor + self.batch_size <= self.num_data:
                res = [(x[1][self._index[self.cursor:self.cursor+self.batch_size]]) for x in data_source]
            else:
                pad = self.batch_size - self.num_data + self.cursor
                res = [(np.concatenate((x[1][self._index[self.cursor:]], x[1][self._index[:pad]]), axis=0)) for x in data_source] 
                
                
#        if data_source is self.data:       
#             new_res = []           
#             for a in res:
#                 new_res.append(a.asnumpy() if isinstance(a, mx.nd.NDArray) else a)
#             res = new_res             
#             
#             res = [a.reshape(a.shape[0], a.shape[1], -1) for a in res]
#             res = [self._get_sigimg(a.asnumpy() if isinstance(a, mx.nd.NDArray) else a) for a in res]
#             res = [a.reshape(a.shape[0], 1, a.shape[1], -1) for a in res]

        res = [a if isinstance(a, mx.nd.NDArray) else mx.nd.array(a) for a in res]
        return res

    def _rand(self, smin, smax, shape):
        return (smax - smin) * self.random_state.rand(*shape) + smin

    def _do_shuffle(self):
        if not self.adabn or len(set(self._subject)) == 1:
            self.random_state.shuffle(self._index)
        else:
            batch_size = self.mini_batch_size
            # batch_size = self.batch_size
            # logger.info('AdaBN shuffle with a mini batch size of {}', batch_size)
            self.random_state.shuffle(self._index)
            subject_shuffled = self._subject[self._index]
            index_batch = []
            for i in sorted(set(self._subject)):
                index = self._index[subject_shuffled == i]
                index = index[:len(index) // batch_size * batch_size]
                index_batch.append(index.reshape(-1, batch_size))
            index_batch = np.vstack(index_batch)
            index = np.arange(len(index_batch))
            self.random_state.shuffle(index)
            self._index = index_batch[index, :].ravel()
            #  assert len(self._index) == len(set(self._index))

            for i in range(0, len(self._subject), batch_size):
                # Make sure that the samples in one batch are from the same subject
                assert np.all(self._subject[self._index[i:i + batch_size - 1]] ==
                              self._subject[self._index[i + 1:i + batch_size]])

            if batch_size != self.batch_size:
                assert self.batch_size % batch_size == 0
                #  assert (self.batch_size // batch_size) % self.num_subject == 0
                self._index = self._index[:len(self._index) // self.batch_size * self.batch_size].reshape(
                    -1, self.batch_size // batch_size, batch_size).transpose(0, 2, 1).ravel()

    def reset(self):
        self._reset()
        super(CAPGMYODYImageData, self).reset()

    def _reset(self):
        self._index = np.copy(self._index_orig)

#        if self.amplitude_weighting:
#            assert np.all(self._index[:-1] < self._index[1:])
#            if not hasattr(self, 'amplitude_weight'):
#                self.amplitude_weight = get_amplitude_weight(
#                    self.data[0][1], self._segment, self.framerate)
#            if self.shuffle:
#                random_state = self.random_state
#            else:
#                random_state = np.random.RandomState(677)
#            self._index = random_state.choice(
#                self._index, len(self._index), p=self.amplitude_weight)
#            if self.amplitude_weighting_sort:
#                logger.debug('Amplitude weighting sort')
#                self._index.sort()

        if self.downsample:
            samples = np.arange(len(self._index))
            np.random.RandomState(667).shuffle(samples)
            assert self.downsample > 0 and self.downsample <= 1
            samples = samples[:int(np.round(len(samples) * self.downsample))]
            assert len(samples) > 0
            self._index = self._index[samples]

        if self.balance_gesture:
            num_sample_per_gesture = int(np.round(self.balance_gesture *
                                                  len(self._index) / self.num_gesture))
            choice = []
            for gesture in set(self.gesture):
                mask = self._gesture[self._index] == gesture
                choice.append(self.random_state.choice(np.where(mask)[0],
                                                       num_sample_per_gesture))
            choice = np.hstack(choice)
            self._index = self._index[choice]

        if self.shuffle:
            self._do_shuffle()

        self.num_data = len(self._index)


class GetTrial(object):

    def __init__(self, gestures, trials, preprocess=None):
        self.preprocess = preprocess
        self.memo = {}
        self.gesture_and_trials = list(product(gestures, trials))

    def get_path(self, root, combo):
        return os.path.join(
            root,
            '{c.subject:03d}-{c.gesture:03d}-{c.trial:03d}.mat'.format(c=combo))

    def __call__(self, root, combo):
        path = self.get_path(root, combo)
        if path not in self.memo:
            logger.debug('Load subject {}', combo.subject)
            paths = [self.get_path(root, Combo(combo.subject, gesture, trial))
                     for gesture, trial in self.gesture_and_trials]
            self.memo.update({path: data for path, data in
                              zip(paths, _get_data(paths, self.preprocess))})
        data = self.memo[path]
        data = data.copy()
        gesture = np.repeat(combo.gesture, len(data))
        subject = np.repeat(combo.subject, len(data))
        return Trial(data=data, gesture=gesture, subject=subject)


@cached
def _get_data(paths, preprocess):
    #  return list(Context.parallel(
        #  jb.delayed(_get_data_aux)(path, preprocess) for path in paths))
    return [_get_data_aux(path, preprocess) for path in paths]


def _get_data_aux(path, preprocess):
    data = sio.loadmat(path)['data'].astype(np.float32)
    if preprocess:
        data = preprocess(data, **PREPROCESS_KARGS)
    return data


#  @cached
#  def _get_data(path, bandstop, cut, downsample):
    #  data = sio.loadmat(path)['gestures']
    #  data = [np.transpose(np.delete(segment.astype(np.float32), np.s_[7:192:8], 0))
            #  for segment in data.flat]
    #  if bandstop:
        #  data = list(Context.parallel(jb.delayed(get_bandstop)(segment) for segment in data))
    #  if cut is not None:
        #  data = list(Context.parallel(jb.delayed(cut)(segment, framerate=FRAMERATE) for segment in data))
    #  if downsample > 1:
        #  data = [segment[::downsample].copy() for segment in data]
    #  return data





def get_index(a):
    '''Convert label to 0 based index'''
    b = list(set(a))
    return np.array([x if x < 0 else b.index(x) for x in a.ravel()]).reshape(a.shape)

from . import capgmyoiter_dbb
assert capgmyoiter_dbb
