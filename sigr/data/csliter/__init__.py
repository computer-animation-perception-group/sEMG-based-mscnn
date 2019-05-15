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
from itertools import product, izip
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
#from lru import LRU
from .. import Combo, Trial
from ... import Context, constant
from nose.tools import assert_is_not_none
from csl_piecewise_multistream_iter import CSLPieceWiseMultiStreamData
from csl_piecewise_multistream_iter_v2 import CSLPieceWiseMultiStreamData_V2
from csl_piecewise_plusrawimg_multistream_iter import CSLPieceWisePlusRawImgMultiStreamData
from csl_blockwise_multistream_iter import CSLBlockWiseMultiStreamData
from csl_blockwise_diff_multistream_iter import CSLBlockWiseDiffMultiStreamData
from csl_blockwise_plusrawimg_multistream_iter import CSLBlockWisePlusRawImgMultiStreamData
from csl_blockwise_multistream_iter_v2 import CSLBlockWiseMultiStreamData_V2
from csl_blockpiecewise_multistream_iter import CSLBlockPieceWiseMultiStreamData
from .. import Dataset as Base

#from dualCHstream_iter import CSLDualCHStreamData

logger = Logger(__name__)


NUM_TRIAL = 10
SUBJECTS = list(range(1, 6))
SESSIONS = list(range(1, 6))
NUM_SESSION = len(SESSIONS)
NUM_SUBJECT = len(SUBJECTS)
NUM_SUBJECT_AND_SESSION = len(SUBJECTS) * NUM_SESSION
SUBJECT_AND_SESSIONS = list(range(1, NUM_SUBJECT_AND_SESSION + 1))
GESTURES = list(range(27))
REST_TRIALS = [x - 1 for x in [2, 4, 7, 8, 11, 13, 19, 25, 26, 30]]
NUM_SEMG_ROW = 24
NUM_SEMG_COL = 7
FRAMERATE = 2048
framerate = FRAMERATE
TRIALS = list(range(NUM_TRIAL))
PREPROCESS_KARGS = dict(
    framerate=FRAMERATE,
    num_semg_row=NUM_SEMG_ROW,
    num_semg_col=NUM_SEMG_COL
)


class Dataset(Base):

    name = 'csliter'
   
    num_semg_row = NUM_SEMG_ROW
    num_semg_col = NUM_SEMG_COL
       
   
    root = ('.cache/csl')
          
#    subjects = list(range(27))
#    gestures = list(range(53))
#    trials = list(range(10))

    def __init__(self, root):
        self.root = root
        
    @classmethod
    def parse(cls, text):
        if text == 'csliter':
           return cls(root = '.cache/csl')
#        if cls is not Dataset and text == cls.name:
#            return cls(root=getattr(cls, 'root', '/home/weiwentao/public/duyu/misc/csl'))    
        
    def get_trial_func(self, **kargs):
        return GetTrial(self.root, self.gestures, self.trials, **kargs)    

#    @classmethod
#    def get_preprocess_kargs(cls):
#        return dict(
#            framerate=cls.framerate,
#            num_semg_row=cls.num_semg_row,
#            num_semg_col=cls.num_semg_col
#        )

#    def get_trial_func(self, *args, **kargs):
#        return GetTrial(*args, **kargs)

    def get_dataiter(self, get_trial, combos, feature_name, window=1, adabn=False, mean=None, scale=None, scaler=None, **kargs):
        
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
        segment = np.hstack(segment)
        
        label = []
        

        label.append(('gesture_softmax_label', gesture))
        
        assert (feature_name == 'rawimg_dy' or 
                feature_name == 'piece_multistream_v2' or 
                feature_name == 'piece_multistream' or 
                feature_name == 'piece_plusrawimg_multistream' or
                feature_name == 'block_multistream' or
                feature_name == 'block_diff_multistream' or
                feature_name == 'block_multistream_v2' or
                feature_name == 'block_piece_multistream' or
                feature_name == 'block_plusrawimg_multistream')
        
        logger.debug('Make data iter')
         
         
        if feature_name == 'rawimg_dy':  
                data = CSLDYImageData(
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
               data = CSLPieceWisePlusRawImgMultiStreamData(
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
                        **kargs)
        elif feature_name == 'block_plusrawimg_multistream':
               data = CSLBlockWisePlusRawImgMultiStreamData(
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
                        **kargs)
        elif feature_name == 'block_multistream':
              
#            if scaler is None:              
#              shape_1 = data.shape[1]
#              shape_2 = data.shape[2]
#              shape_3 = data.shape[3]
#              data = data.reshape(data.shape[0], -1)
#              print 'Normalize training set'
#              print data.shape
#              data = StandardScaler().fit_transform(data)
#              data = data.reshape(data.shape[0], shape_1, shape_2, shape_3)
#            else:
#              shape_1 = data.shape[1]
#              shape_2 = data.shape[2]
#              shape_3 = data.shape[3]
#              data = data.reshape(data.shape[0], -1)
#              print 'Normalize testing set'
#              print data.shape
#              data = scaler.transform(data)
#              data = data.reshape(data.shape[0], shape_1, shape_2, shape_3)
                          
            
            data = CSLBlockWiseMultiStreamData(
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
                        **kargs)
        elif feature_name == 'block_diff_multistream':
               data = CSLBlockWiseDiffMultiStreamData(
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
                        **kargs)          
        elif feature_name == 'block_piece_multistream':
               data = CSLBlockPieceWiseMultiStreamData(
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
                        **kargs)                                                
        elif feature_name == 'block_multistream_v2':
               data = CSLBlockWiseMultiStreamData_V2(
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
                        **kargs)                   
        elif feature_name == 'piece_multistream':
                data = CSLPieceWiseMultiStreamData(
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
                data = CSLPieceWiseMultiStreamData_V2(
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
        
        data = Preload(data)
        return data
             
    def get_general_data(self, batch_size, adabn, minibatch, downsample, feature_name, window, **kargs):
                get_trial = GetTrial(downsample=downsample)
#                get_trial = self.get_trial_func(preprocess=preprocess, norest=True)
                load = partial(self.get_dataiter,
                               get_trial=get_trial,
                               framerate=FRAMERATE,
                               last_batch_handle='pad',
                               batch_size=batch_size,
                               num_semg_row=NUM_SEMG_ROW,
                               num_semg_col=NUM_SEMG_COL)
                train = load(combos=get_combos(product(SUBJECT_AND_SESSIONS, GESTURES[1:], range(0, NUM_TRIAL, 2)),
                                               product(SUBJECT_AND_SESSIONS, GESTURES[:1], REST_TRIALS[0::2])),
                             adabn=adabn,
                             shuffle=True,
                             random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                             random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                             random_shift_vertical=kargs.get('random_shift_vertical', 0),
                             mini_batch_size=batch_size // (NUM_SUBJECT_AND_SESSION if minibatch else 1),
                             feature_name = feature_name,
                             window=window)
                logger.debug('Training set loaded')
                val = load(combos=get_combos(product(SUBJECT_AND_SESSIONS, GESTURES[1:], range(1, NUM_TRIAL, 2)),
                                             product(SUBJECT_AND_SESSIONS, GESTURES[:1], REST_TRIALS[1::2])),
                           shuffle=False)
                logger.debug('Test set loaded')
                return train, val
        
        
    def get_intra_session_val(self, fold, batch_size, preprocess, feature_name, window,  **kargs):
                get_trial = GetTrial(preprocess=preprocess)
                load = partial(self.get_dataiter,
                               get_trial=get_trial,
                               amplitude_weighting=kargs.get('amplitude_weighting', False),
                               amplitude_weighting_sort=kargs.get('amplitude_weighting_sort', False),
                               framerate=FRAMERATE,
                               last_batch_handle='pad',
                               batch_size=batch_size,
                               num_semg_row=NUM_SEMG_ROW,
                               num_semg_col=NUM_SEMG_COL,
                               random_state=np.random.RandomState(42))
                subject = fold // (NUM_SESSION * NUM_TRIAL) + 1
                session = fold // NUM_TRIAL % NUM_SESSION + 1
                fold = fold % NUM_TRIAL
                val = load(combos=get_combos(product([encode_subject_and_session(subject, session)],
                                                     GESTURES[1:], [fold]),
                                             product([encode_subject_and_session(subject, session)],
                                                     GESTURES[:1], REST_TRIALS[fold:fold + 1])),
                           shuffle=False,
                           feature_name = feature_name,
                           window=window)
                return val
        
        
    def get_universal_intra_session_data(self, fold, batch_size, preprocess, balance_gesture, feature_name, window,  **kargs):
                get_trial = GetTrial(preprocess=preprocess)
                load = partial(self.get_dataiter,
                               get_trial=get_trial,
                               amplitude_weighting=kargs.get('amplitude_weighting', False),
                               amplitude_weighting_sort=kargs.get('amplitude_weighting_sort', False),
                               framerate=FRAMERATE,
                               last_batch_handle='pad',
                               batch_size=batch_size,
                               num_semg_row=NUM_SEMG_ROW,
                               num_semg_col=NUM_SEMG_COL)
                trial = fold
                train = load(combos=get_combos(product(SUBJECT_AND_SESSIONS,
                                                       GESTURES[1:], [i for i in range(NUM_TRIAL) if i != trial]),
                                               product(SUBJECT_AND_SESSIONS,
                                                       GESTURES[:1], [REST_TRIALS[i] for i in range(NUM_TRIAL) if i != trial])),
                             balance_gesture=balance_gesture,
                             random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                             random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                             random_shift_vertical=kargs.get('random_shift_vertical', 0),
                             shuffle=True,
                             feature_name=feature_name,
                             window=window)
                assert_is_not_none(train)
                logger.debug('Training set loaded')
                val = load(combos=get_combos(product(SUBJECT_AND_SESSIONS,
                                                     GESTURES[1:], [trial]),
                                             product(SUBJECT_AND_SESSIONS,
                                                     GESTURES[:1], REST_TRIALS[trial:trial + 1])),
                           shuffle=False,
                           feature_name=feature_name,
                           window=window)
                logger.debug('Test set loaded')
                assert_is_not_none(val)
                return train, val
        
        
    def get_intra_session_data(self,  fold, batch_size, preprocess, balance_gesture, feature_name, window, **kargs):
                get_trial = GetTrial(preprocess=preprocess)
                load = partial(self.get_dataiter,
                               get_trial=get_trial,
                               amplitude_weighting=kargs.get('amplitude_weighting', False),
                               amplitude_weighting_sort=kargs.get('amplitude_weighting_sort', False),
                               framerate=FRAMERATE,
                               last_batch_handle='pad',
                               batch_size=batch_size,
                               num_semg_row=NUM_SEMG_ROW,
                               num_semg_col=NUM_SEMG_COL)
                subject = fold // (NUM_SESSION * NUM_TRIAL) + 1
                session = fold // NUM_TRIAL % NUM_SESSION + 1
                fold = fold % NUM_TRIAL
                train = load(combos=get_combos(product([encode_subject_and_session(subject, session)],
                                                       GESTURES[1:], [f for f in range(NUM_TRIAL) if f != fold]),
                                               product([encode_subject_and_session(subject, session)],
                                                       GESTURES[:1], [REST_TRIALS[f] for f in range(NUM_TRIAL) if f != fold])),
                             balance_gesture=balance_gesture,
                             random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                             random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                             random_shift_vertical=kargs.get('random_shift_vertical', 0),
                             shuffle=True,
                             feature_name=feature_name,
                             window=window)
                assert_is_not_none(train)
                logger.debug('Training set loaded')
                val = load(combos=get_combos(product([encode_subject_and_session(subject, session)],
                                                     GESTURES[1:], [fold]),
                                             product([encode_subject_and_session(subject, session)],
                                                     GESTURES[:1], REST_TRIALS[fold:fold + 1])),
                           shuffle=False,
                           feature_name=feature_name,
                           window=window)
                logger.debug('Test set loaded')
                assert_is_not_none(val)
                return train, val
            
        
    def get_inter_session_data(self, fold, batch_size, preprocess, adabn, minibatch, balance_gesture, feature_name, window, **kargs):
                #  TODO: calib
                get_trial = GetTrial(preprocess=preprocess)
                load = partial(self.get_dataiter,
                               get_trial=get_trial,
                               framerate=FRAMERATE,
                               last_batch_handle='pad',
                               batch_size=batch_size,
                               num_semg_row=NUM_SEMG_ROW,
                               num_semg_col=NUM_SEMG_COL)
                subject = fold // NUM_SESSION + 1
                session = fold % NUM_SESSION + 1
                train = load(combos=get_combos(product([encode_subject_and_session(subject, i) for i in SESSIONS if i != session],
                                                       GESTURES[1:], TRIALS),
                                               product([encode_subject_and_session(subject, i) for i in SESSIONS if i != session],
                                                       GESTURES[:1], REST_TRIALS)),
                             adabn=adabn,
                             mini_batch_size=batch_size // (NUM_SESSION - 1 if minibatch else 1),
                             balance_gesture=balance_gesture,
                             random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                             random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                             random_shift_vertical=kargs.get('random_shift_vertical', 0),
                             shuffle=True,
                             feature_name=feature_name,
                             window=window)
                logger.debug('Training set loaded')
                val = load(combos=get_combos(product([encode_subject_and_session(subject, session)],
                                                     GESTURES[1:], TRIALS),
                                             product([encode_subject_and_session(subject, session)],
                                                     GESTURES[:1], REST_TRIALS)),
                           shuffle=False,
                           feature_name=feature_name,
                           window=window)
                logger.debug('Test set loaded')
                return train, val
        
        
    def get_inter_session_val(self, fold, batch_size, preprocess, feature_name, window,  **kargs):
                #  TODO: calib
                get_trial = GetTrial(preprocess=preprocess)
                load = partial(self.get_dataiter,
                               get_trial=get_trial,
                               framerate=FRAMERATE,
                               last_batch_handle='pad',
                               batch_size=batch_size,
                               num_semg_row=NUM_SEMG_ROW,
                               num_semg_col=NUM_SEMG_COL,
                               random_state=np.random.RandomState(42))
                subject = fold // NUM_SESSION + 1
                session = fold % NUM_SESSION + 1
                val = load(combos=get_combos(product([encode_subject_and_session(subject, session)],
                                                     GESTURES[1:], TRIALS),
                                             product([encode_subject_and_session(subject, session)],
                                                     GESTURES[:1], REST_TRIALS)),
                           shuffle=False,
                           feature_name=feature_name,
                           window=window)
                return val
        
        
    def get_universal_inter_session_data(self, fold, batch_size, preprocess, adabn, minibatch, balance_gesture, feature_name, window, **kargs):
                #  TODO: calib
                get_trial = GetTrial(preprocess=preprocess)
                load = partial(self.get_dataiter,
                               get_trial=get_trial,
                               framerate=FRAMERATE,
                               last_batch_handle='pad',
                               batch_size=batch_size,
                               num_semg_row=NUM_SEMG_ROW,
                               num_semg_col=NUM_SEMG_COL)
                session = fold + 1
                train = load(combos=get_combos(product([encode_subject_and_session(s, i) for s, i in
                                                        product(SUBJECTS, [i for i in SESSIONS if i != session])],
                                                       GESTURES[1:], TRIALS),
                                               product([encode_subject_and_session(s, i) for s, i in
                                                        product(SUBJECTS, [i for i in SESSIONS if i != session])],
                                                       GESTURES[:1], REST_TRIALS)),
                             adabn=adabn,
                             mini_batch_size=batch_size // (NUM_SUBJECT * (NUM_SESSION - 1) if minibatch else 1),
                             balance_gesture=balance_gesture,
                             random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                             random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                             random_shift_vertical=kargs.get('random_shift_vertical', 0),
                             shuffle=True,
                             feature_name=feature_name,
                             window=window)
                logger.debug('Training set loaded')
                val = load(combos=get_combos(product([encode_subject_and_session(s, session) for s in SUBJECTS],
                                                     GESTURES[1:], TRIALS),
                                             product([encode_subject_and_session(s, session) for s in SUBJECTS],
                                                     GESTURES[:1], REST_TRIALS)),
                           adabn=adabn,
                           mini_batch_size=batch_size // (NUM_SUBJECT if minibatch else 1),
                           shuffle=False,
                           feature_name=feature_name,
                           window=window)
                logger.debug('Test set loaded')
                return train, val
        
        
    def get_intra_subject_data(self, fold, batch_size, cut, bandstop, adabn, minibatch, feature_name, window, **kargs):
                get_trial = GetTrial(cut=cut, bandstop=bandstop)
                load = partial(self.get_dataiter,
                               get_trial=get_trial,
                               framerate=FRAMERATE,
                               last_batch_handle='pad',
                               batch_size=batch_size,
                               num_semg_row=NUM_SEMG_ROW,
                               num_semg_col=NUM_SEMG_COL)
                subject = fold // NUM_TRIAL + 1
                fold = fold % NUM_TRIAL
                train = load(combos=get_combos(product([encode_subject_and_session(subject, session) for session in SESSIONS],
                                                       GESTURES[1:], [f for f in range(NUM_TRIAL) if f != fold]),
                                               product([encode_subject_and_session(subject, session) for session in SESSIONS],
                                                       GESTURES[:1], [REST_TRIALS[f] for f in range(NUM_TRIAL) if f != fold])),
                             adabn=adabn,
                             mini_batch_size=batch_size // (NUM_SESSION if minibatch else 1),
                             random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                             random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                             random_shift_vertical=kargs.get('random_shift_vertical', 0),
                             shuffle=True,
                             feature_name=feature_name,
                             window=window)
                logger.debug('Training set loaded')
                val = load(combos=get_combos(product([encode_subject_and_session(subject, session) for session in SESSIONS],
                                                     GESTURES[1:], [fold]),
                                             product([encode_subject_and_session(subject, session) for session in SESSIONS],
                                                     GESTURES[:1], REST_TRIALS[fold:fold + 1])),
                           shuffle=False,
                           feature_name=feature_name,
                           window=window)
                logger.debug('Test set loaded')
                return train, val

class Preload(mx.io.PrefetchingIter):

    def __getattr__(self, name):
        if name != 'iters' and hasattr(self, 'iters') and hasattr(self.iters[0], name):
            return getattr(self.iters[0], name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ('shuffle', 'downsample', 'last_batch_handle'):
            return setattr(self.iters[0], name, value)
        return super(Preload, self).__setattr__(name, value)

    def iter_next(self):
        for e in self.data_ready:
            e.wait()
        if self.next_batch[0] is None:
            #  for i in self.next_batch:
                #  assert i is None, "Number of entry mismatches between iterators"
            return False
        else:
            #  for batch in self.next_batch:
                #  assert batch.pad == self.next_batch[0].pad, "Number of entry mismatches between iterators"
            self.current_batch = mx.io.DataBatch(sum([batch.data for batch in self.next_batch], []),
                                                 sum([batch.label for batch in self.next_batch], []),
                                                 self.next_batch[0].pad,
                                                 self.next_batch[0].index)
            for e in self.data_ready:
                e.clear()
            for e in self.data_taken:
                e.set()
            return True


class CSLDYImageData(mx.io.NDArrayIter):

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
        super(CSLDYImageData, self).__init__(*args, **kargs)

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
                assert (self.batch_size // batch_size) % self.num_subject == 0
                self._index = self._index[:len(self._index) // self.batch_size * self.batch_size].reshape(
                    -1, self.num_subject, batch_size).transpose(0, 2, 1).ravel()

    def reset(self):
        self._reset()
        super(CSLDYImageData, self).reset()

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
            
#        if self.random_shift_horizontal or self.random_shift_vertical or self.random_scale or self.random_bad_channel:
#            data = [(k, a.copy()) for k, a in self.data_orig]
#            if self.random_shift_horizontal or self.random_shift_vertical:
#                logger.info('shift {} {} {}',
#                            self.random_shift_fill,
#                            self.random_shift_horizontal,
#                            self.random_shift_vertical)
#                hss = self.random_state.choice(1 + 2 * self.random_shift_horizontal,
#                                               len(data[0][1])) - self.random_shift_horizontal
#                vss = self.random_state.choice(1 + 2 * self.random_shift_vertical,
#                                               len(data[0][1])) - self.random_shift_vertical
#                #  data = [(k, np.array([np.roll(row, s, axis=1) for row, s in izip(a, shift)]))
#                        #  for k, a in data]
#                data = [(k, np.array([_shift(row, hs, vs, self.random_shift_fill)
#                                      for row, hs, vs in izip(a, hss, vss)]))
#                        for k, a in data]
#            if self.random_scale:
#                s = self.random_scale
#                ss = s / 4
#                data = [
#                    (k, a * 2 ** (self._rand(-s, s, (a.shape[0], 1, 1, 1)) + self._rand(-ss, ss, a.shape)))
#                    for k, a in data
#                ]
#            if self.random_bad_channel:
#                mask = self.random_state.choice(2, len(data[0][1])) > 0
#                if mask.sum():
#                    ch = self.random_state.choice(np.prod(data[0][1].shape[2:]), mask.sum())
#                    row = ch // data[0][1].shape[3]
#                    col = ch % data[0][1].shape[3]
#                    val = self.random_state.choice(self.random_bad_channel, mask.sum())
#                    val = np.tile(val.reshape(-1, 1), (1, data[0][1].shape[1]))
#                    for k, a in data:
#                        a[mask, :, row, col] = val
#            self.data = data    

        self.num_data = len(self._index)


# -----------------------------------------

class CSLImageData(mx.io.NDArrayIter):

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
        super(CSLImageData, self).__init__(*args, **kargs)

#        self.num_channel = self.window
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
         res = [(k, tuple([self.batch_size, self.num_channel] + list([self.window, self.num_semg_col*self.num_semg_row]))) for k, v in self.data]
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
                
                
        if data_source is self.data:       
             new_res = []           
             for a in res:
                 new_res.append(a.asnumpy() if isinstance(a, mx.nd.NDArray) else a)
             res = new_res             
             
#             res = [a.reshape(a.shape[0], a.shape[1], -1) for a in res]
#             res = [self._get_sigimg(a.asnumpy() if isinstance(a, mx.nd.NDArray) else a) for a in res]
             res = [a.reshape(a.shape[0], 1, a.shape[1], -1) for a in res]

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

            for i in range(0, len(self._subject), batch_size):
                # Make sure that the samples in one batch are from the same subject
                assert np.all(self._subject[self._index[i:i + batch_size - 1]] ==
                              self._subject[self._index[i + 1:i + batch_size]])

            if batch_size != self.batch_size:
                assert self.batch_size % batch_size == 0
                assert (self.batch_size // batch_size) % self.num_subject == 0
                self._index = self._index[:len(self._index) // self.batch_size * self.batch_size].reshape(
                    -1, self.num_subject, batch_size).transpose(0, 2, 1).ravel()

    def reset(self):
        self._reset()
        super(CSLImageData, self).reset()

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


def _shift(a, hs, vs, fill):
    if fill == 'zero':
        b = np.zeros(a.shape, dtype=a.dtype)
    elif fill == 'margin':
        b = np.empty(a.shape, dtype=a.dtype)
    else:
        assert False, 'Known fill type: {}'.format(fill)

    s = a.shape
    if hs < 0:
        shb, she = -hs, s[2]
        thb, the = 0, s[2] + hs
    else:
        shb, she = 0, s[2] - hs
        thb, the = hs, s[2]
    if vs < 0:
        svb, sve = -vs, s[1]
        tvb, tve = 0, s[1] + vs
    else:
        svb, sve = 0, s[1] - vs
        tvb, tve = vs, s[1]
    b[:, tvb:tve, thb:the] = a[:, svb:sve, shb:she]

    if fill == 'margin':
        #  Corners
        b[:, :tvb, :thb] = b[:, tvb, thb]
        b[:, tve:, :thb] = b[:, tve - 1, thb]
        b[:, tve:, the:] = b[:, tve - 1, the - 1]
        b[:, :tvb, the:] = b[:, tvb, the - 1]
        #  Borders
        b[:, :tvb, thb:the] = b[:, tvb:tvb + 1, thb:the]
        b[:, tvb:tve, :thb] = b[:, tvb:tve, thb:thb + 1]
        b[:, tve:, thb:the] = b[:, tve - 1:tve, thb:the]
        b[:, tvb:tve, the:] = b[:, tvb:tve, the - 1:the]

    return b

# -----------------------------------------
class GetTrial(object):

    def __init__(self, preprocess=None):
        self.preprocess = preprocess
        self.memo = {}

    def __call__(self, root, combo):
        subject, session = decode_subject_and_session(combo.subject)
        path = os.path.join(root,
                            'subject%d' % subject,
                            'session%d' % session,
                            'gest%d.mat' % combo.gesture)
        if path not in self.memo:
            data = _get_data(path, self.preprocess)
            self.memo[path] = data
            logger.debug('{}', path)
        else:
            data = self.memo[path]
        assert combo.trial < len(data), str(combo)
        data = data[combo.trial].copy()
        gesture = np.repeat(combo.gesture, len(data))
        subject = np.repeat(combo.subject, len(data))
        return Trial(data=data, gesture=gesture, subject=subject)


@cached
def _get_data(path, preprocess):
    data = sio.loadmat(path)['gestures']
    data = [np.transpose(np.delete(segment.astype(np.float32), np.s_[7:192:8], 0))
            for segment in data.flat]
    if preprocess:
        data = list(Context.parallel(jb.delayed(preprocess)(segment, **PREPROCESS_KARGS)
                                     for segment in data))
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


def decode_subject_and_session(ss):
    return (ss - 1) // NUM_SESSION + 1, (ss - 1) % NUM_SESSION + 1


def encode_subject_and_session(subject, session):
    return (subject - 1) * NUM_SESSION + session


def get_bandstop(data):
    from ..utils import butter_bandstop_filter
    return np.array([butter_bandstop_filter(ch, 45, 55, 2048, 2) for ch in data])


def get_combos(*args):
    for arg in args:
        if isinstance(arg, tuple):
            arg = [arg]
        for a in arg:
            combo = Combo(*a)
            if ignore_missing(combo):
                continue
            yield combo


def ignore_missing(combo):
    return combo.subject == 19 and combo.gesture in (8, 9) and combo.trial == 9


def get_index(a):
    '''Convert label to 0 based index'''
    b = list(set(a))
    return np.array([x if x < 0 else b.index(x) for x in a.ravel()]).reshape(a.shape)


