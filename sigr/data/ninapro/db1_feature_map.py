from __future__ import division
import os
from nose.tools import assert_equal
from functools import partial
from logbook import Logger
import mxnet as mx
import numpy as np
import scipy.io as sio
from itertools import product
from collections import OrderedDict
#from lru import LRU
from . import Dataset as Base
from .. import Combo, Trial
from ... import utils, constant


logger = Logger(__name__)

class Dataset(Base):

    name = 'ninapro-db1-features'
   
    img_preprocess = constant.FEATURE_MAP_PREPROCESS
    feature_extraction_winlength = constant.FEATURE_EXTRACTION_WIN_LEN
    feature_extraction_winstride = constant.FEATURE_EXTRACTION_WIN_STRIDE 
    feature_names = constant.FEATURE_NAMES

    root = ('/home/weiwentao/public/semg/ninapro-feature/ninapro-db1-var-raw-prepro-%s-win-%d-stride-%d/' % (img_preprocess, 
                                                                                                             feature_extraction_winlength, 
                                                                                                             feature_extraction_winstride))           
    feature_dim = 0    
    for i in range(len(feature_names)):
        input_dir = os.path.join(root,
                            '001',
                            '001',
                            '001_001_001_{0}.mat'
                           ).format(feature_names[i])                          
        mat = sio.loadmat(input_dir)
        data = mat['data'].astype(np.float32)           
        feature_dim = feature_dim + data.shape[1]
        
        
    
    num_semg_row = 1
    num_semg_col = 10
              
    subjects = list(range(27))
    gestures = list(range(53))
    trials = list(range(10))

    def __init__(self, root):
        self.root = root

#    @classmethod
#    def get_preprocess_kargs(cls):
#        return dict(
#            framerate=cls.framerate,
#            num_semg_row=cls.num_semg_row,
#            num_semg_col=cls.num_semg_col
#        )

    def get_trial_func(self, *args, **kargs):
        return GetTrial(*args, **kargs)

    def get_dataiter(self, get_trial, combos, adabn=False, mean=None, scale=None, **kargs):
        
        print ("Use feature extracted by window %d stride %d, semg row = %d semg col = %d" % (self.feature_extraction_winlength,
                                                                                              self.feature_extraction_winstride,
                                                                                              self.num_semg_row,
                                                                                              self.num_semg_col))             
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
            index.append(np.arange(n, n + len(seg)))
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
        

        
        logger.debug('Make data iter')
           
        return ActivityImageData(
                data=OrderedDict([('data', data)]),
                label=OrderedDict(label),
                gesture=gesture.copy(),
                subject=subject.copy(),
                segment=segment.copy(),
                index=index,
                adabn=adabn,
                num_gesture=gesture.max() + 1,
                num_subject=subject.max() + 1,
                **kargs
        )  


        
        
    def get_one_fold_intra_subject_trials(self):
        return [0, 2, 3, 5, 7, 8, 9], [1, 4, 6]
        
    def get_one_fold_intra_subject_caputo_trials(self):
        return [i - 1 for i in [1, 3, 4, 5, 9]], [i - 1 for i in [2, 6, 7, 8, 10]]
    
    def get_inter_subject_data(self, fold, batch_size, preprocess,
                               adabn, minibatch, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train = load(
            combos=self.get_combos(product([i for i in self.subjects if i != subject],
                                           self.gestures, self.trials)),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject - 1 if minibatch else 1),
            shuffle=True)
        val = load(
            combos=self.get_combos(product([subject], self.gestures, self.trials)),
            shuffle=False)
        return train, val
    
    def get_inter_subject_val(self, fold, batch_size, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        val = load(
            combos=self.get_combos(product([subject], self.gestures, self.trials)),
            shuffle=False)
        return val
    
    def get_intra_subject_data(self, fold, batch_size, preprocess,
                               adabn, minibatch, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
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
            shuffle=True)
        val = load(
            combos=self.get_combos(product([subject], self.gestures, [trial])),
            shuffle=False)
        return train, val
    
    def get_intra_subject_val(self, fold, batch_size, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold // self.num_trial]
        trial = self.trials[fold % self.num_trial]
        val = load(
            combos=self.get_combos(product([subject], self.gestures, [trial])),
            shuffle=False)
        return val
    
    def get_universal_intra_subject_data(self, fold, batch_size, preprocess,
                                         adabn, minibatch, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
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
            shuffle=True)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures, [trial])),
            shuffle=False)
        return train, val
    
    def get_one_fold_intra_subject_val(self, fold, batch_size, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        _, val_trials = self.get_one_fold_intra_subject_trials()
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False)
        return val
        
    def get_one_fold_intra_subject_caputo_val(self, fold, batch_size, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        _, val_trials = self.get_one_fold_intra_subject_caputo_trials()
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False)
        return val    
        
    
    def get_one_fold_intra_subject_data(self, fold, batch_size, preprocess,
                                        adabn, minibatch, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
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
            shuffle=True)
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False)
        print 'Data loading finished!'    
        return train, val
      
    def get_one_fold_intra_subject_caputo_data(self, fold, batch_size, preprocess,
                                        adabn, minibatch, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_caputo_trials()
        train = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True)
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False)
        print 'Data loading finished!'    
        return train, val  
    
    def get_universal_one_fold_intra_subject_data(self, fold, batch_size, preprocess,
                                                  adabn, minibatch, **kargs):
        assert_equal(fold, 0)
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
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
            shuffle=True)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in val_trials])),
            shuffle=False)
        return train, val
        
    def get_universal_one_fold_intra_subject_caputo_data(self, fold, batch_size, preprocess,
                                                          adabn, minibatch, **kargs):
        assert_equal(fold, 0)
        get_trial = self.get_trial_func(self.gestures, self.trials, self.feature_name, preprocess=preprocess)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        train_trials, val_trials = self.get_one_fold_intra_subject_caputo_trials()
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in val_trials])),
            shuffle=False)
        return train, val   

class ActivityImageData(mx.io.NDArrayIter):

    def __init__(self, *args, **kargs):
        print 'Initialization Data Iter!'
        self.random_shift_vertical = kargs.pop('random_shift_vertical', 0)
        self.random_shift_horizontal = kargs.pop('random_shift_horizontal', 0)
        self.random_shift_fill = kargs.pop('random_shift_fill', constant.RANDOM_SHIFT_FILL)
        self.amplitude_weighting = kargs.pop('amplitude_weighting', False)
        self.amplitude_weighting_sort = kargs.pop('amplitude_weighting_sort', False)
        self.downsample = kargs.pop('downsample', None)
        self.shuffle = kargs.pop('shuffle', False)
        self.adabn = kargs.pop('adabn', False)
        self._gesture = kargs.pop('gesture')
        self._subject = kargs.pop('subject')
        self._segment = kargs.pop('segment')
#        self.window = kargs.pop('window')
        self._index_orig = kargs.pop('index')
        self._index = np.copy(self._index_orig)
        self.num_gesture = kargs.pop('num_gesture')
        self.num_subject = kargs.pop('num_subject')
        self.mini_batch_size = kargs.pop('mini_batch_size', kargs.get('batch_size'))
        self.random_state = kargs.pop('random_state', np.random)
        self.balance_gesture = kargs.pop('balance_gesture', 0)
        self.num_channel = 1        

        super(ActivityImageData, self).__init__(*args, **kargs)

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
         return res

#    def _expand_index(self, index):
#        return np.hstack([np.arange(i, i + self.window) for i in index])
#
#    def _reshape_data(self, data):
#        return data.reshape(-1, self.window, *data.shape[2:])
#
##    def _get_fft(self, data):
##        from .. import Context
##        import joblib as jb
##        res = []
##        for amp in Context.parallel(jb.delayed(_get_fft_aux)(sample, self.fft_append) for sample in data):
##            res.append(amp[np.newaxis, ...])
##        return np.concatenate(res, axis=0)
#
#    def _get_segments(self, a, index):
#        b = mx.nd.empty((len(index), self.window) + a.shape[2:], dtype=a.dtype)
#        for i, j in enumerate(index):
#            b[i] = a[j:j + self.window].reshape(self.window, *a.shape[2:])
#        return b

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only"""
        assert(self.cursor < self.num_data), "DataIter needs reset."

#        if data_source is self.data and self.window > 1:
#            if self.cursor + self.batch_size <= self.num_data:
#                #  res = [self._reshape_data(x[1][self._expand_index(self._index[self.cursor:self.cursor+self.batch_size])]) for x in data_source]
#                res = [self._get_segments(x[1], self._index[self.cursor:self.cursor+self.batch_size]) for x in data_source]
#            else:
#                pad = self.batch_size - self.num_data + self.cursor
#                res = [(np.concatenate((self._reshape_data(x[1][self._expand_index(self._index[self.cursor:])]),
#                                        self._reshape_data(x[1][self._expand_index(self._index[:pad])])), axis=0)) for x in data_source]
#        else:
        if self.cursor + self.batch_size <= self.num_data:
            res = [(x[1][self._index[self.cursor:self.cursor+self.batch_size]]) for x in data_source]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            res = [(np.concatenate((x[1][self._index[self.cursor:]], x[1][self._index[:pad]]), axis=0)) for x in data_source]


#        if data_source is self.data and self.fft:
#            if not self.dual_stream:
#                res = [self._get_fft(a.asnumpy() if isinstance(a, mx.nd.NDArray) else a) for a in res]
#            else:
#                res = res + [self._get_fft(a.asnumpy() if isinstance(a, mx.nd.NDArray) else a) for a in res]
#                assert_equal(len(res), 2)

#        if data_source is self.data and self.faug:
#            res += [self.faug * self.random_state.randn(self.batch_size, 16)]

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
        super(ActivityImageData, self).reset()

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

    def __init__(self, gestures, trials, feature_names, preprocess=None):
        self.memo = {}
        self.gesture_and_trials = list(product(gestures, trials))
        self.feature_names = feature_names

    def get_path(self, root, combo):
        return os.path.join(root,
                            '{0.subject:03d}',
                            '{0.gesture:03d}',
                            '{0.subject:03d}_{0.gesture:03d}_{0.trial:03d}'
                           ).format(combo)

    def __call__(self, root, combo):
       

                
       
#        print "call get trial!"
        path = self.get_path(root, combo)
        if path not in self.memo:
            logger.debug('Load subject {}', combo.subject)
            paths = [self.get_path(root, Combo(combo.subject, gesture, trial))
                     for gesture, trial in self.gesture_and_trials]                         
            self.memo.update({path: data for path, data in
                              zip(paths, _get_data(paths, self.feature_names))})
        data, gesture, subject = self.memo[path]
        data = data.copy()
        gesture = gesture.copy()
        subject = subject.copy()    
        return Trial(data=data, gesture=gesture, subject=subject)     


@utils.cached
def _get_data(paths, feature_names):
    return [_get_data_aux(path, feature_names) for path in paths]


def _get_data_aux(path, feature_names):
    
    logger.debug('Load {}', path)
    
    data = []
    for i in range(len(feature_names)):
        input_dir = path+'_'+feature_names[i]+'.mat'
        mat = sio.loadmat(input_dir)
        tmp = mat['data'].astype(np.float32)
        data.append(tmp)
        if i == 0:
           gesture = np.repeat(label_to_gesture(np.asscalar(mat['label'].astype(np.int))), tmp.shape[0])
           subject = np.repeat(np.asscalar(mat['subject'].astype(np.int)), tmp.shape[0])        
    data = np.concatenate(data, axis=1)
    assert(len(gesture)==data.shape[0])
    return data, gesture, subject    
    
    
    
    
    
def label_to_gesture(label):
    '''Convert maxforce to -1'''
    return label if label < 100 else -1


#def get_amplitude_weight(data, segment, framerate):
#    from .. import Context
#    import joblib as jb
#    indices = [np.where(segment == i)[0] for i in set(segment)]
#    w = np.empty(len(segment), dtype=np.float)
#    for i, ret in zip(
#        indices,
#        Context.parallel(jb.delayed(get_amplitude_weight_aux)(data[i], framerate)
#                         for i in indices)
#    ):
#        w[i] = ret
#    return w / max(w.sum(), 1e-8)
#
#
#def get_amplitude_weight_aux(data, framerate):
#    return _get_amplitude_weight_aux(data, framerate)
#
#
#@utils.cached
#def _get_amplitude_weight_aux(data, framerate):
#    # High-Density Electromyography and Motor Skill Learning for Robust Long-Term Control of a 7-DoF Robot Arm
#    lowpass = utils.butter_lowpass_filter
#    shape = data.shape
#    data = np.abs(data.reshape(shape[0], -1))
#    data = np.transpose([lowpass(ch, 3, framerate, 4, zero_phase=True) for ch in data.T])
#    data = data.mean(axis=1)
#    data -= data.min()
#    data /= max(data.max(), 1e-8)
#    return data

def get_index(a):
    '''Convert label to 0 based index'''
    b = list(set(a))
    return np.array([x if x < 0 else b.index(x) for x in a.ravel()]).reshape(a.shape)
