import os
import scipy.io as sio
import numpy as np
from itertools import product
from collections import namedtuple
import cv2
#from ..utils import butter_lowpass_filter as lowpass

dataset = 'csl'

subjects = list(range(27))
gestures = list(range(53))
trials = list(range(10))
#input_path = '/home/weiwentao/public/duyu/misc/ninapro-db1'
#output_path = '/home/weiwentao/public/semg/ninapro-feature/ninapro-db1-var-raw'


input_path = 'Y:/duyu/misc'



if dataset is 'csl':
      subjects = list(range(1, 6))
      sessions = list(range(1, 6))
      gestures = list(range(27))
      trials = list(range(10))
      framerate = 2048
      
     

      Combo = namedtuple('Combo', ['subject', 'session', 'gesture', 'trial'], verbose=False)

#elif dataset is 'dba' or 'dbb' or 'dbc'




filtering_type = 'lowpass'

window_length_ms = 250
window_stride_ms = 10

window = window_length_ms*framerate/1000
stride = window_stride_ms*framerate/1000

output_path = ('Y:/semg/%s-feature/%s-prepro-%s-win-%d-stride-%d' % (dataset, dataset, filtering_type, window, stride))


#from contextlib import contextmanager
#@contextmanager
#def Context(log=None, parallel=False, level=None):
#    from .utils import logging_context
#    with logging_context(log, level=level):
#        if not parallel:
#            yield
#        else:
#            import joblib as jb
#            from multiprocessing import cpu_count
#            with jb.Parallel(n_jobs=cpu_count()) as par:
#                Context.parallel = par
#                yield



def get_combos(*args):
   for arg in args:
       if isinstance(arg, tuple):
           arg = [arg]
       for a in arg:
           yield Combo(*a)
           


#the following functions can be loaded from ..utils


def butter_lowpass_filter(data, cut, fs, order, zero_phase=False):
    from scipy.signal import butter, lfilter, filtfilt

    nyq = 0.5 * fs
    cut = cut / nyq

    b, a = butter(order, cut, btype='low')
    y = (filtfilt if zero_phase else lfilter)(b, a, data)
    return y

    
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data)
    return y    
    

def median_filter(data, num_semg_row, num_semg_col):
    return np.array([median_filter(image, 3).ravel() for image
                         in data.reshape(-1, num_semg_row, num_semg_col)])
                         

def cslcut(data, framerate):
    begin, end = _csl_cut(data, framerate)
    return data[begin:end]

def _csl_cut(data, framerate):
    window = int(np.round(150 * framerate / 2048))
    data = data[:len(data) // window * window].reshape(-1, 150, data.shape[1])
    rms = np.sqrt(np.mean(np.square(data), axis=1))
    rms = [median_filter(image, 3).ravel() for image in rms.reshape(-1, 24, 7)]
    rms = np.mean(rms, axis=1)
    threshold = np.mean(rms)
    mask = rms > threshold
    for i in range(1, len(mask) - 1):
        if not mask[i] and mask[i - 1] and mask[i + 1]:
            mask[i] = True
    from .. import utils
    begin, end = max(utils.continuous_segments(mask),
                     key=lambda s: (mask[s[0]], s[1] - s[0]))
    return begin * window, end * window

def downsample(data, step):
    return data[::step].copy()
                         


def get_segments(data, window, stride):
    return windowed_view(
        data.flat,
        window * data.shape[1],
        (window-stride)* data.shape[1]
    )

def windowed_view(arr, window, overlap):
    from numpy.lib.stride_tricks import as_strided
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)
    

def dft(data):
     f = np.fft.fft2(data)
     fshift = np.fft.fftshift(f)
     magnitude_spectrum = 20*np.log(np.abs(fshift))
     return magnitude_spectrum

#the following functions can be loaded from ..
    
def dft_dy(data):
    data = data.T
    n = data.shape[-1]
    window = np.hanning(n)
    windowed = data * window
    spectrum = np.fft.fft(windowed)
    return np.abs(spectrum)



if __name__ == '__main__':

    print ("%s stacked optical flow generation, use window = %d frames, stride = %d frames" % (dataset, window, stride)) 

    if dataset is 'csl':

        combos = get_combos(product(subjects, sessions, gestures, trials))
               
        combos = list(combos)      
    
        for combo in combos:
            in_path = os.path.join(
                    input_path, dataset,
                    'subject%d' % combo.subject,
                    'session%d' % combo.session,
                    'gest%d.mat' % combo.gesture)
                    
            out_dir = os.path.join(
                    output_path,
                    '{c.subject:03d}',
                    '{c.session:03d}',  
                    '{c.gesture:03d}').format(c=combo)  
                    
            if os.path.isdir(out_dir) is False:
                 os.makedirs(out_dir)                 
         
            data = sio.loadmat(in_path)['gestures']
            data = [np.transpose(np.delete(segment.astype(np.float32), np.s_[7:192:8], 0))
                for segment in data.flat]
            
            
            
            print ("Subject %d Session %d Gesture %d data loaded!" % (combo.subject, combo.session, combo.gesture))
            
            
        
#        if filtering_type is 'lowpass':            
##             data = np.transpose([lowpass(ch, 1, framerate, 1, zero_phase=True) for ch in data.T])
#             data = np.transpose([butter_lowpass_filter(ch, 1, framerate, 1, zero_phase=True) for ch in data.T])   
#             print ("Subject %d Gesture %d Trial %d bandpass filtering finished!" % (combo.subject, combo.gesture, combo.trial))
#        else:
#             pass
#           
#        
#        
#        chnum = data.shape[1];     
#        data = get_segments(data, window, stride)
#        data = data.reshape(-1, window, chnum)
#           
#        data = np.transpose(get_signal_img(data.T))
#
#        out_path = os.path.join(
#                out_dir,
#                '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}_sigimg.mat').format(c=combo)  
#        sio.savemat(out_path, {'data': data, 'label': combo.gesture, 'subject': combo.subject, 'trial':combo.trial})         
#           
#        print ("Subject %d Gesture %d Trial %d sig image saved!" % (combo.subject, combo.gesture, combo.trial))   
# 
###        for test only
##        data = data[0:25,]
##        data = dft(data)        
##        data = cv2.resize(data,None,fx=20,fy=20)
##        cv2.imshow('image',data)
##        cv2.waitKey(0)
##        cv2.destroyAllWindows()     
#
#
#
#        
#
#        data = [dft(seg) for seg in data]       
#        data = np.array(data)
#               
#        
#        print ("Subject %d Gesture %d Trial %d data windowing finished!" % (combo.subject, combo.gesture, combo.trial))
#        
#        out_path = os.path.join(
#                out_dir,
#                '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}_actimg.mat').format(c=combo)  
#        sio.savemat(out_path, {'data': data, 'label': combo.gesture, 'subject': combo.subject, 'trial':combo.trial})   

        
     
            
            