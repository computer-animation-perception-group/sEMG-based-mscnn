import os
import scipy.io as sio
import numpy as np
import cv2
from itertools import product
from collections import namedtuple


#import cv2
#from ..utils import butter_lowpass_filter as lowpass


subjects = list(range(27))
gestures = list(range(53))
trials = list(range(10))
#input_path = '/home/weiwentao/public/duyu/misc/ninapro-db1'
#output_path = '/home/weiwentao/public/semg/ninapro-feature/ninapro-db1-var-raw'


input_path = 'Y:/duyu/misc/ninapro-db1'


filtering_type = 'none'
framerate = 100

window_length_ms = 150
window_stride_ms = 10

window = window_length_ms*framerate/1000
stride = window_stride_ms*framerate/1000

output_path = ('Y:/semg/ninapro-feature/TEST-ninapro-db1-var-raw-prepro-%s-win-%d-stride-%d' % (filtering_type, window, stride))

Combo = namedtuple('Combo', ['subject', 'gesture', 'trial'], verbose=False)

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
    return np.abs(spectrum.T)

def lucas_kanade_np(im1, im2, win=2):
    assert im1.shape == im2.shape
    I_x = np.zeros(im1.shape)
    I_y = np.zeros(im1.shape)
    I_t = np.zeros(im1.shape)
    I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
    params = np.zeros(im1.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
    params[..., 0] = I_x * I_x # I_x2
    params[..., 1] = I_y * I_y # I_y2
    params[..., 2] = I_x * I_y # I_xy
    params[..., 3] = I_x * I_t # I_xt
    params[..., 4] = I_y * I_t # I_yt
    del I_x, I_y, I_t
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    del params
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])
    del cum_params
    op_flow = np.zeros(im1.shape + (2,))
    det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2
    op_flow_x = np.where(det != 0,
                         (win_params[..., 1] * win_params[..., 3] -
                          win_params[..., 2] * win_params[..., 4]) / det,
                         0)
    op_flow_y = np.where(det != 0,
                         (win_params[..., 0] * win_params[..., 4] -
                          win_params[..., 2] * win_params[..., 3]) / det,
                         0)
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
    return op_flow

if __name__ == '__main__':

    print ("NinaPro activity image generation, use window = %d frames, stride = %d frames" % (window, stride)) 

    combos = get_combos(product(subjects, gestures, trials))
           
    combos = list(combos)      

    for combo in combos:
        in_path = os.path.join(
                input_path, 'data',
                '{c.subject:03d}',
                '{c.gesture:03d}',
                '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}.mat').format(c=combo)
                
        out_dir = os.path.join(
                output_path,
                '{c.subject:03d}',
                '{c.gesture:03d}').format(c=combo)  
                
        if os.path.isdir(out_dir) is False:
             os.makedirs(out_dir)                 
     
                
        data = sio.loadmat(in_path)['data'].astype(np.float32)
        
        print ("Subject %d Gesture %d Trial %d data loaded!" % (combo.subject, combo.gesture, combo.trial))
        
        if filtering_type is 'lowpass':            
#             data = np.transpose([lowpass(ch, 1, framerate, 1, zero_phase=True) for ch in data.T])
             data = np.transpose([butter_lowpass_filter(ch, 1, framerate, 1, zero_phase=True) for ch in data.T])   
             print ("Subject %d Gesture %d Trial %d bandpass filtering finished!" % (combo.subject, combo.gesture, combo.trial))
        else:
             pass
           
        
        data = data.T
        chnum = data.shape[0]    
        
        first_frame = data[:,30]
        second_frame = data[:,50]
     
        first_frame = np.reshape(first_frame, (10,1))
        second_frame = np.reshape(second_frame, (10,1))
        
#        first_frame = cv2.cv.fromarray(first_frame) 
#        second_frame = cv2.cv.fromarray(second_frame)
        
        flow = lucas_kanade_np(first_frame, second_frame)
        
        print flow
            