# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:36:47 2019

@author: Georgios
"""

import keras.backend as K
#from tensorflow.image import rgb_to_grayscale
import numpy as np
import scipy.stats as st

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def gauss_kernel(kernlen=21, nsig=3, channels=3):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def converter(x):

    weights = K.constant([[[[0.21 , 0.72 , 0.07]]]])
    return K.sum(x*weights, axis=-1,keepdims=True)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
