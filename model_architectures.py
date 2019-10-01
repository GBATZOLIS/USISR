# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:13:21 2019

@author: Georgios
"""

#this file defines the actual models to be used in training

from bucket_architectures import EDSR
from keras.layers import *
from keras.activations import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

def SR(scale, input_shape, n_feats, n_resblocks, name):
    model = EDSR(scale = scale, input_shape = input_shape, 
                  n_feats = n_feats, n_resblocks = n_resblocks, name=name)
    return model

def G3(input_shape, name):
    
    def block(input_tensor):
        
        x = Conv2D(64, 3, strides=1, padding='same')(input_tensor)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(64, 3, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        output_tensor = Add()([input_tensor, x])
        
        return output_tensor
    
    input_tensor=Input(input_shape)
    
    x = Conv2D(64, 7, strides=1, padding='same', data_format="channels_last")(input_tensor)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(64, 4, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    #repeated residual blocks
    
    for i in range(6):
        x = block(x)
        
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(3, 7, strides=1, padding='same')(x)
    
    model = Model(inputs = input_tensor, outputs = x, name = name)
    
    return model

def D2(input_shape, name):
    
    input_tensor = Input(input_shape)
    
    x = Conv2D(64, 4, strides=2, padding='same', data_format="channels_last")(input_tensor)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    #x = BatchNormalization()(x)
    
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    #x = BatchNormalization()(x)
    
    x = Conv2D(512, 4, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    #x = BatchNormalization()(x)
    
    x = Conv2D(1, 4, strides=1, padding='same')(x)
    
    model = Model(inputs = input_tensor, outputs = x, name = name)
    
    return model
    
    

#model = G3(input_shape = (64,64,3), name="G3")
#model.compile(loss="mse", optimizer=Adam(0.001))
#model.summary()
    
    
    
    
    