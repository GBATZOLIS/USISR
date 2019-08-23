# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:55:26 2019

@author: Georgios
"""

import keras.backend as K
from keras.layers import *
from keras.activations import *
from keras.initializers import glorot_normal
from keras.models import Model
import numpy as np
from preprocessing import rgb2gray


def resblock(feature_in, num):
    # subblock (conv. + BN + relu)
    temp =  Conv2D(64, (3, 3), strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_1' %num), kernel_initializer = glorot_normal())(feature_in)
    temp = BatchNormalization(name = ('resblock_%d_BN_1' %num))(temp)
    temp = Activation('relu')(temp)
        
    # subblock (conv. + BN + relu)
    temp =  Conv2D(64, (3, 3), strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_2' %num), kernel_initializer = glorot_normal())(temp)
    temp = BatchNormalization(name = ('resblock_%d_BN_2' %num))(temp)
    temp = Activation('relu')(temp)
    
    return Add()([temp, feature_in])


def generator_network(image_shape, name):
    
    image = Input(image_shape)
    b1_in = Conv2D(64, (9,9), strides = 1, padding = 'SAME', name = 'CONV_1', activation = 'relu', kernel_initializer = glorot_normal())(image)
    #b1_in = relu()(b1_in)
    # residual blocks
    b1_out = resblock(b1_in, 1)
    b2_out = resblock(b1_out, 2)
    b3_out = resblock(b2_out, 3)
    b4_out = resblock(b3_out, 4)
    
    # conv. layers after residual blocks
    temp = Conv2D(64, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_2', kernel_initializer=glorot_normal())(b4_out)
    temp = Activation('relu')(temp)
    
    temp = Conv2D(64, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer=glorot_normal())(b4_out)
    temp = Activation('relu')(temp)
    
    temp = Conv2D(64, (3,3) , strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer=glorot_normal())(b4_out)
    temp = Activation('relu')(temp)
    
    temp = Conv2D(3, (1,1) , strides = 1, padding = 'SAME', name = 'CONV_5', kernel_initializer=glorot_normal())(b4_out)
    
    return Model(inputs=image, outputs=temp, name=name)

def discriminator_network(name, preprocess = 'gray'):
        
        image = Input((128,128,3))
        
        if preprocess == 'gray':
            #convert to grayscale image
            print("Discriminator-texture")
            
            #output_shape=(image.shape[0], image.shape[1], 1)
            image_processed=Lambda(rgb2gray)(image)
            print(type(image_processed))
            #print(image_processed.shape)
            #image_processed = rgb_to_grayscale(image)
            
        elif preprocess == 'blur':
            print("Discriminator-color (blur)")
            
            g_layer = DepthwiseConv2D(self.kernel_size, use_bias=False, padding='same')
            image_processed = g_layer(image)
            
            g_layer.set_weights([self.blur_kernel_weights])
            g_layer.trainable = False

            
        else:
            print("Discriminator-color (none)")
            image_processed = image
            
        # conv layer 1 
        temp = Conv2D(48, (11,11), strides = 4, padding = 'SAME', name = 'CONV_1', kernel_initializer = glorot_normal())(image_processed)
        print(type(temp))
        temp = LeakyReLU(alpha=0.3)(temp)
        print(type(temp))
        
        # conv layer 2
        temp = Conv2D(128, (5,5), strides = 2, padding = 'SAME', name = 'CONV_2', kernel_initializer = glorot_normal())(temp)
        print(type(temp))
        temp = LeakyReLU(alpha=0.3)(temp)
        print(type(temp))
        temp = BatchNormalization(name = "BN_1")(temp)
        print(type(temp))
        
        # conv layer 3
        temp = Conv2D(192, (3,3), strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer = glorot_normal())(temp)
        print(type(temp))
        temp = LeakyReLU(alpha=0.3)(temp)
        print(type(temp))
        temp = BatchNormalization(name = "BN_2")(temp)
        print(type(temp))
        
        # conv layer 4
        temp = Conv2D(192, (3,3), strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer = glorot_normal())(temp)
        temp = LeakyReLU(alpha=0.3)(temp)
        temp = BatchNormalization(name = "BN_3")(temp)
        
        
        # conv layer 5
        temp = Conv2D(128, (3,3), strides = 2, padding = 'SAME', name = 'CONV_5', kernel_initializer = glorot_normal())(temp)
        temp = LeakyReLU(alpha=0.3)(temp)
        temp = BatchNormalization(name = "BN_4")(temp)
        temp_shape = (np.prod(K.int_shape(temp)[1:]),)
       
        
        
        # FC layer 1
        fc_in = Lambda(lambda v: K.batch_flatten(v), output_shape=temp_shape)(temp)
        print(fc_in.shape)
        #fc_in = Flatten()(temp)
        
        #fc_in = Lambda(lambda v: K.batch_flatten(v))(temp)
        #print(type(fc_in))
        #print(fc_in.shape)
        #fc_in=temp
        
        fc_out = Dense(1024, activation="relu")(fc_in)
        #fc_out = LeakyReLU(alpha=0.3)(fc_out)
        
        # FC layer 2
        logits = Dense(1)(fc_out)
        probability = sigmoid(logits)
        
        return Model(inputs=image, outputs=probability, name=name)
    
    
