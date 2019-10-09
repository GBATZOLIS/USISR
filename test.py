# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 20:13:03 2019

@author: Georgios
"""
import keras
import keras.backend as K
from data_loader import DataLoader
import numpy as np
from model_architectures import SR
import matplotlib.pyplot as plt
from preprocessing import NormalizeData
import cv2 as cv
#this file is used to test the performance of a saved generator model

main_path = r"C:\\Users\\Giorgos\\Documents\\data\\dped\\iphone\\full_size_test_images"
data_loader = DataLoader(test_data_dir = main_path)

imgs=data_loader.load_data(domain="A", batch_size=10, patch_dimension = (128,128), is_testing=True)
print(imgs.shape)
imgs_tensor = K.variable(imgs)

#load the model
generator = SR(scale = 2, input_shape = imgs[0].shape, n_feats=256, n_resblocks=8, name = "Test_Generator")

model_path="C:\\Users\\Giorgos\\Documents\\Github\\USISR saved models\\2\\6_800.h5"
generator.load_weights(model_path)



for i in range(imgs.shape[0]):
    image=np.expand_dims(imgs[i,:,:,:], axis=0)
    fake_B_image = generator.predict(image)
    fake_B_image = NormalizeData(fake_B_image[0])
    fake_B_image=np.expand_dims(fake_B_image, axis=0)
    print(fake_B_image.shape)
    #plt.figure()
    original = NormalizeData(imgs[i,:,:,:])
    
    original = cv.resize(src = original, dsize=(2*imgs[0].shape[0], 2*imgs[0].shape[1]), interpolation=cv.INTER_CUBIC)
    original = np.clip(original, 0, 1)
    original=np.expand_dims(original, axis=0)
    print(original.shape)
    #original = np.expand_dims(original, axis=0)
    
    
    both_images=np.concatenate([original, fake_B_image], axis=0)
    titles = ['domain RGB', 'domain DSLR']
    fig, axs = plt.subplots(1, 2, figsize=(6,8))
    
    j=0
    for ax in axs.flat:
        ax.imshow(both_images[j])
        ax.set_title(titles[j])
        j += 1
    
    fig.savefig("%s.png" % (i))
