# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:03:47 2019

@author: Georgios
"""

#Unsupervised single image super-resolution

from __future__ import print_function, division
import scipy

import keras.backend as K
from keras.datasets import mnist
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, GlobalAveragePooling2D,Flatten, BatchNormalization, LeakyReLU, Lambda, DepthwiseConv2D
from keras.activations import relu,tanh,sigmoid
from keras.initializers import glorot_normal
from keras.models import Model
from preprocessing import gauss_kernel, rgb2gray, NormalizeData

from loss_functions import  total_variation, binary_crossentropy
from keras.applications.vgg19 import VGG19
from scipy.misc import imresize
from model_architectures import *
import cv2 as cv

class CINGAN():
    def __init__(self, patch_size=(100,100), SRscale=2):
        # Input shape
        self.SRscale=SRscale
        self.img_rows = patch_size[0]
        self.img_cols = patch_size[1]
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.target_res = (self.SRscale*self.img_shape[0], self.SRscale*self.img_shape[1], self.channels)
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.target_res[0] / 2**3)
        self.disc_patch = (patch, patch, 1)
        #print(self.disc_patch)
        
        # Configure data loader
        #self.main_path = "C:\\Users\\Georgios\\Desktop\\4year project\\wespeDATA"
        #self.dataset_name = "cycleGANtrial"
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols), SRscale=SRscale)
        
        #configure perceptual loss 
        #self.content_layer = 'block1_conv1'
        
        #set the blurring settings
        #self.kernel_size=21
        #self.std = 3
        #self.blur_kernel_weights = gauss_kernel(self.kernel_size, self.std, self.channels)
        #self.texture_weights = np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.2989, 0.5870, 0.1140]), axis=0), axis=0), axis=-1)
        #print(self.texture_weights.shape)
        
        #set the optimiser
        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the discriminators
        
        self.D2 = self.discriminator_network(name="Color_Discriminator")
        
        
        self.D2.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])
        print(self.D2.summary())
        
        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.G = self.forward_generator_network(name = "SR")
        self.F = self.backward_generator_network(name = "G3")

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.target_res)
        downscaled_img_B = Input(shape=self.img_shape)
        #downscaled_img_B = AveragePooling2D(pool_size=(2, 2))(img_B)
        
        # Translate images to the other domain
        fake_B = self.G(img_A)
        #fake_A = self.F(img_B)
        
        #identity
        identity_B = self.G(downscaled_img_B)
        
        # Translate images back to original domain
        reconstr_A = self.F(fake_B)
        #reconstr_B = self.G(fake_A)
        
        # For the combined model we will only train the generators
        self.D2.trainable = False

        # Discriminators determines validity of translated images
        valid_B = self.D2(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B, downscaled_img_B] ,
                              outputs=[valid_B, reconstr_A, identity_B, fake_B])
        
        self.combined.compile(loss=[binary_crossentropy, self.vgg_loss, self.vgg_loss, total_variation],
                            loss_weights=[1, 0.1, 0.1, 0.01],
                            optimizer=optimizer)
        
        print(self.combined.summary())
        
        
        
    
    def vgg_loss(self, y_true, y_pred):
        
        input_tensor = K.concatenate([y_true, y_pred], axis=0)
        model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        layer_features = outputs_dict['block2_conv2']
        y_true_features = layer_features[0, :, :, :]
        y_pred_features = layer_features[1, :, :, :]
        
        return K.mean(K.square(y_true_features - y_pred_features)) 
        
    def forward_generator_network(self, name):
        generator_model = SR(scale = self.SRscale, input_shape = self.img_shape, n_feats=128, n_resblocks=32, name = name)
        return generator_model
        
    
    def backward_generator_network(self,name):
        generator_model = G3(input_shape = self.target_res, name = name)
        return generator_model
    

    def discriminator_network(self, name):
        discriminator_model = D2(input_shape = self.target_res, name = name)
        return discriminator_model

    def train(self, epochs, batch_size=1, sample_interval=50):
        #every sample_interval batches, the model is saved and sample images are generated and saved
        
        
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B, downscaled_img_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.G.predict(imgs_A)
                #fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                d_loss_real = self.D2.train_on_batch(imgs_B, valid)
                d_loss_fake = self.D2.train_on_batch(fake_B, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B, downscaled_img_B], [valid, imgs_A, imgs_B, imgs_B])
                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, ID: %05f, TV: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            g_loss[1],
                                                                            g_loss[2],
                                                                            g_loss[3],
                                                                            g_loss[4],
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    print("Epoch: {} --- Batch: {} ---- saved".format(epoch, batch_i))
                    self.sample_images(epoch, batch_i)
                    self.G.save("models/{}_{}.h5".format(epoch, batch_i))
        
    def sample_images(self, epoch, batch_i):
        r, c = 1, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=10, is_testing=True)
        #print(imgs_A.shape)
        #imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)
        
        
        for i in range(imgs_A.shape[0]):
            
            image=np.expand_dims(imgs_A[i,:,:,:], axis=0)
            # Translate images to the other domain
            fake_B = self.G.predict(image)
            #fake_A = self.g_BA.predict(imgs_B)
            # Translate back to original domain
            reconstr_A = self.F.predict(fake_B)
            
            #reconstr_B = self.g_AB.predict(fake_A)
            
            n_image = NormalizeData(image[0])
            n_image = cv.resize(n_image, (self.target_res[0], self.target_res[1]), interpolation = cv.INTER_CUBIC)
            n_image = np.clip(n_image, 0, 1)
            n_image=np.expand_dims(n_image, axis=0)
            
            n_fake_B = NormalizeData(fake_B)
            
            n_reconstr_A = NormalizeData(reconstr_A[0])
            n_reconstr_A = cv.resize(n_reconstr_A, (self.target_res[0], self.target_res[1]), interpolation = cv.INTER_CUBIC)
            n_reconstr_A = np.clip(n_reconstr_A, 0, 1)
            n_reconstr_A = np.expand_dims(n_reconstr_A, axis=0)
            
            gen_imgs = np.concatenate([n_image, n_fake_B, n_reconstr_A])
    
            # Rescale images 0 - 1
            
            #gen_imgs = 0.5 * gen_imgs + 0.5
    
            titles = ['Original(A)', 'Enhanced(B)', 'Reconstructed(A)']
            fig, axs = plt.subplots(r, c)
            cnt = 0
            
            for ax in axs.flat:
                ax.imshow(gen_imgs[cnt])
                ax.set_title(titles[cnt])
                cnt += 1
            
             
            
            fig.savefig("generated_images/%d_%d_%d.png" % (epoch, batch_i, i))
            
        plt.close()
        

if __name__ == '__main__':
    patch_size=(32,32)
    epochs=20
    batch_size=1
    sample_interval = 100 #after sample_interval batches save the model and generate sample images
    
    gan = CINGAN(patch_size=patch_size)
    gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)