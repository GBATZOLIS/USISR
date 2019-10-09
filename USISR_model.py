# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:46:47 2019

@author: Giorgos
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:03:47 2019

@author: Georgios
"""

#Unsupervised single image super-resolution

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
from evaluator import evaluator
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, GlobalAveragePooling2D,Flatten, BatchNormalization, LeakyReLU, Lambda, DepthwiseConv2D
from keras.activations import relu,tanh,sigmoid
from keras.initializers import glorot_normal
from keras.models import Model
from preprocessing import gauss_kernel, rgb2gray, NormalizeData

from loss_functions import  total_variation, binary_crossentropy, vgg_loss
from keras.applications.vgg19 import VGG19
import cv2 as cv
from model_architectures import *



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
        patch = int(np.ceil(self.target_res[0] / 2**4))
        self.disc_patch = (patch, patch, 1)
        print(self.disc_patch)
        
        # Configure data loader
        #self.main_path = "C:\\Users\\Georgios\\Desktop\\4year project\\wespeDATA"
        #self.dataset_name = "cycleGANtrial"
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols), SRscale=SRscale)
        
        
        #LOGGER settings
        self.log_TrainingPoints=[]
        self.log_D_loss=[]
        self.log_G_loss=[]
        self.log_ReconstructionLoss=[]
        self.log_ID_loss=[]
        self.log_TotalVariation=[]
        
        self.log_sample_ssim_time_point=[]
        self.log_sample_ssim=[]
        
        #configure perceptual loss 
        #self.content_layer = 'block1_conv1'
        
        #set the blurring settings
        #self.kernel_size=21
        #self.std = 3
        #self.blur_kernel_weights = gauss_kernel(self.kernel_size, self.std, self.channels)
        #self.texture_weights = np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.2989, 0.5870, 0.1140]), axis=0), axis=0), axis=-1)
        #print(self.texture_weights.shape)
        
        #set the optimiser
        optimizer = Adam(0.0001, 0.5)
        
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
        
        #instantiate the VGG models
        self.vgg_model_LR = VGG19(weights='imagenet', include_top=False, input_shape = self.img_shape)
        self.block2_conv1_LR = Model(inputs=self.vgg_model_LR.input, outputs=self.vgg_model_LR.get_layer("block1_conv1").output)
        self.block2_conv1_LR.trainable=False
    
        self.vgg_model_SR = VGG19(weights='imagenet', include_top=False, input_shape = self.target_res)
        self.block2_conv1_SR = Model(inputs=self.vgg_model_SR.input, outputs=self.vgg_model_SR.get_layer("block1_conv1").output)
        self.block2_conv1_SR.trainable=False
        
        

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.target_res)
        #downscaled_img_B = Input(shape=self.target_res)
        downscaled_img_B = AveragePooling2D(pool_size=(2, 2))(img_B)
        
        # Translate images to the other domain
        fake_B = self.G(img_A)
        identity_B = self.G(downscaled_img_B)
        identity_B_vgg = self.block2_conv1_SR(identity_B)
        
        fake_A = self.F(img_B)
        
        # Translate images back to original domain
        reconstr_A = self.F(fake_B)
        reconstr_A_vgg = self.block2_conv1_LR(reconstr_A)
        
        reconstr_B = self.G(fake_A)
        reconstr_B_vgg = self.block2_conv1_SR(reconstr_B)
        
        # For the combined model we will only train the generators
        self.D2.trainable = False

        # Discriminators determines validity of translated images
        valid_B = self.D2(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B] ,
                              outputs=[valid_B, reconstr_A_vgg, reconstr_B_vgg, identity_B_vgg, fake_B])
        
        self.combined.compile(loss=[binary_crossentropy, 'mae', 'mae', 'mae', total_variation],
                            loss_weights=[4, 2, 2, 0.5, 0.8],
                            optimizer=optimizer)
        
        print(self.combined.summary())
        
    def logger(self,):
        fig, axs = plt.subplots(2, 2, figsize=(6,8))
        
        ax = axs[0,0]
        ax.plot(self.log_TrainingPoints, self.log_D_loss, label="D_adv_loss")
        ax.plot(self.log_TrainingPoints, self.log_G_loss, label="G_adv_loss")
        ax.legend()
        ax.set_title("Adversarial losses")
        
        ax = axs[0,1]
        ax.plot(self.log_TrainingPoints, self.log_ReconstructionLoss, label="Reconstruction")
        ax.legend()
        ax.set_title("Reconstruction losses")
        
        ax = axs[1,0]
        ax.plot(self.log_TrainingPoints, self.log_ID_loss)
        ax.set_title("Identity loss")
        
        ax = axs[1,1]
        ax.plot(self.log_TrainingPoints, self.log_TotalVariation)
        ax.set_title("Total Variation loss")
        
        fig.savefig("progress/log.png")
        
        fig, axs = plt.subplots(1,1)
        ax=axs
        ax.plot(self.log_sample_ssim_time_point, self.log_sample_ssim)
        ax.set_title("sample SSIM value")
        fig.savefig("progress/sample_ssim.png")
        
        plt.close("all")
        
    def forward_generator_network(self, name):
        generator_model = SR(scale = self.SRscale, input_shape = self.img_shape, n_feats=128, n_resblocks=8, name = name)
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
        print("valid shape:{}".format(valid.shape))
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        dynamic_evaluator = evaluator(img_res=self.img_shape, SRscale = self.SRscale)
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.G.predict(imgs_A)
                imgs_A_vgg = self.block2_conv1_LR.predict(imgs_A)
                imgs_B_vgg = self.block2_conv1_SR.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                d_loss_real = self.D2.train_on_batch(imgs_B, valid)
                d_loss_fake = self.D2.train_on_batch(fake_B, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                self.log_D_loss.append(d_loss[0])

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A_vgg, imgs_B_vgg, imgs_B_vgg, imgs_B])
                elapsed_time = datetime.datetime.now() - start_time
                
                training_time_point = epoch+batch_i/self.data_loader.n_batches
                self.log_TrainingPoints.append(np.around(training_time_point,3))
                self.log_G_loss.append(g_loss[1])
                self.log_ReconstructionLoss.append(np.mean(g_loss[2:4]))
                self.log_ID_loss.append(g_loss[4])
                self.log_TotalVariation.append(g_loss[5])

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, ID: %05f, TV: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            g_loss[1],
                                                                            np.mean(g_loss[2:4]),
                                                                            g_loss[4],
                                                                            g_loss[5],
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    print("Epoch: {} --- Batch: {} ---- saved".format(epoch, batch_i))
                    dynamic_evaluator.model = self.G
                    dynamic_evaluator.epoch = epoch
                    dynamic_evaluator.batch = batch_i
                    dynamic_evaluator.perceptual_test(5)
                    
                    sample_mean_ssim = dynamic_evaluator.objective_test(batch_size=800)
                    print("Sample mean SSIM: -------------------  %05f   -------------------" % (sample_mean_ssim))
                    self.log_sample_ssim_time_point.append(np.around(training_time_point,3))
                    self.log_sample_ssim.append(sample_mean_ssim)
                    #self.sample_images(epoch, batch_i)
                    self.G.save("models/{}_{}.h5".format(epoch, batch_i))
                    self.logger()
        


patch_size=(50,50)
epochs=5
batch_size=4
sample_interval = 100 #after sample_interval batches save the model and generate sample images

gan = CINGAN(patch_size=patch_size)
gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)