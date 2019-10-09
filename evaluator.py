# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 03:44:35 2019

@author: Test-PC
"""

#evaluator class

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 20:13:03 2019

@author: Georgios
"""
import time
import keras
import keras.backend as K
from data_loader import DataLoader
import numpy as np
#from architectures import generator_network
import matplotlib.pyplot as plt
from preprocessing import NormalizeData
from glob import glob
import os
from skimage.measure import compare_ssim as ssim
#from skvideo.measure import niqe
import cv2 as cv
from model_architectures import SR

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

class evaluator(object):
    
    def __init__(self, img_res=(50, 50), SRscale=2, model=None, model_name=None, epoch=None, batch=None):
        
        self.SRscale = SRscale
        self.data_loader = DataLoader(img_res=img_res, SRscale=SRscale)
        
        self.img_res = img_res
        self.target_res = (SRscale*self.img_res[0], SRscale*self.img_res[1])
        
        self.model_name = model_name
        self.model = model
        self.epoch = epoch
        self.batch = batch
        self.training_points=[] #training time locations where mean SSIM value on test data has been calculated
        self.ssim_vals = [] #calculated SSIM values on test data
    
    def perceptual_test(self, batch_size):
        phone_imgs, dslr_imgs = self.data_loader.load_paired_data(batch_size=batch_size)
        
        fake_dslr_images = self.model.predict(phone_imgs)
        
        i=0
        for phone, fake_dslr, real_dslr in zip(phone_imgs, fake_dslr_images, dslr_imgs):
            #phone = NormalizeData(phone)
            phone = cv.resize(phone, (self.target_res[0], self.target_res[1]), interpolation = cv.INTER_CUBIC)
            np.clip(phone, 0, 1, out=phone)
            phone = np.expand_dims(phone, axis=0)
            #print(phone.shape)
            np.clip(fake_dslr, 0, 1, out=fake_dslr)
            fake_dslr = np.expand_dims(fake_dslr, axis=0)
            #print(fake_dslr.shape)
            #real_dslr = NormalizeData(real_dslr)
            real_dslr = cv.resize(real_dslr, (self.target_res[0], self.target_res[1]), interpolation = cv.INTER_CUBIC)
            np.clip(real_dslr, 0, 1, out=real_dslr)
            real_dslr = np.expand_dims(real_dslr, axis=0)
            #print(real_dslr.shape)
            
            
            all_imgs = np.concatenate([phone, fake_dslr, real_dslr])
            titles = ['source', 'Super-Resolved ', 'Ground Truth']
            fig, axs = plt.subplots(1, 3, figsize=(6,8))
            
            j=0
            for ax in axs.flat:
                ax.imshow(all_imgs[j])
                ax.set_title(titles[j])
                j += 1
            
            if self.model_name:
                fig.savefig("generated_images/test%s.png" % (i))
            else:
                fig.savefig("generated_images/%d_%d_%d.png" % (self.epoch, self.batch, i))
            
            i+=1
        
        plt.close('all')
        print("Perceptual results have been generated")
        
    
    def objective_test(self, batch_size=None, baseline=False):
        phone_imgs, dslr_imgs = self.data_loader.load_paired_data(batch_size=batch_size)
        dslr_imgs = dslr_imgs.astype('float32') #necessary typecasting
        
        if baseline:
            fake_dslr_images=phone_imgs
        else:
            fake_dslr_images = self.model.predict(phone_imgs)
        
        batch_size=phone_imgs.shape[0]
        total_ssim=0
        for i in range(batch_size):
            x = cv.resize(dslr_imgs[i,:,:,:], (self.target_res[0], self.target_res[1]), interpolation = cv.INTER_CUBIC)
            total_ssim+=ssim(fake_dslr_images[i,:,:,:], x, multichannel=True)
        
        mean_ssim = total_ssim/batch_size
        #print("Sample mean SSIM ---------%05f--------- " %(mean_ssim))
        
        #db_ssim = 10*np.log10(mean_ssim)
        return mean_ssim
        
    def enhance(self, img_path, model_name, reference=True):
        phone_image = self.data_loader.load_img(img_path) #load image
        
        phone_image = phone_image[0]
        #phone_image = phone_image[400:1400, 600:1600, :]
        img_shape = phone_image.shape #get dimensions to build the suitable model
        
        
        generator_model = SR(scale = self.SRscale, input_shape = img_shape, n_feats=128, n_resblocks=16, name = "Test Super-resolver")
        
        self.model_name = model_name
        self.model = generator_model
        self.model.load_weights("models/%s" % (model_name))
        
        start=time.time()
        fake_dslr_image = self.model.predict(np.expand_dims(phone_image, axis=0))
        end=time.time()
        print("TIME TAKEN: ", end-start)
        print(np.amax(fake_dslr_image))
        print(np.amin(fake_dslr_image))
        
        if reference:
            fig, axs = plt.subplots(1, 2)
            ax = axs[0]
            bi_cubic_upscaling = cv.resize(phone_image, (img_shape[0]*self.SRscale, img_shape[1]*self.SRscale), interpolation = cv.INTER_CUBIC)
            bi_cubic_upscaling = np.clip(bi_cubic_upscaling, 0, 1)
            ax.imshow(bi_cubic_upscaling)
            ax.set_title("2x Bi-cubic upscaling")
            
            ax = axs[1]
            ax.imshow(np.clip(fake_dslr_image[0], 0, 1))
            ax.set_title("2x SR upscaling")
            plt.show()
            


"""
new_evaluator = evaluator()
img_path = "C:\\Users\\Test-PC\\Desktop\\Github\dped\\iphone\\test_data\\full_size_test_images\\29.png"


new_evaluator.enhance(img_path, model_name = "1_2400.h5")
"""

