# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:24:53 2019

@author: Georgios
"""
  
import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.misc import imresize
import keras.backend as K
#from skimage.transform import downscale_local_mean
#This file contains the DataLoader class

class DataLoader():
    def __init__(self, img_res=(128, 128), SRscale=2):
        #self.dataset_name = dataset_name
        self.SRscale=SRscale
        self.img_res = img_res
        self.target_res = (SRscale*self.img_res[0], SRscale*self.img_res[1], 3)
        #self.main_path = main_path
        
    
    #def resize(self, x):
    #    return K.resize_images(x, 1/self.SRscale, 1/self.SRscale, data_format = "channels_last", interpolation = "bilinear")
    
    def get_random_patch(self, img, patch_dimension):
        if img.shape[0]==patch_dimension[0] and img.shape[1]==patch_dimension[1]:
            return img
        
        else:
            image_shape=img.shape
            image_length = img.shape[0]
            image_width = img.shape[1]
            patch_length = patch_dimension[0]
            patch_width = patch_dimension[1]
            
            if (image_length >= patch_length) and (image_width >= patch_width):
                x_max=image_shape[0]-patch_dimension[0]
                y_max=image_shape[1]-patch_dimension[1]
                x_index=np.random.randint(x_max)
                y_index=np.random.randint(y_max)
            else:
                print("Error. Not valid patch dimensions")
            
            return img[x_index:x_index+patch_dimension[0], y_index:y_index+patch_dimension[1], :]
        
    def load_data(self, domain, patch_dimension=None, batch_size=1, is_testing=False):
        data_type = r"train%s" % domain if not is_testing else "test%s" % domain
        path = glob(r'data/%s/*' % (data_type))
        batch_images = np.random.choice(path, size=batch_size)
        
        if patch_dimension==None and is_testing==True:
            #if the patch dimension is not specified, use the training dimensions
            patch_dimension = self.img_res
            
        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            img = self.get_random_patch(img, patch_dimension)
            
            #img=zoom(img, zoom = (self.SRscale, self.SRscale, 1), order=1) #new addition
            
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob(r'data/%sA/*' % (data_type))
        path_B = glob(r'data/%sB/*' % (data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B, downscaled_imgs_B = [], [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)
                #img_A = scipy.misc.imresize(img_A, self.img_res)
                #img_B = scipy.misc.imresize(img_B, self.img_res)
                if (img_A.shape[0]>self.img_res[0]) or (img_A.shape[1]>self.img_res[1]):
                    img_A=self.get_random_patch(img_A, patch_dimension = self.img_res)
                    
                    #img_A=zoom(img_A, zoom = (self.SRscale, self.SRscale, 1), order=1)#new addition
                    
                    img_B=self.get_random_patch(img_B, patch_dimension = self.target_res)
                
                downscaled_img_B = imresize(arr = img_B, size = 0.5)
                downscaled_imgs_B.append(downscaled_img_B)
                #downscaled_img_B = downscale_local_mean(img_B, (self.SRscale, self.SRscale))
                
                    
                #if not is_testing and np.random.random() > 0.5:
                #        img_A = np.fliplr(img_A)
                #        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                #downscaled_imgs_B.append(imgs_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            downscaled_imgs_B = np.array(downscaled_imgs_B)/127.5 - 1.
            print(imgs_A.shape)
            print(imgs_B.shape)
            print(downscaled_imgs_B.shape)
            #downscaled_img_B = self.resize(K.variable(imgs_B))

            yield imgs_A, imgs_B, downscaled_imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return plt.imread(path).astype(np.float)

