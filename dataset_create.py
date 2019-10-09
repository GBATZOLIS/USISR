# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:29:31 2019

@author: Test-PC
"""

import numpy as np
#this file creates the dataset

#inputs: pathA, pathB

def get_random_patch(img, patch_dimension):
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

def createSquareBoxes(rect_box, size):
        m,n,channels=rect_box.shape
        
        stepsX=int(m/size)
        stepsY=int(n/size)    
        
        cropped_boxes=[]
        for i in range(stepsX):
            for j in range(stepsY):
                crop_box=rect_box[size*i:size*(i+1), size*j:size*(j+1), :]
                cropped_boxes.append(crop_box)
        
        
        if m-stepsX*size>0:
            for j in range(stepsY):
                crop_box=rect_box[-size:, size*j:size*(j+1), :]
                cropped_boxes.append(crop_box)
        
        if n-stepsY*size>0:
            for i in range(stepsX):
                crop_box=rect_box[size*i:size*(i+1), -size:, :]
                cropped_boxes.append(crop_box)
        
        return cropped_boxes
    

import matplotlib.pyplot as plt
from glob import glob
from os.path import basename
import os
import cv2 as cv
from PIL import Image
                
def create_dataset(patch_size, pathA, pathB):
    path_A = glob(pathA+"/*")
    path_B = glob(pathB+"/*")
    
    if not os.path.exists("data/patchesA"):
        os.makedirs("data/patchesA")
        
    if not os.path.exists("data/patchesB"):
        os.makedirs("data/patchesB")
    
    if not os.path.exists("data/downscaled_patchesB"):
        os.makedirs("data/downscaled_patchesB")
    """   
    for path in path_A:
        name = basename(path).split(".")[0]
        image = plt.imread(path).astype(np.uint8)
        patches = createSquareBoxes(image, patch_size)
        
        i=1
        for patch in patches:
            filesave="data/patchesA/"+name+"_"+str(i)+".jpg"
            patch = Image.fromarray(patch)
            patch.save(filesave)
            i+=1
    """
    
    for path in path_B:
        name = basename(path).split(".")[0]
        image = plt.imread(path).astype(np.float32)/255
        patches = createSquareBoxes(image, 2*patch_size)
        i=1
        for patch in patches:
            filesave="data/patchesB/"+name+"_"+str(i)+".png"
            plt.imsave(filesave, patch)
            
            downscaled_patch = cv.resize(patch, (patch_size, patch_size), interpolation = cv.INTER_LINEAR)
            filesave="data/downscaled_patchesB/"+name+"_"+str(i)+".png"
            downscaled_patch=np.clip(downscaled_patch, 0, 1)
            plt.imsave(filesave, downscaled_patch)
            i+=1

create_dataset(50, "data/trainA", "data/trainB")
