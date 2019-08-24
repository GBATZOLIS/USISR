from __future__ import print_function, division
import scipy

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
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
from architectures import resblock

from loss_functions import  total_variation, binary_crossentropy
from keras.applications.vgg19 import VGG19

class WespeGAN():
    def __init__(self, patch_size=(100,100), SRscale=2):
        # Input shape
        self.img_rows = patch_size[0]
        self.img_cols = patch_size[1]
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.target_res = (SRscale*self.img_shape[0], SRscale*self.img_shape[1], self.channels)
        
        # Configure data loader
        #self.main_path = "C:\\Users\\Georgios\\Desktop\\4year project\\wespeDATA"
        #self.dataset_name = "cycleGANtrial"
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols), SRscale=SRscale)
        
        #configure perceptual loss 
        self.content_layer = 'block1_conv2'
        
        #set the blurring settings
        self.kernel_size=21
        self.std = 3
        self.blur_kernel_weights = gauss_kernel(self.kernel_size, self.std, self.channels)
        self.texture_weights = np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.2989, 0.5870, 0.1140]), axis=0), axis=0), axis=-1)
        print(self.texture_weights.shape)
        
        #set the optimiser
        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the discriminators
        
        self.D_color = self.discriminator_network(name="Color_Discriminator", preprocess = "blur")
        self.D_texture = self.discriminator_network(name="Texture_Discriminator", preprocess = "gray")
        
        
        self.D_color.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])
        
        self.D_texture.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.G = self.generator_network(name = "Forward_Generator_G")
        self.F = self.generator_network(name = "Backward_Generator_F")

        # Input images from both domains
        img_A = Input(shape=self.target_res)
        #img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.G(img_A)
        #fake_A = self.g_BA(img_B)
        
        # Translate images back to original domain
        reconstr_A = self.F(fake_B)
        #reconstr_B = self.g_AB(fake_A)
        

        # For the combined model we will only train the generators
        self.D_color.trainable = False
        self.D_texture.trainable = False

        # Discriminators determines validity of translated images
        valid_A_color = self.D_color(fake_B)
        valid_A_texture = self.D_texture(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=img_A,
                              outputs=[valid_A_color, valid_A_texture, reconstr_A, fake_B])
        
        
        
        self.combined.compile(loss=[binary_crossentropy, binary_crossentropy, self.vgg_loss, total_variation],
                            loss_weights=[20, 3, 0.1, 1/400],
                            optimizer=optimizer)
        
        print(self.combined.summary())
        
        

    def generator_network(self, name):
        
        image=Input(self.target_res)
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


    def discriminator_network(self, name, preprocess = 'gray'):
        
        image = Input(self.target_res)
        
        
        if preprocess == 'gray':
            #convert to grayscale image
            print("Discriminator-texture")
            
            #output_shape=(image.shape[0], image.shape[1], 1)
            gray_layer=Conv2D(1, (1,1), strides = 1, padding = "SAME", use_bias=False, name="Gray_layer")
            image_processed=gray_layer(image)
            gray_layer.set_weights([self.texture_weights])
            gray_layer.trainable = False
            
            #image_processed=Lambda(rgb2gray, output_shape = output_gray_shape)(image)
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
        print(temp.shape)
        
        
        # FC layer 1
        fc_in = Flatten()(temp)
        
        
        fc_out = Dense(1024, activation="relu")(fc_in)
        #fc_out = LeakyReLU(alpha=0.3)(fc_out)
        
        # FC layer 2
        logits = Dense(1)(fc_out)
        #probability = sigmoid(logits)
        
        return Model(inputs=image, outputs=logits, name=name)
    
    
    def vgg_loss(self, y_true, y_pred):
        
        input_tensor = K.concatenate([y_true, y_pred], axis=0)
        model = VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        layer_features = outputs_dict[self.content_layer]
        y_true_features = layer_features[0, :, :, :]
        y_pred_features = layer_features[1, :, :, :]
        
        return K.mean(K.square(y_true_features - y_pred_features)) 

    def train(self, epochs, batch_size=1, sample_interval=50):
        #every sample_interval batches, the model is saved and sample images are generated and saved
        
        
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.G.predict(imgs_A)
                #fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dcolor_loss_real = self.D_color.train_on_batch(imgs_B, valid)
                dcolor_loss_fake = self.D_color.train_on_batch(fake_B, fake)
                dcolor_loss = 0.5 * np.add(dcolor_loss_real, dcolor_loss_fake)

                dtexture_loss_real = self.D_texture.train_on_batch(imgs_B, valid)
                dtexture_loss_fake = self.D_texture.train_on_batch(fake_B, fake)
                dtexture_loss = 0.5 * np.add(dtexture_loss_real, dtexture_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dcolor_loss, dtexture_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch(imgs_A, [valid, valid,
                                                        imgs_A, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, D_color_loss: %f, D_texture_loss: %f acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, TV: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], dcolor_loss[0], dtexture_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3]),
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
            
            n_image = NormalizeData(image)
            n_fake_B = NormalizeData(fake_B)
            n_reconstr_A = NormalizeData(reconstr_A)
            
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
    patch_size=(20,20)
    epochs=20
    batch_size=2
    sample_interval = 50 #after sample_interval batches save the model and generate sample images
    
    gan = WespeGAN(patch_size=patch_size)
    gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)