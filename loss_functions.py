
import keras.backend as K
from keras.applications.vgg19 import VGG19
import tensorflow as tf

#file containing the custom loss functions

def binary_crossentropy(y_true, y_pred):
    #the input tensors are expected to be logits (not passed through softmax)
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                   logits=y_pred)
    
    
# Define custom loss
def vgg_loss(y_true, y_pred):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    
    vggmodel = VGG19(include_top=False)
    f_p = vggmodel(y_pred)  
    f_t = vggmodel(y_true)  
    return K.mean(K.square(f_p - f_t))

#total variation loss used for spatial smoothing (makes sure the reconstructed image does not have very steep gradients)
def total_variation(y_true, y_pred):
    
    images=y_pred
    
    pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
    pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
      
    a = K.square(pixel_dif1)
    b = K.square(pixel_dif2)
    
    sum_axis = [1, 2, 3]
    
    return (K.sum(a, axis = sum_axis)+K.sum(b, axis = sum_axis))