
import keras.backend as K
from keras.applications.vgg19 import VGG19
import tensorflow as tf

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


def total_variation(y_true, y_pred):
    
    x=y_pred
    assert K.ndim(x) == 4
    
    img_nrows=x.shape[1]
    img_ncols=x.shape[2]
    
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    
    return K.sum(K.pow(a + b, 1.25))