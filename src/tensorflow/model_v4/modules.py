import tensorflow_addons.image as tfa_image
import tensorflow as tf
import keras

from keras import layers
from keras import models

from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16


"""
The output activation returns linear values cropped to range from 0 to 1
"""
def output_activation(x):
    return tf.math.minimum(tf.math.maximum(x, 0), 1)

"""
The L1 reconstruction loss function makes the final image colors 
look the same as the colors in the ground-truth image
"""
def l1(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


"""
The L2 reconstruction loss function is similar to L1
"""
def l2(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


"""
The PSNR loss function is responsible for boosting the overall quality of the image by reducing its noise 
(The higher the PSNR the better so we return 1 - PSNR because the loss function tries to minimize it)
"""
def psnr(y_true, y_pred):
    psnr = tf.math.reduce_mean(tf.image.psnr(y_true, y_pred, max_val = 1.0))
    return 1 - psnr / 40.0


"""
The SSIM loss function keeps the result image structure
(The more significant the SSIM the more similar the final image is)
"""
def ssim(y_true, y_pred):
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return 1 - ssim

    
"""
The final loss is linear combination of ssim, psnr, l1, l2 losses
"""
def loss(y_true, y_pred):
    ssim_ = ssim(y_true, y_pred)
    psnr_ = psnr(y_true, y_pred)
    l1_ = l1(y_true, y_pred)
    l2_ = l2(y_true, y_pred)
    
    return ssim_ + psnr_ + 5.0*l1_ + 10.0*l2_


def warp(image: tf.Tensor, flow: tf.Tensor) -> tf.Tensor:
    warped = tf.keras.layers.Lambda(
        lambda x: tfa_image.dense_image_warp(*x))((image, -flow))
    return tf.reshape(warped, shape=tf.shape(image))


class BidirectionalFlowEstimation(layers.Layer):
    def __init__(self, filter_count=[32, 64, 64, 16], filter_size=[(3, 3), (3, 3), (1, 1), (1, 1)], activation='relu', regularizer=None, interpolation='bilinear', **kwargs):
        super(BidirectionalFlowEstimation, self).__init__(**kwargs)
        
        self.flow_add_1_2 = layers.Add()
        self.flow_add_2_1 = layers.Add()
        
        self.flow_upsample_1_2 = layers.UpSampling2D((2, 2), interpolation=interpolation)
        self.flow_upsample_2_1 = layers.UpSampling2D((2, 2), interpolation=interpolation)
        
        self.flow_1_2_concat = layers.Concatenate(axis=3)
        self.flow_2_1_concat = layers.Concatenate(axis=3)
        
        # flow estimation sequentails layers
        self.flow_prediction =  keras.Sequential()
        for fc, fs in zip(filter_count, filter_size):
            self.flow_prediction.add(
                layers.Conv2D(fc, fs, activation=activation, kernel_regularizer=regularizer, padding='same')
            )
        self.flow_prediction.add(layers.Conv2D(2, (1, 1), kernel_regularizer=regularizer, padding='same'))
        
        self.filter_count = filter_count
        self.filter_size = filter_size
        self.activation = activation
        self.regularizer = regularizer
        self.interpolation = interpolation
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filter_count": self.filter_count,
            "filter_size": self.filter_size,
            "activation": self.activation,
            "regularizer": self.regularizer,
            "interpolation": self.interpolation,
        })
        return config

    def call(self, inputs):
        input_1 = inputs[0]
        input_2 = inputs[1]
        flow_1_2 = inputs[2]
        flow_2_1 = inputs[3]

        if tf.is_tensor(flow_1_2) and tf.is_tensor(flow_2_1):
            input_1_warped_1 = BidirectionalFlowEstimation.warp(input_1, flow_1_2)
            input_2_warped_1 = BidirectionalFlowEstimation.warp(input_2, flow_2_1)
        else:
            input_1_warped_1 = input_1
            input_2_warped_1 = input_2
            
        flow_change_1_2_concat = self.flow_1_2_concat([input_2, input_1_warped_1])
        flow_change_1_2 = self.flow_prediction(flow_change_1_2_concat)
        flow_change_2_1_concat = self.flow_2_1_concat([input_1, input_2_warped_1])
        flow_change_2_1 = self.flow_prediction(flow_change_2_1_concat)
        
        if tf.is_tensor(flow_1_2) and tf.is_tensor(flow_2_1):
            flow_1_2_changed = self.flow_add_1_2([flow_1_2, flow_change_1_2])
            flow_2_1_changed = self.flow_add_2_1([flow_2_1, flow_change_2_1])
        else:
            flow_1_2_changed = flow_change_1_2
            flow_2_1_changed = flow_change_2_1
        
        input_1_warped_2 = BidirectionalFlowEstimation.warp(input_1, flow_1_2_changed)
        input_2_warped_2 = BidirectionalFlowEstimation.warp(input_2, flow_2_1_changed)
        flow_1_2_changed_upsampled = self.flow_upsample_1_2(flow_1_2_changed)
        flow_2_1_changed_upsampled = self.flow_upsample_2_1(flow_2_1_changed)
        
        return input_1_warped_2, input_2_warped_2, flow_1_2_changed_upsampled, flow_2_1_changed_upsampled
    
    @staticmethod
    def warp(image: tf.Tensor, flow: tf.Tensor) -> tf.Tensor:
        warped = tf.keras.layers.Lambda(
            lambda x: tfa_image.dense_image_warp(*x))((image, -flow))
        return tf.reshape(warped, shape=tf.shape(image))
    

"""
The function loads the trained FBNet model
"""
def load_model(path):
    return keras.models.load_model(
        path,
        custom_objects = {
            "BidirectionalFlowEstimation": BidirectionalFlowEstimation,
            'loss': loss,
            'l1': l1,
            "ssim": ssim,
            "psnr": psnr,
            "l2": l2,
            "output_activation": output_activation
        }
    )
