import tensorflow_addons as tfa
import tensorflow as tf
import keras

from keras import layers
from keras import models

from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16


VGG_model = VGG16(include_top=False)
VGG_model_1 = models.Model(inputs=VGG_model.inputs, outputs=VGG_model.layers[2].output)
VGG_model_2 = models.Model(inputs=VGG_model.inputs, outputs=VGG_model.layers[5].output)
VGG_model_3 = models.Model(inputs=VGG_model.inputs, outputs=VGG_model.layers[9].output)
VGG_model_4 = models.Model(inputs=VGG_model.inputs, outputs=VGG_model.layers[13].output)
VGG_model_5 = models.Model(inputs=VGG_model.inputs, outputs=VGG_model.layers[17].output)

VGG_model_1.trainable = False
for layer in VGG_model_1.layers:
    layer.trainable = False
    
VGG_model_2.trainable = False
for layer in VGG_model_2.layers:
    layer.trainable = False
    
VGG_model_3.trainable = False
for layer in VGG_model_3.layers:
    layer.trainable = False
    
VGG_model_4.trainable = False
for layer in VGG_model_4.layers:
    layer.trainable = False
        
VGG_model_5.trainable = False
for layer in VGG_model_5.layers:
    layer.trainable = False


"""
The perceptual loss function compares two images that differ very little. 
It cares about the high-level features of the images.
"""
def perceptual(y_true, y_pred):
    vgg_true_1 = VGG_model_1(y_true)
    vgg_pred_1 = VGG_model_1(y_pred)
    
    vgg_true_2 = VGG_model_2(y_true)
    vgg_pred_2 = VGG_model_2(y_pred)
    
    vgg_true_3 = VGG_model_3(y_true)
    vgg_pred_3 = VGG_model_3(y_pred)
    
    vgg_true_4 = VGG_model_4(y_true)
    vgg_pred_4 = VGG_model_4(y_pred)
    
    vgg_true_5 = VGG_model_5(y_true)
    vgg_pred_5 = VGG_model_5(y_pred)

    vgg_1_mse = K.mean(K.square(vgg_true_1 - vgg_pred_1))
    vgg_2_mse = K.mean(K.square(vgg_true_2 - vgg_pred_2))
    vgg_3_mse = K.mean(K.square(vgg_true_3 - vgg_pred_3))
    vgg_4_mse = K.mean(K.square(vgg_true_4 - vgg_pred_4))
    vgg_5_mse = K.mean(K.square(vgg_true_5 - vgg_pred_5))

    return (vgg_1_mse + vgg_2_mse + vgg_3_mse + vgg_4_mse + vgg_5_mse) / 5.0


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
    psnr = tf.image.psnr(y_true, y_pred, max_val = 1.0)
    return 1 - psnr / 40.0


"""
The SSIM loss function keeps the result image structure
(The more significant the SSIM the more similar the final image is)
"""
def ssim(y_true, y_pred):
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return 1 - ssim


"""
PyramidFeatureExtraction is a class that makes the pyramid of the input image of depth 4. Then the CNNs extracts 
features of the image at each level and concatenates them with the features captured from the previous level.
This layer output is a 4-dimensional tuple with extracted features from each level. The first element of that tuple
are the features from the finest level.
"""
class PyramidFeatureExtraction(layers.Layer):
    def __init__(self, filter_count=[16, 32, 64], filter_size=(3, 3), activation='relu', regularizer=None, **kwargs):
        super(PyramidFeatureExtraction, self).__init__(**kwargs)
        
        # for pyramid creation
        self.downsample_avg = layers.AveragePooling2D((2, 2))
        
        # for feature extration (those layers are shared)
        self.cnn_1st_level = layers.Conv2D(filter_count[0], filter_size, activation=activation, kernel_regularizer=regularizer, padding='same')
        self.cnn_2nd_level = layers.Conv2D(filter_count[1], filter_size, activation=activation, kernel_regularizer=regularizer, padding='same')
        self.cnn_3rd_level = layers.Conv2D(filter_count[2], filter_size, activation=activation, kernel_regularizer=regularizer, padding='same')
        
        # concatenation layers
        self.concat_2nd_level = layers.Concatenate()
        self.concat_3rd_level = layers.Concatenate()
        self.concat_4th_level = layers.Concatenate()
        
        self.filter_count = filter_count
        self.filter_size = filter_size
        self.activation = activation
        self.regularizer = regularizer
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filter_count": self.filter_count,
            "filter_size": self.filter_size,
            "activation": self.activation,
            "regularizer": self.regularizer,
        })
        return config

    def call(self, inputs):
        # pyramid
        input_1 = inputs
        input_2 = self.downsample_avg(input_1)
        input_3 = self.downsample_avg(input_2)
        input_4 = self.downsample_avg(input_3)
        
        # feature extraction for layer 1
        input_1_column_1_row_1 = self.cnn_1st_level(input_1)
        input_2_column_1_row_2 = self.cnn_1st_level(input_2)
        input_3_column_1_row_3 = self.cnn_1st_level(input_3)
        input_4_column_1_row_4 = self.cnn_1st_level(input_4)
        
        # downsample layer 1
        input_1_column_2_row_2 = self.downsample_avg(input_1_column_1_row_1)
        input_2_column_2_row_3 = self.downsample_avg(input_2_column_1_row_2)
        input_3_column_2_row_4 = self.downsample_avg(input_3_column_1_row_3)
        
        # feature extraction for layer 2
        input_1_column_2_row_2 = self.cnn_2nd_level(input_1_column_2_row_2)
        input_2_column_2_row_3 = self.cnn_2nd_level(input_2_column_2_row_3)
        input_3_column_2_row_4 = self.cnn_2nd_level(input_3_column_2_row_4)
        
        # downsample layer 2
        input_1_column_3_row_3 = self.downsample_avg(input_1_column_2_row_2)
        input_2_column_3_row_4 = self.downsample_avg(input_2_column_2_row_3)
        
        # feature extraction for layer 3
        input_1_column_3_row_3 = self.cnn_3rd_level(input_1_column_3_row_3)
        input_2_column_3_row_4 = self.cnn_3rd_level(input_2_column_3_row_4)
        
        # concatenate
        concat_1st = input_1_column_1_row_1
        concat_2nd = self.concat_2nd_level([input_2_column_1_row_2, input_1_column_2_row_2])
        concat_3rd = self.concat_3rd_level([input_3_column_1_row_3, input_2_column_2_row_3, input_1_column_3_row_3])
        concat_4th = self.concat_4th_level([input_4_column_1_row_4, input_3_column_2_row_4, input_2_column_3_row_4])
        
        return concat_1st, concat_2nd, concat_3rd, concat_4th
    

"""
BidirectionalFlowEstimation is a layer that warps the features extracted at the given level trying 
to modify them to fit the target image. It predicts the flow between features of both the input_1 
and the input_2 and warps them to get the final output. The output shape is the same as the input shape.
"""
class BidirectionalFlowEstimation(layers.Layer):
    def __init__(self, filter_count=[32, 64, 64, 16], filter_size=[(3, 3), (3, 3), (1, 1), (1, 1)], activation='relu', regularizer=None, interpolation='bilinear', **kwargs):
        super(BidirectionalFlowEstimation, self).__init__(**kwargs)
        
        # flow 1 -> 2
        self.flow_add_1_2 = layers.Add()
        self.flow_upsample_1_2 = layers.UpSampling2D((2, 2), interpolation=interpolation)
        self.flow_1_2_concat = layers.Concatenate(axis=3)
        self.flow_prediction_1_2 =  keras.Sequential([
            layers.Conv2D(filter_count[0], filter_size[0], activation=activation, kernel_regularizer=regularizer, padding='same'),
            layers.Conv2D(filter_count[1], filter_size[1], activation=activation, kernel_regularizer=regularizer, padding='same'),
            layers.Conv2D(filter_count[2], filter_size[2], activation=activation, kernel_regularizer=regularizer, padding='same'),
            layers.Conv2D(filter_count[3], filter_size[3], activation=activation, kernel_regularizer=regularizer, padding='same'),
            layers.Conv2D(2, (1, 1), kernel_regularizer=regularizer, padding='same')
        ])
        
        # flow 2 -> 1
        self.flow_add_2_1 = layers.Add()
        self.flow_upsample_2_1 = layers.UpSampling2D((2, 2), interpolation=interpolation)
        self.flow_2_1_concat = layers.Concatenate(axis=3)
        self.flow_prediction_2_1 = keras.Sequential([
            layers.Conv2D(filter_count[0], filter_size[0], activation=activation, kernel_regularizer=regularizer, padding='same'),
            layers.Conv2D(filter_count[1], filter_size[1], activation=activation, kernel_regularizer=regularizer, padding='same'),
            layers.Conv2D(filter_count[2], filter_size[2], activation=activation, kernel_regularizer=regularizer, padding='same'),
            layers.Conv2D(filter_count[3], filter_size[3], activation=activation, kernel_regularizer=regularizer, padding='same'),
            layers.Conv2D(2, (1, 1), kernel_regularizer=regularizer, padding='same')
        ])
        
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
        
        # input_1 to input_2 flow prediction
        input_1_warped_1 = tfa.image.dense_image_warp(input_1, flow_1_2)
            
        flow_change_1_2_concat = self.flow_1_2_concat([input_2, input_1_warped_1])
        flow_change_1_2 = self.flow_prediction_1_2(flow_change_1_2_concat)
        
        flow_1_2_changed = self.flow_add_1_2([flow_1_2, flow_change_1_2])
        input_1_warped_2 = tfa.image.dense_image_warp(input_1, flow_1_2_changed)
        flow_1_2_changed_upsampled = self.flow_upsample_1_2(flow_1_2_changed)
        
        # input_2 to input_1 flow prediction
        input_2_warped_1 = tfa.image.dense_image_warp(input_2, flow_2_1)
            
        flow_change_2_1_concat = self.flow_2_1_concat([input_1, input_2_warped_1])
        flow_change_2_1 = self.flow_prediction_2_1(flow_change_2_1_concat)

        flow_2_1_changed = self.flow_add_2_1([flow_2_1, flow_change_2_1])
        input_2_warped_2 = tfa.image.dense_image_warp(input_2, flow_2_1_changed)
        flow_2_1_changed_upsampled = self.flow_upsample_2_1(flow_2_1_changed)
        
        return input_1_warped_2, input_2_warped_2, flow_1_2_changed_upsampled, flow_2_1_changed_upsampled
    

"""
This output functions cuts the linear function to the range from 0 to 1
"""
def output_activation(x):
    return tf.math.minimum(tf.math.maximum(x, 0), 1)


"""
WarpedFeatureFusion is a layer that concatenates and fuses all the warped features 
obtained from the FlowEstimator layer. This layer generates the final output.
"""
class WarpedFeatureFusion(layers.Layer):
    def __init__(self, feature_extractor_filter_count=[16, 32, 64], filter_size=(3, 3), activation='relu', regularizer=None, interpolation='bilinear', **kwargs):
        super(WarpedFeatureFusion, self).__init__(**kwargs)
        fefc = feature_extractor_filter_count
        
        self.cnn_1st_level_o = layers.Conv2D(3, (1, 1), activation=output_activation, padding='same')
        self.add_1st_level = layers.Add()
        
        self.up_2nd_level = layers.UpSampling2D((2, 2), interpolation=interpolation)
        self.cnn_2nd_level_2 = layers.Conv2D(fefc[0], filter_size, activation=activation, kernel_regularizer=regularizer, padding='same')
        self.cnn_2nd_level_1 = layers.Conv2D(fefc[0], filter_size, activation=activation, kernel_regularizer=regularizer, padding='same')
        self.add_2nd_level = layers.Add()
        
        self.up_3rd_level = layers.UpSampling2D((2, 2), interpolation=interpolation)
        self.cnn_3rd_level_2 = layers.Conv2D(fefc[0] + fefc[1], filter_size, activation=activation, kernel_regularizer=regularizer, padding='same')
        self.cnn_3rd_level_1 = layers.Conv2D(fefc[0] + fefc[1], filter_size, activation=activation, kernel_regularizer=regularizer, padding='same')
        self.add_3rd_level = layers.Add()
        
        self.up_4th_level = layers.UpSampling2D((2, 2), interpolation=interpolation)
        self.cnn_4th_level_2 = layers.Conv2D(fefc[0] + fefc[1] + fefc[2], filter_size, activation=activation, kernel_regularizer=regularizer, padding='same')
        self.cnn_4th_level_1 = layers.Conv2D(fefc[0] + fefc[1] + fefc[2], filter_size, activation=activation, kernel_regularizer=regularizer, padding='same')
        self.add_4th_level = layers.Add()
        
        self.feature_extractor_filter_count = feature_extractor_filter_count
        self.filter_size = filter_size
        self.activation = activation
        self.regularizer = regularizer
        self.interpolation = interpolation
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_extractor_filter_count": self.feature_extractor_filter_count,
            "filter_size": self.filter_size,
            "activation": self.activation,
            "regularizer": self.regularizer,
            "interpolation": self.interpolation,
        })
        return config
    
    def call(self, inputs):
        input_1 = inputs[0]
        input_2 = inputs[1]
        
        # merge 4th level
        added_4th_level = self.add_4th_level([input_1[3], input_2[3]])
        cnn_4th_1 = self.cnn_4th_level_1(added_4th_level)
        cnn_4th_2 = self.cnn_4th_level_2(cnn_4th_1)
        up_4th = self.up_4th_level(cnn_4th_2)
        
        # merge 3rd level
        added_3rd_level = self.add_3rd_level([input_1[2], input_2[2], up_4th])
        cnn_3rd_1 = self.cnn_3rd_level_1(added_3rd_level)
        cnn_3rd_2 = self.cnn_3rd_level_2(cnn_3rd_1)
        up_3rd = self.up_3rd_level(cnn_3rd_2)
        
        # merge 2nd level
        added_2nd_level = self.add_2nd_level([input_1[1], input_2[1], up_3rd])
        cnn_2nd_1 = self.cnn_2nd_level_1(added_2nd_level)
        cnn_2nd_2 = self.cnn_2nd_level_2(cnn_2nd_1)
        up_2nd = self.up_3rd_level(cnn_2nd_2)
        
        # merge 1st level
        added_1st_level = self.add_1st_level([input_1[0], input_2[0], up_2nd])
        outputs = self.cnn_1st_level_o(added_1st_level)
    
        return outputs
    

"""
FBNet is a layer that merges all the custom layers declared above making
the full ready-to-use frame interpolation network.
"""
class FBNet(layers.Layer):
    def __init__(self, pyramid_filter_count=[16, 32, 64], pyramid_filter_size=(3, 3), flow_filter_count=[32, 64, 64, 16], flow_filter_sizes=[(3, 3), (3, 3), (1, 1), (1, 1)], activation='relu', regularizer=None, interpolation='bilinear', **kwargs):
        super(FBNet, self).__init__(**kwargs)
        
        assert len(pyramid_filter_count) == 3, 'pyramid_filter_count length is expected to be 3. Got: ' + str(len(pyramid_filter_count))
        assert len(pyramid_filter_size) == 2, 'pyramid_filter_size length is expected to be 2. Got: ' + str(len(pyramid_filter_size))
        assert len(flow_filter_count) == 4, 'flow_filter_count length is expected to be 4. Got ' + str(len(flow_filter_count))
        assert len(flow_filter_sizes) == 4, 'flow_filter_sizes length is expected to be 4. Got ' + str(len(flow_filter_sizes))
        
        # for building the pyramid
        self.pyramid_input_1 = PyramidFeatureExtraction(
            filter_count=pyramid_filter_count, 
            filter_size=pyramid_filter_size, 
            activation=activation, 
            regularizer=regularizer
        )
        self.pyramid_input_2 = PyramidFeatureExtraction(
            filter_count=pyramid_filter_count, 
            filter_size=pyramid_filter_size, 
            activation=activation, 
            regularizer=regularizer
        )
        
        # for flow estimation
        self.bidirectional_flow_estimation_1 = BidirectionalFlowEstimation(
            filter_count=flow_filter_count, 
            filter_size=flow_filter_sizes, 
            activation=activation, 
            regularizer=regularizer, 
            interpolation=interpolation
        )
        self.bidirectional_flow_estimation_2 = BidirectionalFlowEstimation(
            filter_count=flow_filter_count, 
            filter_size=flow_filter_sizes, 
            activation=activation, 
            regularizer=regularizer, 
            interpolation=interpolation
        )
        self.bidirectional_flow_estimation_3 = BidirectionalFlowEstimation(
            filter_count=flow_filter_count, 
            filter_size=flow_filter_sizes, 
            activation=activation, 
            regularizer=regularizer, 
            interpolation=interpolation
        )
        self.bidirectional_flow_estimation_4 = BidirectionalFlowEstimation(
            filter_count=flow_filter_count, 
            filter_size=flow_filter_sizes, 
            activation=activation, 
            regularizer=regularizer, 
            interpolation=interpolation
        )
        
        # for the final fusion
        self.warped_feature_fusion = WarpedFeatureFusion(
            feature_extractor_filter_count=pyramid_filter_count, 
            filter_size=pyramid_filter_size, 
            activation=activation, 
            regularizer=regularizer, 
            interpolation=interpolation
        )
        
        self.pyramid_filter_count = pyramid_filter_count
        self.pyramid_filter_size = pyramid_filter_size
        self.flow_filter_count = flow_filter_count
        self.flow_filter_sizes = flow_filter_sizes
        self.activation = activation
        self.regularizer = regularizer
        self.interpolation = interpolation
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "pyramid_filter_count": self.pyramid_filter_count,
            "pyramid_filter_size": self.pyramid_filter_size,
            "flow_filter_count": self.flow_filter_count,
            "flow_filter_sizes": self.flow_filter_sizes,
            "activation": self.activation,
            "regularizer": self.regularizer,
            "interpolation": self.interpolation,
        })
        return config
    
    def call(self, inputs, batch_size=1):
        # these are the input images
        input_1 = inputs[0]
        input_2 = inputs[1]

        # feature extractor layers for each image indepedantly
        feature_extraction_1 = self.pyramid_input_1(input_1)
        feature_extraction_2 = self.pyramid_input_2(input_2)

        # create empty flow for the coarest level
        empty_flow_1 = tf.zeros(shape=(batch_size, feature_extraction_1[3].shape[1], feature_extraction_1[3].shape[2], 2))
        empty_flow_2 = tf.zeros(shape=(batch_size, feature_extraction_1[3].shape[1], feature_extraction_1[3].shape[2], 2))

        # calculate the flow for each level using the input of current level and the upsampled flow from the level + 1
        bfe_4_i1, bfe_4_i2, bfe_4_f_1_2, bfe_4_f_2_1 = self.bidirectional_flow_estimation_1([feature_extraction_1[3], feature_extraction_2[3], empty_flow_1, empty_flow_2])
        bfe_3_i1, bfe_3_i2, bfe_3_f_1_2, bfe_3_f_2_1 = self.bidirectional_flow_estimation_2([feature_extraction_1[2], feature_extraction_2[2], bfe_4_f_1_2, bfe_4_f_2_1])
        bfe_2_i1, bfe_2_i2, bfe_2_f_1_2, bfe_2_f_2_1 = self.bidirectional_flow_estimation_3([feature_extraction_1[1], feature_extraction_2[1], bfe_3_f_1_2, bfe_3_f_2_1])
        bfe_1_i1, bfe_1_i2, _, _ = self.bidirectional_flow_estimation_4([feature_extraction_1[0], feature_extraction_2[0], bfe_2_f_1_2, bfe_2_f_2_1])

        # merge the features extracted by the previous layers
        outputs = self.warped_feature_fusion([(bfe_1_i1, bfe_2_i1, bfe_3_i1, bfe_4_i1), (bfe_1_i2, bfe_2_i2, bfe_3_i2, bfe_4_i2)])

        return outputs
    

def loss(y_true, y_pred):
    ssim_ = ssim(y_true, y_pred)
    psnr_ = psnr(y_true, y_pred)
    l1_ = l1(y_true, y_pred)
    l2_ = l2(y_true, y_pred)
    return ssim_ + psnr_ + 5.0*l1_ + 10.0*l2_
