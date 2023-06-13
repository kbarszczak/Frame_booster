import model_v1.modules as modules
import argparse
import tensorflow as tf
import keras
import time
import os
from keras import optimizers
from keras import callbacks
from keras import layers
import numpy as np


"""
The function parses the program arguments
"""
def get_parser():
    parser = argparse.ArgumentParser(description='Training FBNet model')
    parser.add_argument("-tr", "--train", required=True, type=str, help="The source path of the training dataset")
    parser.add_argument("-trc", "--train_count", required=True, type=int, help="The size of the training dataset")
    parser.add_argument("-ts", "--test", required=True, type=str, help="The source path of the testing dataset")
    parser.add_argument("-tsc", "--test_count", required=True, type=int, help="The size of the testing dataset")
    parser.add_argument("-v", "--valid", required=True, type=str, help="The source path of the validating dataset")
    parser.add_argument("-vc", "--valid_count", required=True, type=int, help="The size of the validating dataset")
    parser.add_argument("-t", "--target_path", required=True, type=str, help="The path where models are saved during the training (models are saved after each 500 batches)")
    parser.add_argument("-n", "--name", required=True, type=str, help="The name of the created model)")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=5, help="The batch size")
    parser.add_argument("-e", "--epochs", required=False, type=int, default=10, help="The epochs count")
    parser.add_argument("-iw", "--width", required=False, type=int, default=256, help="The input image widht")
    parser.add_argument("-ih", "--height", required=False, type=int, default=144, help="The input image height")
    return parser.parse_args()


"""
The feature of tfrecords files
"""
name_to_features = {
    'image_1': tf.io.FixedLenFeature([], tf.string),
    'image_2': tf.io.FixedLenFeature([], tf.string),
    'image_3': tf.io.FixedLenFeature([], tf.string),
}


"""
The fucntion parses and decodes the TFRecords for learning
"""
def parse_decode_record(record):
    features = tf.io.parse_single_example(record, name_to_features)
    image_1 = tf.io.decode_raw(
        features['image_1'], out_type='float32', little_endian=True, fixed_length=None, name=None
    )
    image_1 = tf.reshape(image_1, (144, 256, 3))
    
    image_2 = tf.io.decode_raw(
        features['image_2'], out_type='float32', little_endian=True, fixed_length=None, name=None
    )
    image_2 = tf.reshape(image_2, (144, 256, 3))
    
    image_3 = tf.io.decode_raw(
        features['image_3'], out_type='float32', little_endian=True, fixed_length=None, name=None
    )
    image_3 = tf.reshape(image_3, (144, 256, 3))
    
    return (image_1, image_3), image_2


"""
The function returns TFRecord generator
"""
def load_generator(path, epochs, batch_size):
    generator = tf.data.TFRecordDataset(path)

    generator = generator.map(parse_decode_record)
    generator = generator.repeat(epochs)
    generator = generator.prefetch(5)
    generator = generator.shuffle(buffer_size=5 * batch_size)
    generator = generator.batch(batch_size, drop_remainder=True)

    return generator


"""
The function evaluates the training quality
"""
def evaluate(model, generator):
    result_dict = {}
    result = model.evaluate(generator, verbose=0)
    for index, metric in enumerate(model.metrics):
        result_dict[metric.name] = result[index]
        print(f'{metric.name.zfill(13).replace("0", " ")}: {np.round(result[index], 4)}')
        
    return result_dict


"""
The function performs training
"""
def train(model_name, model_path, train_path, train_size, test_path, test_size, valid_path, valid_size, epochs, batch_size, height, width):
    train_generator = load_generator(
        path = train_path, 
        epochs=epochs,
        batch_size=batch_size
    )
    test_generator = load_generator(
        path = test_path, 
        epochs=epochs,
        batch_size=batch_size
    )
    valid_generator = load_generator(
        path = valid_path, 
        epochs=epochs,
        batch_size=batch_size
    )

    # create input layers
    inputs = [layers.Input(shape=(height, width, 3)), layers.Input(shape=(height, width, 3))]

    # create the net
    fb_net = modules.FBNet(
        pyramid_filter_count = [16, 32, 64], 
        pyramid_filter_size = (3, 3), 
        flow_filter_count = [48, 80, 80, 32], 
        flow_filter_sizes = [(3, 3), (3, 3), (1, 1), (1, 1)],
        activation = layers.LeakyReLU(0.2), 
        regularizer = None, 
        interpolation = 'bicubic'
    )

    # create the output layer
    outputs = fb_net(inputs, batch_size)

    # create and compile the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss = modules.loss,
        optimizer = optimizers.Nadam(0.0001, clipvalue=1.0, clipnorm=1.0),
        metrics = [modules.l1, modules.l2, modules.psnr, modules.ssim]
    )
    model.summary()

    # train the model
    _ = model.fit(
        train_generator,
        epochs=epochs,
        validation_data = valid_generator,
        steps_per_epoch = int(train_size) // batch_size,
        validation_steps = int(valid_size) // batch_size,
        callbacks = [
            callbacks.ModelCheckpoint(
                os.path.join(model_path, model_name+'_'+'{loss:.4f}_{epoch:02d}_'+str(int(time.time()))+'.h5'),
                monitor = 'loss',
                mode = 'min',
                save_best_only = True,
                save_weights_only = False,
                save_freq = 500,
            )
        ]
    )

    # print evaluation and save the results
    eval_dict = evaluate(model, test_generator)
    model_target_path = os.path.join(model_path, f'{model_name}_loss_{eval_dict["loss"]}_l1_{eval_dict["l1"]}_l2_{eval_dict["l2"]}_psnr_{eval_dict["psnr"]}_ssim_{eval_dict["ssim"]}.h5')
    model.save(model_target_path)
    print(f'Model saved to "{model_target_path}".\nTraining results:\n{eval_dict}')


if __name__ == "__main__":
    parser = get_parser()
    train(
        model_name = parser.name, 
        model_path = parser.target_path, 
        train_path = parser.train, 
        train_size = parser.train_count, 
        test_path = parser.test, 
        test_size = parser.test_count, 
        valid_path = parser.valid, 
        valid_size = parser.valid_count, 
        epochs = parser.epochs, 
        batch_size = parser.batch_size, 
        height = parser.height, 
        width = parser.width
    )
    