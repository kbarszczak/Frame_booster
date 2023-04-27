import tensorflow as tf
import numpy as np
import argparse
import random
import cv2
import os

"""
The function parses the program arguments
"""
def get_parser():
    parser = argparse.ArgumentParser(description='Creating dataset for training FBNet model')
    parser.add_argument("-s", "--source", required=True, type=str, help="The source path of the raw dataset to process")
    parser.add_argument("-t", "--target", required=True, type=str, help="The target path where the results files will be created")
    parser.add_argument("-l", "--loader", required=True, type=str, default='vimeo90k', help="The name of the function that loads the data for tfrecords processor")
    parser.add_argument("-tr", "--train_limit", required=False, type=int, default=1000, help="The max samples count in the training set")
    parser.add_argument("-ts", "--test_limit", required=False, type=int, default=200, help="The max samples count in the testing set")
    parser.add_argument("-tv", "--valid_split", required=False, type=float, default=0.9, help="The train to validation split")
    parser.add_argument("-i", "--interpolation", required=False, type=str, default='bicubic', help="The interpolation type")
    parser.add_argument("-iw", "--width", required=False, type=int, default=256, help="The width of target images")
    parser.add_argument("-ih", "--height", required=False, type=int, default=144, help="The height of target images")
    parser.add_argument("-d", "--delay", required=False, type=int, default=5, help="The delay that is applied during loading frames from the custom dataset. It is used only with -l option set to 'custom'")
    return parser.parse_args()


"""
The processes each image before it is saved to the result file
"""
def preprocess(img, size, interpolation):
    img = cv2.resize(img, size, interpolation=interpolation)
    return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0).astype('float32')


"""
The function converts numpy array (written in bytes) to the Feature object understadable for TFRecordWriter
"""
def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


"""
The function creates a TFRecord example that is stored in the tfrecords file
"""
def sample_example(imgs):
    feature = {
        'image_1': bytes_feature(imgs[0].tobytes()),
        'image_2': bytes_feature(imgs[1].tobytes()),
        'image_3': bytes_feature(imgs[2].tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


"""
The function reads the file_path to get the data split information and returns proper sequence of paths with images to read
"""
def create_paths_vimeo90k(file_path, limit=None, split=None):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    postfixes = [line.strip() for line in lines]
    random.shuffle(postfixes)
    if limit is not None:
        postfixes = postfixes[0:limit]
        
    if split is not None:
        index = int(len(postfixes) * split)
        return postfixes[0:index], postfixes[index:]
    else:
        return postfixes
    

"""
The function reads each record from base_path + path in paths and saves it to the result .tfrecords file
"""
def create_tf_record_vimeo90k(base_path, paths, target_path, size, interpolation):
    with tf.io.TFRecordWriter(target_path) as writer:
        for path in paths:
            full_path = os.path.join(base_path, path)
            imgs = []
            for name in ['im1.png', 'im2.png', 'im3.png']:
                img = cv2.imread(os.path.join(full_path, name))
                img = preprocess(img, size, interpolation)
                imgs.append(img)

            tf_example = sample_example(imgs)
            writer.write(tf_example.SerializeToString())


"""
The function handles processing vimeo90k triple dataset with the original naming of the train and test split filenames ('tri_trainlist.txt', 'tri_testlist.txt')
"""
def vimeo90k(base_path, target_path, size, interpolation, train_limit, test_limit, train2valid_split):
    train_paths, valid_paths = create_paths_vimeo90k(os.path.join(base_path, "tri_trainlist.txt"), limit = train_limit, split=train2valid_split)
    test_paths = create_paths_vimeo90k(os.path.join(base_path, "tri_testlist.txt"), limit = test_limit, split=None)

    print("Processing train dataset ...")
    train_target_path = os.path.join(target_path, f'train_{size[1]}x{size[0]}_{len(train_paths)}.tfrecords')
    create_tf_record_vimeo90k(os.path.join(base_path, 'sequences'), train_paths, train_target_path, size, interpolation)
    print(f'Train dataset successfully created at "{train_target_path}"')

    print("Processing test dataset ...")
    test_target_path = os.path.join(target_path, f'test_{size[1]}x{size[0]}_{len(test_paths)}.tfrecords')
    create_tf_record_vimeo90k(os.path.join(base_path, 'sequences'), test_paths, test_target_path, size, interpolation)
    print(f'Test dataset successfully created at "{test_target_path}"')

    print("Processing validation dataset ...")
    valid_target_path = os.path.join(target_path, f'valid_{size[1]}x{size[0]}_{len(valid_paths)}.tfrecords')
    create_tf_record_vimeo90k(os.path.join(base_path, 'sequences'), valid_paths, valid_target_path, size, interpolation)
    print(f'Validation dataset successfully created at "{test_target_path}"')


"""
The function reads each video from the pointed directory and returns the list of tuples. Each tuple contains 3 opposite frames
"""
def read_opposite_frames(base_path, size, interpolation, delay, limit):
    data = []
    for video_filename in os.listdir(base_path):
        video_path = os.path.join(base_path, video_filename)
        if os.path.isfile(video_path):    
            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            image = preprocess(image, size, interpolation)
            
            first = image
            second = None
            third = None
            
            index = 1
            while success:
                if first is not None and second is not None and third is not None:
                    data.append(np.array((first, second, third)))
                    first = None
                    second = None
                    third = None
                
                success, image = vidcap.read()
                index += 1
                
                if not success:
                    break
                  
                if index % delay != 0:
                    continue
                    
                image = preprocess(image, size, interpolation)
                third = second
                second = first
                first = image

                if limit is not None and len(data) >= limit:
                    return data

    return data


"""
The function takes the data and saves it to the .tfrecord file under the pointed directory
"""
def create_tfrecord_custom(data, target_path):
     with tf.io.TFRecordWriter(target_path) as writer:
        for d in data:
            tf_example = sample_example(d)
            writer.write(tf_example.SerializeToString())


"""
The function creates the .tfrecords files using the custom videos located under the pointed directory
"""
def custom(base_path, target_path, size, interpolation, train_limit, test_limit, train2valid_split, delay):
    print(f'Reading frames from {base_path}/* ...')
    data = read_opposite_frames(base_path, size, interpolation, delay, train_limit + test_limit)

    print("Splitting the data ...")
    random.shuffle(data)
    train = data[0: train_limit]
    valid = train[0: int(len(train) * train2valid_split)]
    train = train[int(len(train) * train2valid_split):]
    test = data[train_limit:]

    print("Saving the data ...")
    create_tfrecord_custom(train, os.path.join(target_path, f'train_{size[1]}x{size[0]}_{len(train)}.tfrecords'))
    create_tfrecord_custom(test, os.path.join(target_path, f'test_{size[1]}x{size[0]}_{len(test)}.tfrecords'))
    create_tfrecord_custom(valid, os.path.join(target_path, f'valid_{size[1]}x{size[0]}_{len(valid)}.tfrecords'))

    print(f"Data successfully saved to '{target_path}'")


if __name__ == "__main__":
    # try:
    parser = get_parser()
    source_path = parser.source
    target_path = parser.target
    loader = parser.loader
    train_limit = parser.train_limit
    test_limit = parser.test_limit
    valid_split = parser.valid_split
    interpolation = parser.interpolation
    width = parser.width
    height = parser.height
    delay = parser.delay
    size = (width, height)
    
    assert loader in ["vimeo90k", "custom"], "The loader function can only be: vimeo90 (for vimeo90k triplet dataset) or custom (loads each video file from pointer directory)"
    assert valid_split >= 0 or valid_split <= 1, "Validation split is expected to be in range from 0 to 1"
    assert interpolation in ['bicubic', 'bilinear'], "Interpolation is either bicubic or bilinear"

    if interpolation == "bicubic":
        interpolation = cv2.INTER_CUBIC
    elif interpolation == 'bilinear':
        interpolation = cv2.INTER_LINEAR

    if loader == "vimeo90k":
        vimeo90k(source_path, target_path, size, interpolation, train_limit, test_limit, valid_split)
    elif loader == "custom":
        custom(source_path, target_path, size, interpolation, train_limit, test_limit, valid_split, delay)
    # except Exception as e:
    #     print(e)
