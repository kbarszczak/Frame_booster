![banner (1)](https://user-images.githubusercontent.com/72699445/232047516-68669452-efd0-4500-9c52-c4a377ff4a11.png)

The project uses AI techniques to generate new frames in the video. The key factor of the system is the neural network that uses mostly convolutional layers to predict the final frame.

## Motivation

The key motivation is to build a system that generates new frames more accurately than existing methods such as interpolation or optical flow. The idea is to create a neural network that properly moves the objects on the scene without leaving blank spots behind them.

## Build Status

The FBNet model version v1 is already built and trained using Vimeo90K triplet dataset. There are written scripts that can be used either to train your own version of the net or test already existing one. The pretrained model can be found here: models/model_v1/FBNet.h5

## Screenshots

Some results of the model in version 1.0

![best_6](https://user-images.githubusercontent.com/72699445/235080906-dc429c74-6286-4280-a42c-e79a961aa8fd.png)

And the video shows the original video on the left and the result of boosting the frame rate 4x on the right.

[![result_video_alt](https://img.youtube.com/vi/844G_KYDchw/0.jpg)](https://www.youtube.com/watch?v=844G_KYDchw)

## Tech/Framework used

The project uses the following tech/frameworks:
- python 3.10
- keras/tensorflow
- opencv
- numpy

## Features

The net can process every type of video without any restrictions regarding its length. The only restriction is that the generated video is always in size 144x256 px. The net can also be used to predict a single image from 2 similar images rather than predicting the frames based on the sequence of frames.

## Installation

1. Clone the repository
```
mkdir frame_booster
cd frame_booster
git clone https://github.com/kbarszczak/Frame_booster .
```

2. Set up python environment
    1.   Either install the requirements
    ```
    pip install -r requirements.txt
    ```

    2. or create the conda environment and install the requirements inside a conda container
    ```
    conda create --name frame_booster
    conda activate frame_booster
    pip install -r requirements.txt
    ```

After these steps, everything is set up and ready to use.

## How to use?

1. **Frame boosting** may be performed by launching the 'src/frame_generator.py' script with the following switches:
- -s the filename of the source video (absolute or relative)
- -m the path to the trained model saved in a .h5 format
- -t the path to a dictionary where the results files will be created
- -vn the name of the created video
- -c the boosting rate (2, 4, 8, 16, 32). example: rate 4 will add 3 new frames between each frame in the original video
- -e the extension of the result file (mp4, avi)
- -md the mode of the generator (fast, low_mem)
- -iw the width of the net input
- -ih the height of the net input
Example:
```
python src/frame_generator.py -s 'test.mp4' -m 'FBNet.h5' -t 'C:/Users/kamil/test' -c 2 -vn 'test_result_2x' -md fast -e avi
```

2. **To train** your own net one may use the 'src/model_v1/model.py' with the switches:
- -tr the path to the training .tfrecords file
- -trc the amount of training data
- -ts the path to the testing .tfrecords file
- -tsc the amount of testing data
- -v the path to the validating .tfrecords file
- -vc the amount of validating data
- -t the path where trained models will be saved
- -n the name of the saved model
- -b the batch size
- -e the epochs
- -iw the input images width
- -ih the input images height
Example:
```
python src/model_v1/model.py -tr train_144x256_19000.tfrecords -trc 19000 -ts test_144x256_1000.tfrecords -tsc 1000 -v valid_144x256_500.tfrecords -vc 500 -t C:/Users/kamil/test -n model -b 5 -e 10
```

3. **Create your datasets** by running 'src/create_data.py' with the following switches:
- -s the source path of the raw dataset to process
- -t the target path where the result files will be created
- -l the loaded script (vimeo90k, custom). In the case of vimeo90k, the -d parameter is ignored and the source path has to point to the vimeo90k triplet dataset where the following files/dirs are: sequences, tri_trainlist.txt, tri_testlist.txt. In custom, the path has to point to a directory containing only the video files. Each file will be loaded and processed
- -tr the limit of the train data
- -ts the limit of the test data
- -tv the split ratio between training and validating datasets
- -i the interpolation method (bilinear, bicubic)
- -iw the width of the target images
- -ih the height of the target images
- -d the delay parameres means that d - 1 frames will be skipped between generated data (applies only to -l set to custom)
Example:
```
python src/create_data.py -s C:/Users/kamil/raw/240fps_horizontal -t data -l custom -tr 1000 -ts 500 -tv 0.9 -i bicubic -iw 256 -ih 144 -d 5
```

## Contribute
- clone the repository
- make the changes
- create the pull request with a detailed description of your changes

## Acknowledgements

```
@article{xue2019video,
  title={Video Enhancement with Task-Oriented Flow},
  author={Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
  journal={International Journal of Computer Vision (IJCV)},
  volume={127},
  number={8},
  pages={1106--1125},
  year={2019},
  publisher={Springer}
}
```
