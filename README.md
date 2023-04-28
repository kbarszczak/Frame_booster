![banner (1)](https://user-images.githubusercontent.com/72699445/232047516-68669452-efd0-4500-9c52-c4a377ff4a11.png)

The project uses AI techniques to generate new frames in the video. The key factor of the system is the neural network that uses mostly convolutional layers to predict the final frame.

## Motivation

The key motivation is to build a system that generates new frames more accurately than existing methods such as interpolation or optical flow. The idea is to create a neural network that properly moves the objects on the scene without leaving blank spots behind them.

## Build Status

The FBNet model version v1 is already built and trained using Vimeo90K triplet dataset. There are written scripts that can be used either to train your own version of the net or test already existing one.

## Screenshots

Some results of the model in version 1.0

![best_6](https://user-images.githubusercontent.com/72699445/235080906-dc429c74-6286-4280-a42c-e79a961aa8fd.png)

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

2a. Either install the requirements
```
pip install -r requirements.txt
```

2b. or create the conda environment and install the requirements inside a conda container
```
conda create --name frame_booster
conda activate frame_booster
pip install -r requirements.txt
```

After these steps, everything is set up and ready to use.

## How to use?

1. Frame boosting

To generate new frames use the 'src/frame_generator.py' script with the follwoing switches:
- -s the filename of the source video (absolute or relative)
- -m the path to the trained model saved in a .h5 format
- -t the path to a dictionary where the results files will be created
- -vn the name of the created video
- -c the boosting rate (2, 4, 8, 16, 32). example: rate 4 will add 3 new frames between each frames in the original video
- -e the extension of the result file (mp4, avi)
- -md the mode of the generator (fast, low_mem)
- -iw the width of the net input
- -ih the height of the net input

Example:
```
python src/frame_generator.py -s 'test.mp4' -m 'FBNet.h5' -t 'C:\Users\kamil\test' -c 2 -vn 'test_result_2x' -md fast
```

python model.py -tr E:\Data\Video_Frame_Interpolation\processed\med_motion\valid_144x256_45.tfrecords -trc 45 -ts E:\Data\Video_Frame_Interpolation\processed\med_motion\test_144x256_10.tfrecords -tsc 10 -v E:\Data\Video_Frame_Interpolation\processed\med_motion\train_144x256_5.tfrecords -vc 5 -t E:\Data\Video_Frame_Interpolation\processed\med_motion -n model
python create_data.py -s E:\Data\Video_Frame_Interpolation\raw\240fps_horizontal -t E:\Data\Video_Frame_Interpolation\processed\med_motion -l custom -tr 100 -ts 50

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
