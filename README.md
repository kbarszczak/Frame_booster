![banner (1)](https://user-images.githubusercontent.com/72699445/232047516-68669452-efd0-4500-9c52-c4a377ff4a11.png)

The project uses AI techniques to generate new frames in the video. The key factor of the system is the neural network that uses recurrent, convolutional, and dense layers to predict the final frame.

## Motivation

The key motivation is to build a system that generates new frames more accurately than existing methods such as interpolation or optical flow. The idea is to create a neural network that properly moves the objects on the scene without leaving blank spots behind them.

## Build Status

Currently, we are creating a proof of concept.

## Screenshots

todo

## Tech/Framework used

todo

## Features

todo

## Installation

todo

## How to use?

```
python frame_generator.py -s 'E:\Data\Video_Frame_Interpolation\test\test_2.mp4' -m 'E:\OneDrive - Akademia Górniczo-Hutnicza im. Stanisława Staszica w Krakowie\Programming\Labs\Frame_booster\models\model_v1\FBNet.h5' -t 'E:\Data\Video_Frame_Interpolation\test' -c 2 -vn 'test_2_result_2x'
python model.py -tr E:\Data\Video_Frame_Interpolation\processed\med_motion\valid_144x256_45.tfrecords -trc 45 -ts E:\Data\Video_Frame_Interpolation\processed\med_motion\test_144x256_10.tfrecords -tsc 10 -v E:\Data\Video_Frame_Interpolation\processed\med_motion\train_144x256_5.tfrecords -vc 5 -t E:\Data\Video_Frame_Interpolation\processed\med_motion -n model
python create_data.py -s E:\Data\Video_Frame_Interpolation\raw\240fps_horizontal -t E:\Data\Video_Frame_Interpolation\processed\med_motion -l custom -tr 100 -ts 50

```

todo

## Contribute
- clone the repository
- make the changes
- create the pull request with a detailed description of your changes
