import model_v1.modules as modules
import numpy as np
import argparse
import keras
import math
import cv2
import os

from create_data import preprocess


"""
The function parses the program arguments
"""
def get_parser():
    parser = argparse.ArgumentParser(description='Frame boosting environment for FBNet')
    parser.add_argument("-s", "--source", required=True, type=str, help="The path of the source video file to process")
    parser.add_argument("-t", "--target", required=True, type=str, help="The path where the results videos will be created")
    parser.add_argument("-vn", "--video_name", required=False, type=str, default="result", help="The name of the result video")
    parser.add_argument("-m", "--model", required=True, type=str, help="The path of model used during the video processing")
    parser.add_argument("-c", "--count", required=False, type=int, default=2, help="The frame multiplying coefficient (2x, 4x, 8x, 16x)")
    parser.add_argument("-md", "--mode", required=False, type=str, default='fast', help="The processing mode. Either 'fast' or 'low_mem'")
    parser.add_argument("-iw", "--width", required=False, type=int, default=256, help="The width of target images")
    parser.add_argument("-ih", "--height", required=False, type=int, default=144, help="The height of target images")
    parser.add_argument("-e", "--extension", required=False, type=str, default='avi', help="The final file format")
    return parser.parse_args()


"""
The function loads the trained FBNet model
"""
def load_model(path):
    return keras.models.load_model(
        path,
        custom_objects = {
            "PyramidFeatureExtraction": modules.PyramidFeatureExtraction,
            "BidirectionalFlowEstimation": modules.BidirectionalFlowEstimation,
            "WarpedFeatureFusion": modules.WarpedFeatureFusion,
            "FBNet": modules.FBNet,
            'loss': modules.loss,
            'perceptual': modules.perceptual,
            'l1': modules.l1,
            "ssim": modules.ssim,
            "psnr": modules.psnr,
            "l2": modules.l2
        }
    )


"""
The function transforms the sequence of frames into sequence of list containing opposite frames to fit the network input
"""
def create_net_input(frames):
    result = []
    current = frames[0]
    first, second = None, None
    for frame in frames[1:]:
        second = first
        first = current

        if first is not None and second is not None:
            result.append([first, second])

        current = frame

    return np.array(result)


"""
The function combines 2 sequences of frames (the original and the generated) into the single one
"""
def combine_frames(frames, new_frames):
    result = []

    for index in range(len(frames)):
        result.append(frames[index])
        if index < len(new_frames):
            result.append(new_frames[index])

    return result


"""
The function boosts the frames in the pointed video. The method is fast but requires lots of memory due to all the frames being kept in the memory
"""
def fast_boosting(source_path, target_path, name, model, count, size, extension):
    if extension == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif extension == "avi":
        fourcc = cv2.VideoWriter_fourcc('I','4','2','0')

    vidcap = cv2.VideoCapture(source_path)
    success, frame = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    original_frames = []
    frames = []

    while success:
        frame = preprocess(frame, size, cv2.INTER_LINEAR)
        original_frames.append(frame)
        success, frame = vidcap.read()

    frames = original_frames.copy()
    original_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in original_frames]
    original_frames = (np.array(original_frames) * 255).astype('uint8')

    for _ in range(int(math.log2(count))):
        net_input = create_net_input(frames)
        new_frames = model.predict([net_input[:, 0, :, : ,:], net_input[:, 1, :, : ,:]], batch_size=1)
        frames = combine_frames(frames, new_frames)

    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    frames = (np.array(frames) * 255).astype('uint8')

    out = cv2.VideoWriter(os.path.join(target_path, f'{name}.{extension}'), fourcc, count*fps, size)
    for frame in frames:
        out.write(frame)

    out = cv2.VideoWriter(os.path.join(target_path, f'{name}_comparision.{extension}'), fourcc, (count*fps)/5.0, (size[0]*2, size[1]))
    for index in range(len(original_frames)):
        coef = count
        for index_generated in range(coef):
            if coef * index + index_generated >= len(frames):
                break
            frame = cv2.hconcat([original_frames[index], frames[coef * index + index_generated]])
            out.write(frame)


"""
The function boosts fps in the pointed video. It is slow but uses significantly less memory than the fast method
"""
def low_mem_boosting(source_path, target_path, name, model, count, extension):
    pass


if __name__ == "__main__":
    parser = get_parser()

    source_path = parser.source
    target_path = parser.target
    name = parser.video_name
    model_path = parser.model
    count = parser.count
    mode = parser.mode
    width = parser.width
    height = parser.height
    extension = parser.extension
    size = (width, height)

    assert count in [2, 4, 8, 16, 32, 64], "Count has to be one of 2, 4, 8, 16"
    assert mode in ['fast', 'low_mem'], "Mode has to be either 'fast' or 'low_mem'"
    assert extension in ['mp4', 'avi'], "Extension is expected to be one of the following: 'mp4', 'avi'"

    model = load_model(model_path)
    if mode == 'fast':
        fast_boosting(source_path, target_path, name, model, count, size, extension)
    elif mode == 'low_mem':
        low_mem_boosting(source_path, target_path, name, model, count, size, extension)
