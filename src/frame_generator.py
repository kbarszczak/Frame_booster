import numpy as np
import argparse
import keras
import math
import cv2
import os


"""
The function parses the program arguments
"""
def get_parser():
    parser = argparse.ArgumentParser(description='Frame boosting environment for FBNet')
    parser.add_argument("-s", "--source", required=True, type=str, help="The path of the source video file to process")
    parser.add_argument("-t", "--target", required=True, type=str, help="The path where the results videos will be created")
    parser.add_argument("-m", "--model", required=True, type=str, help="The path of model used during the video processing")
    parser.add_argument("-mv", "--model_version", required=False, type=str, default="v1", help="The version of the model (v1, v2, v3)")
    parser.add_argument("-vn", "--video_name", required=False, type=str, default="result", help="The name of the result video")
    parser.add_argument("-c", "--count", required=False, type=int, default=2, help="The frame multiplying coefficient (2x, 4x, 8x, 16x)")
    parser.add_argument("-md", "--mode", required=False, type=str, default='low_mem', help="The processing mode. Either 'fast' or 'low_mem'")
    parser.add_argument("-lml", "--low_mem_limit", required=False, type=int, default=500, help="The limit of the input size for the net. Applies only if the mode is set to 'low_mem' (-md switch)")
    parser.add_argument("-iw", "--width", required=False, type=int, default=256, help="The width of target images")
    parser.add_argument("-ih", "--height", required=False, type=int, default=144, help="The height of target images")
    parser.add_argument("-e", "--extension", required=False, type=str, default='avi', help="The final file format")
    return parser.parse_args()


"""
The function transforms the sequence of frames into sequence of list containing opposite frames to fit the network input
"""
def create_net_input(frames):
    result = []
    first, second = frames[0], None
    for frame in frames[1:]:
        second = first
        first = frame
        result.append([first, second])

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
def fast_boosting(source_path, target_path, cmp_target_path, model, count, size, fourcc):
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
        new_frames = model([net_input[:, 0, :, : ,:], net_input[:, 1, :, : ,:]])
        frames = combine_frames(frames, new_frames)

    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    frames = (np.array(frames) * 255).astype('uint8')

    out = cv2.VideoWriter(target_path, fourcc, count*fps, size)
    for frame in frames:
        out.write(frame)

    out = cv2.VideoWriter(cmp_target_path, fourcc, (count*fps), (size[0]*2, size[1]))
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
def low_mem_boosting(source_path, target_path, cmp_target_path, model, count, size, fourcc, mem_limit):
    vidcap = cv2.VideoCapture(source_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    result_out = cv2.VideoWriter(target_path, fourcc, count*fps, size)
    cmp_result_out = cv2.VideoWriter(cmp_target_path, fourcc, (count*fps), (size[0]*2, size[1]))

    has_frames = True
    last_frame = None
    index = 0
    while has_frames:
        frames = []
        if last_frame is not None:
            frames.append(last_frame)
            
        while len(frames) < mem_limit:
            success, frame = vidcap.read()
            if not success:
                has_frames = False
                break

            frame = preprocess(frame, size, cv2.INTER_LINEAR)
            frames.append(frame)
            last_frame = frame

        if len(frames) >= 2:
            for _ in range(int(math.log2(count))):
                result_frames_at_level = []
                for part in range(int(math.ceil(len(frames)/mem_limit))):
                    start = max(part * mem_limit - 1, 0)
                    end = min((part + 1) * mem_limit, len(frames))

                    frames_for_net = frames[start:end]
                    net_input = create_net_input(frames_for_net)

                    generated_frames = model([net_input[:, 0, :, : ,:], net_input[:, 1, :, : ,:]])
                    combined_frames = combine_frames(frames_for_net, generated_frames)
                    if start > 0:
                        combined_frames = combined_frames[1:]
                    result_frames_at_level.extend(combined_frames)

                frames = result_frames_at_level
            
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        frames = (np.array(frames) * 255).astype('uint8')

        if index > 0:
            frames = frames[1:]

        for frame in frames:
            result_out.write(frame)

            if index % count == 0:
                last_original_frame = frame
            
            cmp_frame = cv2.hconcat([last_original_frame, frame])
            cmp_result_out.write(cmp_frame)
            index += 1


if __name__ == "__main__":
    parser = get_parser()

    source_path = parser.source
    target_path = parser.target
    model_path = parser.model
    model_version = parser.model_version
    name = parser.video_name
    count = parser.count
    mode = parser.mode
    low_mem_limit = parser.low_mem_limit
    width = parser.width
    height = parser.height
    extension = parser.extension
    size = (width, height)

    assert count in [2, 4, 8, 16, 32, 64], "Count has to be one of 2, 4, 8, 16, 32, 64"
    assert mode in ['fast', 'low_mem'], "Mode has to be either 'fast' or 'low_mem'"
    assert extension in ['mp4', 'avi'], "Extension is expected to be one of the following: 'mp4', 'avi'"
    assert model_version in ['v1', 'v2', 'v3'], "The model version has to be one of [v1, v2, v3]"
    assert low_mem_limit > 0, "Low mem limit cannot be negative"

    if extension == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif extension == "avi":
        fourcc = cv2.VideoWriter_fourcc('I','4','2','0')


    if model_version in ['v1', 'v2', 'v3']:
        from tensorflow.data_generator import preprocess

    if model_version == "v1":
        import tensorflow.model_v1.modules as modules
    elif model_version == "v2":
        import tensorflow.model_v2.modules as modules
    elif model_version == "v3":
        import tensorflow.model_v3.modules as modules

    model = modules.load_model(model_path)
    if mode == 'fast':
        fast_boosting(
            source_path, 
            os.path.join(target_path, f'{name}.{extension}'), 
            os.path.join(target_path, f'{name}_comparision.{extension}'), 
            model, 
            count, 
            size, 
            fourcc
            )
    elif mode == 'low_mem':
        low_mem_boosting(
            source_path, 
            os.path.join(target_path, f'{name}.{extension}'), 
            os.path.join(target_path, f'{name}_comparision.{extension}'), 
            model, 
            count, 
            size, 
            fourcc,
            low_mem_limit
            )
