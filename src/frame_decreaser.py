import cv2

def process(source, target):
    fourcc = cv2.VideoWriter_fourcc('I','4','2','0')

    vidcap = cv2.VideoCapture(source)
    success, frame = vidcap.read()
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(target, fourcc, fps/4.0, (frame.shape[1], frame.shape[0]))

    index = 0
    while success:
        if index % 4 == 0:
            out.write(frame)
            # cv2.imshow("sad", frame)
            # cv2.waitKey(100)
        success, frame = vidcap.read()
        index += 1


if __name__ == "__main__":
    process('/Users/kamil/Desktop/frame_booster_results/home/home.mov', '/Users/kamil/Desktop/frame_booster_results/home/home_slow.avi')