import cv2
import numpy as np


def read_video(filename, rot=True):
    cap = cv2.VideoCapture(filename)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

    fc = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rot:
                frame = np.rot90(frame, k=3)
            # frames.append({"image": torch.from_numpy(np.moveaxis(frame, -1, 0).copy())})
            frames[fc, :, :, :] = frame
            fc += 1
        else:
            break
    cap.release()

    return frames
