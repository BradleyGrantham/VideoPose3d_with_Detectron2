import cv2
import numpy as np


def read_video(filename, rot=True):
    cap = cv2.VideoCapture(filename)

    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rot:
                frame = np.rot90(frame, k=3)
            # frames.append({"image": torch.from_numpy(np.moveaxis(frame, -1, 0).copy())})
            frames.append(frame)
        else:
            break
    cap.release()

    return frames
