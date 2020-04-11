from itertools import zip_longest

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def grouper(n, iterable):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=None, *args)


def read_video(filename):
    cap = cv2.VideoCapture(filename)

    frames = []
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame, k=3)
        if ret:
            frames.append({"image": torch.from_numpy(np.moveaxis(frame, -1, 0))})
        else:
            break
    cap.release()

    return frames


def plot_image_and_keypoints(image, keypoints, output_path=None):
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1])
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
