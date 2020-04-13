from itertools import zip_longest

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

import swing3d.constants


def grouper(n, iterable):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=None, *args)


def read_video(filename):
    cap = cv2.VideoCapture(filename)

    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame, k=3)
            # frames.append({"image": torch.from_numpy(np.moveaxis(frame, -1, 0).copy())})
            frames.append(frame)
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


def rotate_about_z(predictions, theta):
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],
         [0, 0, 1]])

    return np.matmul(predictions, rot_matrix)


def rotate_about_y(predictions, theta):
    rot_matrix = np.array(
        [[np.cos(theta), 0, -np.sin(theta)],
         [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])

    return np.matmul(predictions, rot_matrix)


def print_debug_banner():
    for l in swing3d.constants.DEBUG_BANNER:
        logger.info(l)