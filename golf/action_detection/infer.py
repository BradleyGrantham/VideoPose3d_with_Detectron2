import numpy as np
import torch
from PIL import Image


import golf.action_detection.mobilenetv2
import golf.io


def classify_image(model, image):
    """
    Classify an image as a victory royale or not.

    Parameters
    ----------
    model : MobileNetV2
    image : np.array

    Returns
    -------
    int : 0-8

    """
    frame = golf.action_detection.mobilenetv2.data_transforms(image)
    frame = frame.unsqueeze(0)
    frame = frame.to(golf.action_detection.mobilenetv2.DEVICE)
    outputs = model(frame)
    _, preds = torch.max(outputs, 1)
    return preds.item()


def load_mobilenet(path):
    model, _, _, _ = golf.action_detection.mobilenetv2.init_mobilenet(pretrained=False)
    model.load_state_dict(
        torch.load(path, map_location=golf.action_detection.mobilenetv2.DEVICE)
    )
    model.eval()
    return model


def infer_video(video_path, model_path):
    frames = golf.io.read_video(video_path, rot=False)
    model = load_mobilenet(model_path)

    for frame in frames:
        im = Image.fromarray(np.squeeze(frame))
        cls = classify_image(model, im)
        print(cls)


if __name__ == "__main__":
    infer_video("../../videos/IMG_2909.MOV", "../../models/mobilenet_837ccf8c96ee4cdd945619874ed03738.pt")