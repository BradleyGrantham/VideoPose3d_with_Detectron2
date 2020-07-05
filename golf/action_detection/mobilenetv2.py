import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose(
    [
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def init_mobilenet(num_classes=9, pretrained=True):
    """Initialize a MobileNetV2 for transfer learning."""
    model_ft = models.mobilenet_v2(pretrained=pretrained)

    num_ftrs = 1280  # taken from printing the model

    model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler


def classify_image(model, image):
    """
    Classify an image as a victory royale or not.

    Parameters
    ----------
    model : MobileNetV2
    image : np.array

    Returns
    -------
    int : 1 for victory_royale, 0 otherwise

    """
    # frame = data_transforms(image)
    frame = image.unsqueeze(0)
    outputs = model(frame)
    _, preds = torch.max(outputs, 1)
    return preds.item()


def load_mobilenet(path):
    model, _, _, _ = init_mobilenet(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model
