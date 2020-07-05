"""Lots of this code taken from
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html."""
import copy
import glob
import os
import time
import uuid

import numpy as np
import torch
from loguru import logger
from torchvision import datasets

import golf.action_detection.mobilenetv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../../data/golfDB/PyTorchImageFolder/"


TEST_SIZE = 40 * -1
BATCH_SIZE = 2
NUM_WORKERS = 4
NUM_EPOCHS = 20


def train_model(
    model, data_train, data_test, criterion, optimizer, scheduler, num_epochs=20
):
    model_id = uuid.uuid4().hex
    logger.info(f"Model id is {model_id}")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase, dataloader in zip(["train", "val"], [data_train, data_test]):
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / (len(dataloader) * dataloader.batch_size)
            epoch_acc = running_corrects.double() / (
                len(dataloader) * dataloader.batch_size
            )

            logger.info(
                "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
            )

            # deep copy the model if best accuracy
            if phase == "val" and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                logger.info(f"Saving model as mobilenet_{model_id}.pt")
                torch.save(model.state_dict(), f"../../models/mobilenet_{model_id}.pt")

    time_elapsed = time.time() - since
    logger.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logger.info("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def init_dataloaders(data_dir):
    dataset = datasets.ImageFolder(
        os.path.join(data_dir), golf.action_detection.mobilenetv2.data_transforms
    )
    class_names = dataset.classes

    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:TEST_SIZE])
    dataset_test = torch.utils.data.Subset(dataset, indices[TEST_SIZE:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return data_loader, data_loader_test, class_names


if __name__ == "__main__":

    dataloader_train, dataloader_test, class_names = init_dataloaders(DATA_DIR)

    mobilenet, crit, optim, sched = golf.action_detection.mobilenetv2.init_mobilenet()

    mobilenet = train_model(
        mobilenet,
        dataloader_train,
        dataloader_train,
        crit,
        optim,
        sched,
        num_epochs=NUM_EPOCHS,
    )
