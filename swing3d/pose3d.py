import json
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

import swing3d.constants
import swing3d.utils


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(
        self,
        num_joints_in,
        in_features,
        num_joints_out,
        filter_widths,
        causal,
        dropout,
        channels,
    ):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, "Only odd filter widths are supported"

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)

        return x


class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(
        self,
        num_joints_in,
        in_features,
        num_joints_out,
        filter_widths,
        causal=False,
        dropout=0.25,
        channels=1024,
        dense=False,
    ):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(
            num_joints_in,
            in_features,
            num_joints_out,
            filter_widths,
            causal,
            dropout,
            channels,
        )

        self.expand_conv = nn.Conv1d(
            num_joints_in * in_features, channels, filter_widths[0], bias=False
        )

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append(
                (filter_widths[i] // 2 * next_dilation) if causal else 0
            )

            layers_conv.append(
                nn.Conv1d(
                    channels,
                    channels,
                    filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                    dilation=next_dilation if not dense else 1,
                    bias=False,
                )
            )
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(
                self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x)))
            )

        x = self.shrink(x)
        return x


def normalize_screen_coordinates(X, w, h):
    logger.info("Normalizing screen coordinates")
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def evaluate(
    inputs: Union[np.ndarray, torch.Tensor], model: TemporalModel
) -> np.ndarray:
    with torch.no_grad():
        model.eval()
        if isinstance(inputs, np.ndarray):
            inputs_torch = torch.from_numpy(inputs.astype("float32"))
        else:
            inputs_torch = inputs

        if torch.cuda.is_available():
            inputs_torch = inputs_torch.cuda()

        predicted_3d_pos = model(inputs_torch)

    return predicted_3d_pos.squeeze(0).cpu().numpy()


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec.double(), v.double(), dim=len(q.shape) - 1)
    uuv = torch.cross(qvec.double(), uv.double(), dim=len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)


def camera_to_world(X, R, t):
    logger.info(f"Performing camera to world with orientation of {R}")
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def calculate_padding(model: TemporalModel) -> int:
    receptive_field = model.receptive_field()
    logger.info(f"Receptive field: {receptive_field} frames")
    pad = (receptive_field - 1) // 2  # Padding on each side
    logger.info(f"Padding: {pad} frames")
    return pad


def pad_frames(keypoints: np.ndarray, padding: int) -> np.ndarray:
    logger.info(f"Padding first axis with {padding} frames")
    return np.expand_dims(
        np.pad(keypoints, ((padding, padding), (0, 0), (0, 0)), "edge"), axis=0
    )


def load_model(num_joints_in: int, in_features: int) -> TemporalModel:
    logger.info(
        f"Initialising temporal model with num_joints_in={num_joints_in} and in_features={in_features}"
    )
    model_pos = swing3d.pose3d.TemporalModel(
        num_joints_in=num_joints_in,
        in_features=in_features,
        num_joints_out=swing3d.constants.NUM_JOINTS,
        filter_widths=swing3d.constants.ARCHITECTURE,
        causal=swing3d.constants.CAUSAL,
        dropout=swing3d.constants.DROPOUT,
        channels=swing3d.constants.CHANNELS,
        dense=swing3d.constants.DENSE,
    )

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    logger.info(f"Trainable parameter count: {model_params}")

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    logger.info(f"Loading checkpoint {swing3d.constants.CHECKPOINT}")
    checkpoint = torch.load(
        swing3d.constants.CHECKPOINT, map_location=lambda storage, loc: storage
    )
    logger.info(f"This model was trained for {checkpoint['epoch']} epochs")
    model_pos.load_state_dict(checkpoint["model_pos"])

    return model_pos


def pose3d(
    keypoints: np.ndarray, resolution: tuple, output_path: Optional[str] = None
) -> np.ndarray:

    height, width = resolution
    logger.info(f"Resolution - height={height} width={width}")

    # Normalize camera frame
    keypoints[..., :2] = swing3d.pose3d.normalize_screen_coordinates(
        keypoints[..., :2], w=resolution[1], h=resolution[0],
    )

    model = load_model(
        num_joints_in=keypoints.shape[-2], in_features=keypoints.shape[-1]
    )

    pad = calculate_padding(model)

    keypoints_padded = pad_frames(keypoints, pad)

    prediction = evaluate(keypoints_padded, model)

    prediction = camera_to_world(
        prediction, R=swing3d.constants.CAMERA_PARAMS["orientation"], t=0
    )

    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    prediction = swing3d.utils.rotate_about_z(prediction, np.pi / 2)
    prediction = swing3d.utils.rotate_about_y(prediction, np.pi)

    if output_path is not None:
        logger.info(f"Outputting 3D keypoints to {output_path}")
        with open(output_path, "w") as f:
            json.dump(prediction.tolist(), f)

    return prediction


if __name__ == "__main__":
    pass
