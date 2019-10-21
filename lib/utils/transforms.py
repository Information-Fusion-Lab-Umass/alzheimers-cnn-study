from abc import ABC

import torch.nn.functional as F


class Transform(ABC):
    pass


class PadToSameDim(Transform):
    """Pad the image so the dimensions match.
    """

    def __init__(self, num_dim=3):
        self.num_dim = num_dim

    def __call__(self, image):
        shape = image.shape
        max_dim = max(shape)
        padding = [0] * (self.num_dim * 2)

        for idx, dim in enumerate(shape):
            if shape[idx] != max_dim:
                diff = max_dim - shape[idx]
                padding[2 * idx] = diff // 2
                padding[2 * idx + 1] = diff // 2

        flipped_padding = list(reversed(padding))
        return F.pad(image, flipped_padding)


class NaNToNum(Transform):
    """Replace nan in Tensor with a constant.
    """

    def __init__(self, constant=None):
        self.constant = constant

    def __call__(self, image):
        if self.constant is not None:
            image[image != image] = self.constant
        else:
            image[image != image] = float("inf")
            image[image == float("inf")] = image.min()

        return image


class RangeNormalize(Transform):
    """Normalize the pixel values to between 0 and 1.
    """

    def __call__(self, image):
        shifted = image - image.min()
        return shifted / shifted.max()


class MeanStdNormalize(Transform):
    """Normalize the pixel values to between -1 and 1 by subtracting the mean and dividing by the standard deviation.
    """

    def __call__(self, image):
        return (image - image.mean()) / image.std()
