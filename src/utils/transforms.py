import torch.nn.functional as F

from pdb import set_trace

class OrientFSImage(object):
    '''
    An object used by Dataset class to orient the FreeSurfer MRI images so they match those of the preprocessed images.
    '''
    def __init__(self):
        pass

    def __call__(self, image):
        # Flipping the coronal (dim=1) and axial (dim=2) plane.
        return image.flip(1).flip(2)

class PadPreprocImage(object):
    '''
    Pads the image on both sides so it becomes a cube of size target_size*target_size*target_size.
    '''
    def __init__(self, target_size=256, num_dim=3):
        self.target_size = target_size
        self.num_dim = num_dim

    def __call__(self, image):
        dim = tuple(image.shape)
        pad_amount = [ 0 ] * (self.num_dim * 2)

        for idx in range(len(dim)):
            if dim[idx] < self.target_size:
                padding = (self.target_size - dim[idx]) // 2
                pad_amount[idx*2] = padding
                pad_amount[idx*2+1] = padding

        pad_params = { "mode": "constant", "value": 0 }
        padded_image = F.pad(image,
                             tuple(reversed(pad_amount)),
                             **pad_params)

        return padded_image

class RangeNormalization(object):
    '''
    Normalize the pixel values to between 0 and 1.
    '''
    def __init__(self):
        pass

    def __call__(self, image):
        shifted = image - image.min()
        return shifted / shifted.max()

class MeanStdNormalization(object):
    '''Normalize the pixel values to between -1 and 1 by subtracting the mean and dividing by the standard deviation.
    '''
    def __init__(self):
        pass

    def __call__(self, image):
        return (image - image.mean()) / image.std()

class NaNToNum(object):
    '''Replace nan in Tensor with a constant.
    '''
    def __init__(self, constant=None):
        self.constant = constant

    def __call__(self, image):
        if self.constant is not None:
            image[image != image] = self.constant
        else:
            image[image != image] = float("inf")
            image[image == float("inf")] = image.min()

        return image

class PadToSameDim(object):
    '''Pad the image so the dimensions match.
    '''
    def __init__(self, num_dim=3):
        self.num_dim = num_dim

    def __call__(self, image):
        shape = image.shape
        max_dim = max(shape)
        padding = [ 0 ] * (self.num_dim * 2)

        for idx, dim in enumerate(shape):
            if shape[idx] != max_dim:
                diff = max_dim - shape[idx]
                padding[2 * idx] = diff // 2
                padding[2 * idx + 1] = diff // 2

        flipped_padding = tuple(reversed(padding))
        return F.pad(image, flipped_padding)
