import traceback
from typing import Optional

import nibabel as nib
import numpy as np


def load_nii_image(image_path: str) -> Optional[np.ndarray]:
    image = None

    try:
        image = nib.load(image_path).get_fdata().squeeze()
    except Exception as e:
        print(f"Failed to load {image_path}")
        print(f"Errors encountered: {e}")
        print(traceback.format_exc())

    return image


def load_npy_image(image_path: str) -> Optional[np.ndarray]:
    return np.load(image_path)
