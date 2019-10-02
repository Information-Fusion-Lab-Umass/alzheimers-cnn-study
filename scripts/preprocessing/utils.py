import traceback
from os import listdir
from typing import Optional

import nibabel as nib
import numpy as np


def safe_listdir(dir_path):
	try:
		return listdir(dir_path)
	except FileNotFoundError as _:
		return []


def load_nii_file(file_path: str) -> Optional[np.ndarray]:
	image = None

	try:
		image = nib.load(file_path).get_fdata().squeeze()
	except Exception as e:
		print(f"Failed to load {file_path}")
		print(f"Errors encountered: {e}")
		print(traceback.format_exc())

	return image
