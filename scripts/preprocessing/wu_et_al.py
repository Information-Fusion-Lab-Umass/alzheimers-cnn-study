"""Pre-processing script for Wu et al: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6288052/
"""
import os

import numpy as np
import pandas as pd

from default_preprocessing_generator import default_preprocessing_generator
from utils import load_nii_file

MRI_LIST_PATH = "../../data/MRILIST.csv"
VISITS_PATH = "../../data/VISITS.csv"
ADNIMERGE_PATH = "../../data/ADNIMERGE_relabeled.csv"
DATASET_PATH = "/mnt/nfs/work1/mfiterau/ADNI_data/MPR__GradWarp__B1_Correction_N3"
OUTPUT_PATH = "/mnt/nfs/work1/mfiterau/ADNI_data/wu_et_al"

if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

columns = ["PTID", "VISCODE", "IMGPATH", "DX"]
image_records = pd.DataFrame(columns=columns)

mri_counter = 0

for record in default_preprocessing_generator(DATASET_PATH, ADNIMERGE_PATH, MRI_LIST_PATH, VISITS_PATH):
    patient_id, visit_code, nii_file, dx = record
    image = load_nii_file(nii_file)

    if image is None:
        continue

    # pre-processing steps per paper

    # (1) "Then, from among about 160 slices of raw MR scans of each subject, we discarded the first and last 15 slices
    # without anatomical information, resulting in about 130 slices for each subject."
    sliced_image = image[:, :, 15:-15]  # input is 192x192x160, output is 192x192x130

    # (2) "Next, we selected 48 different slices randomly from the remaining slices with the interval of 4, and thus
    # generated 16 RGB color images for each subject."
    # NOTE: 4 * 48 = 192 which is way more than the 130 slices available. Even with 3 * 48 = 144, more slices are
    # needed than there are available. We interpret this as randomly picking a slice, then picking two other slices
    # 4 slices away to create the three channels.
    all_slice_idx = np.arange(sliced_image.shape[2])
    # remove the first and last five indices because we cannot generate neighboring slices four away
    valid_slice_idx = all_slice_idx[5:-5]

    num_images = 16
    chosen_slices_idx = np.random.choice(valid_slice_idx, size=num_images, replace=False)
    before_slices_idx = chosen_slices_idx - 5
    after_slices_idx = chosen_slices_idx + 5

    chosen_slices = sliced_image[:, :, chosen_slices_idx]
    before_slices = sliced_image[:, :, before_slices_idx]
    after_slices_idx = sliced_image[:, :, after_slices_idx]

    for idx in range(num_images):
        composite_image = np.stack([before_slices[:, :, idx], chosen_slices[:, :, idx], chosen_slices[:, :, idx]],
                                   axis=2)
        image_path = f"{OUTPUT_PATH}/{patient_id}_{visit_code}_{dx}_slice_{idx}"  # numpy adds .npy
        np.save(image_path, composite_image)
        image_records = image_records.append({
            "PTID": patient_id,
            "VISCODE": visit_code,
            "IMGPATH": image_path,
            "DX": dx
        }, ignore_index=True)

    mri_counter += 1

    if mri_counter % 10 == 0:
        print(f"Processed {mri_counter} MRIs.")

image_records.to_csv(f"{OUTPUT_PATH}/manifest.xml")
print(f"Processed {mri_counter} MRIs.")
