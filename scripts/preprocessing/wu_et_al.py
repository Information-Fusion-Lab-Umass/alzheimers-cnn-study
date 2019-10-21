"""Pre-processing script for Wu et al: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6288052/
"""
import os

import cv2
import numpy as np
import pandas as pd

from default_preprocessing_generator import default_preprocessing_generator
from utils import load_nii_file

DATASET_PATH = "/mnt/nfs/work1/mfiterau/ADNI_data/MPR__GradWarp__B1_Correction_N3"
OUTPUT_PATH = "/mnt/nfs/work1/mfiterau/ADNI_data/wu_et_al"
ADNIMERGE_PATH = "../../data/ADNIMERGE_relabeled.csv"

if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

columns = ["PTID", "VISCODE", "IMGPATH", "DX"]
image_records = pd.DataFrame(columns=columns)

mri_counter = 0

adni_merge = pd.read_csv(ADNIMERGE_PATH)

for record in default_preprocessing_generator(DATASET_PATH, adni_merge):
    patient_id, visit_code, nii_file, dx = record

    # "According to whether the MCI subjects were converted into AD within 3 years, they were categorized
    # into sMCI (keep stable) and cMCI (converted into AD) groups."
    if visit_code == "bl" or visit_code == "sc":
        ptid_match = adni_merge["PTID"] == patient_id
        visit_code_match = adni_merge["VISCODE"] == "m36"  # find the third year DX
        third_year_match = adni_merge[ptid_match & visit_code_match]

        if third_year_match.empty:
            print(f"No third-year match found for {patient_id}, skipping...")
            continue
        else:
            if dx == "MCI" or dx == "EMCI" or dx == "LMCI":
                dx = "cMCI" if third_year_match.iloc[0]["DX"] == "AD" else "sMCI"
            elif dx == "AD":
                print("Patient is AD, skipping...")
                continue
            else:
                pass  # CN stay as is
    else:
        print("Image is not baseline, skipping...")
        continue

    print(f"Processing {visit_code} MRI with DX {dx}.")

    image = load_nii_file(nii_file)

    if image is None:
        continue

    # "Then, from among about 160 slices of raw MR scans of each subject, we discarded the first and last 15 slices
    # without anatomical information, resulting in about 130 slices for each subject."
    sliced_image = image[:, :, 15:-15]  # input is 192x192x160, output is 192x192x130

    # "Next, we selected 48 different slices randomly from the remaining slices with the interval of 4, and thus
    # generated 16 RGB color images for each subject."
    # NOTE: 4 * 48 = 192 which is way more than the 130 slices available. Even with 3 * 48 = 144, more slices are
    # needed than there are available. We interpret this as randomly picking a slice, then picking two other slices
    # 4 slices away to create the three channels.
    all_slice_idx = np.arange(sliced_image.shape[2])
    # remove the first and last five indices because we cannot generate neighboring slices four away
    valid_slice_idx = all_slice_idx[5:-5]

    num_images = 16

    # select the index of the middle slice, and two slices sandwiching the middle slice
    chosen_slices_idx = np.random.choice(valid_slice_idx, size=num_images, replace=False)
    before_slices_idx = chosen_slices_idx - 5
    after_slices_idx = chosen_slices_idx + 5

    chosen_slices = sliced_image[:, :, chosen_slices_idx]
    before_slices = sliced_image[:, :, before_slices_idx]
    after_slices_idx = sliced_image[:, :, after_slices_idx]

    for idx in range(num_images):
        composite_image = np.stack([before_slices[:, :, idx], chosen_slices[:, :, idx], chosen_slices[:, :, idx]],
                                   axis=2)

        # "Finally, all of the RGB color images were resized to 256×256 pixels."
        # Since no interpolation algorithm was specified, using default
        composite_image = cv2.resize(composite_image, dsize=(256, 256))

        image_path = f"{OUTPUT_PATH}/{patient_id}_{visit_code}_{dx}_slice_{idx}"  # numpy adds .npy automatically
        np.save(image_path, composite_image)
        image_records = image_records.append({
            "PTID": patient_id,
            "VISCODE": visit_code,
            "IMGPATH": f"{image_path}.npy",
            "DX": dx
        }, ignore_index=True)

    mri_counter += 1

    if mri_counter % 10 == 0:
        print(f"Processed {mri_counter} MRIs.")

image_records.to_csv(f"{OUTPUT_PATH}/manifest.csv", index=False)
print(f"Processed {mri_counter} MRIs.")
