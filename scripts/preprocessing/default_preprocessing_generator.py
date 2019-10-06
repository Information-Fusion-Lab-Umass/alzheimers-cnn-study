import pandas as pd

from utils import safe_listdir

MRI_LIST_PATH = "../../data/MRILIST.csv"
VISITS_PATH = "../../data/VISITS.csv"


def default_preprocessing_generator(dataset_path, adni_merge):
    """This function provides a generator to iterate through all of the images from the ADNI dataset with the
    "MPR", "GradWarp", "B1 Correction", "N3 Scaled" pre-processing steps applied. This generator assumes the following
    data file structure.

    dataset_path
        | - subset 0
            | - patient_id_1
                |- MPR__GradWarp__B1_Correction__N3
                    |- visit_date_1
                        |- image_id
                            |- *.nii files
                    |- visit_date_2
                        |- visit_code
                            |- *.nii files
            | - patient_id_2
                |- MPR__GradWarp__B1_Correction__N3
                    |- ...
            ...
        | - subset 0
            |- ...
        | - subset 0
            |- ...
        | - subset 0
        ...
    """
    sub_folders = safe_listdir(dataset_path)
    mri_list = pd.read_csv(MRI_LIST_PATH,
                           dtype={"STUDYID": "Int64", "SERIESID": "Int64", "IMAGEUID": "Int64"})
    visit_metadata = pd.read_csv(VISITS_PATH, dtype={"VISORDER": "Int64"})

    for folder in sub_folders:
        patient_folder = f"{dataset_path}/{folder}"
        patient_ids = safe_listdir(patient_folder)

        for patient_id in patient_ids:
            visit_dates_folder = f"{patient_folder}/{patient_id}/MPR__GradWarp__B1_Correction__N3"
            visit_dates = safe_listdir(visit_dates_folder)

            for visit_date in visit_dates:
                image_folder = f"{visit_dates_folder}/{visit_date}"
                series_ids = safe_listdir(image_folder)

                for series_id in series_ids:
                    data_files_folder = f"{image_folder}/{series_id}"
                    files = safe_listdir(data_files_folder)
                    files_with_nii_extension = list(filter(lambda x: x[-3:] == "nii", files))

                    if len(files_with_nii_extension) != 1:
                        print(f"There are {len(files_with_nii_extension)} files with .nii extension, expecting 1.")
                        continue
                    else:
                        nii_file = f"{data_files_folder}/{files_with_nii_extension[0]}"

                        # Matching the image metadata with MRILIST.csv
                        subject_match = mri_list["SUBJECT"] == patient_id
                        series_id_match = mri_list["SERIESID"] == int(series_id.split("S")[1])
                        mri_record = mri_list[subject_match & series_id_match]

                        if mri_record.empty or len(mri_record) > 1:
                            print(f"There are {len(mri_record)} records in MRILIST.csv for {patient_id} {series_id}, "
                                  f"skipped...")
                            continue

                        visit_name = mri_record["VISIT"].iloc[0]

                        # "ADNI *" => "*"
                        if visit_name[:5] == "ADNI ":
                            visname = visit_name.split(" ")[1]
                        elif visit_name[:9] == "ADNI1/GO ":
                            visname = " ".join(visit_name.split(" ")[1:])
                        else:
                            visname = visit_name

                        visit_code = visit_metadata[visit_metadata["VISNAME"] == visname]["VISCODE"].iloc[0]

                        # https://adni.loni.usc.edu/wp-content/uploads/2008/07/adni2-procedures-manual.pdf page 37
                        # The window from Screening to Baseline is 28 days
                        is_screening = visit_code == "sc"

                        ptid_match = adni_merge["PTID"] == patient_id
                        viscode_match = adni_merge["VISCODE"] == (visit_code if not is_screening else "bl")
                        adni_merge_record = adni_merge[ptid_match & viscode_match]

                        if adni_merge_record.empty:
                            print(f"There are no {patient_id} with {visit_code} ({visit_name}, {series_id}) in "
                                  f"ADNIMERGE, skipping...")
                            continue

                    if len(adni_merge_record) > 1:
                        print(f"There are more than 1 records ({len(adni_merge_record)}) for {patient_id} with "
                              f"{visit_code} in ADNIMERGE, skipping...")

                    dx = adni_merge_record["DX"].iloc[0]

                    yield patient_id, visit_code, nii_file, dx
