import os
import pickle
from argparse import ArgumentParser
from itertools import chain
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd

'''Script to generate label mapping for normalized images.

NOTE: The ADNIMERGE.csv file is downloaded from the ADNI website. The original CSV file has a DX column (labels) with missing values, the ADNIMERGE.csv included in this repository contains imputed values based on previous visits and DX_base.
'''

NUM_CPU = os.cpu_count()
# extension of the MRI scan files
FILE_EXTENSION = "*.nii.gz"
OUTPUT_FORMATS = ["json", "pickle"]


def lsdir(path):
    '''Lists all of the files in the specified path.

    NOTE: This excludes files that start with a ".".
    '''
    files = os.listdir(path)
    return list(filter(lambda name: name[0] != ".", files))


def get_image_paths(root, **kwargs):
    '''Generates and return a pandas.DataFrame containing all of the subjects with their respective image paths.

    Example:
        An example directory path,

        path/to/clinica_procesed_data/tissue_segmentation/tissue_seg_0/subjects/sub-ADNI[SUBJECT_ID]/ses-[VISCODE]/t1/spm/segmentation
            -> dartel_input
            -> native_space
            -> normalized_space

        In this case, the root should be "path/to/clinica_processed_data/tissue_segmentation/".

    Args:
        root (string): The directory level below this path should contain folders that begin with either "sub-ADNI" or "tissue_seg_0".
    '''
    max_proc = kwargs.get("max_proc", NUM_CPU - 1)
    num_proc = min(max(1, NUM_CPU - 1), max_proc)

    paths = lsdir(root)
    # root variable cannot be accessed in list comprehension
    path_pairs = list(zip([root] * len(paths), paths))

    if "sub-ADNI" in paths[0]:  # root contains subjects
        subject_paths = [f"{r}/{sdir}" for r, sdir in path_pairs]
    else:  # root has "tissue_seg_0" to "tissue_seg_n"
        subset_paths = [f"{r}/{sdir}/subjects" for r, sdir in path_pairs]

        def list_subject_dirs(subset_path):
            subjects = lsdir(subset_path)
            return list(map(lambda spath: f"{subset_path}/{spath}", subjects))

        subject_paths = list(chain(*map(list_subject_dirs, subset_paths)))

    if max_proc <= 1:  # for debugging
        print("Fetching image paths with a single process.")
        dfs = [get_image_path(spath) for spath in subject_paths]
    else:
        print(f"Fetching image paths with {num_proc} processes.")
        with Pool(num_proc) as pool:
            dfs = pool.map(get_image_path, subject_paths)

    return pd.concat(dfs)


def get_image_path(subject_path):
    '''Returns one pandas.DataFrame for the given subject's directory.
    '''
    subject_id = subject_path.split("sub-ADNI")[1]
    sessions = lsdir(subject_path)

    data = []

    for sess in sessions:
        visit_code = sess.split("-")[1].lower()

        # walk through all of the subpaths
        for path, subdirs, files in os.walk(f"{subject_path}/{sess}"):
            # should be in the normalized_space/ dir
            if not path.split("/")[-1] == "normalized_space":
                continue

            record = {
                # "patient_id": subject_id,
                # "visit_code": visit_code,
                "PTID": subject_id,
                "VISCODE": visit_code,
                "csf_path": None,  # ~500KB
                "gray_matter_path": None,  # ~400KB
                "white_matter_path": None,  # ~300KB
                "forward_deform_path": None,  # ~22MB
                "inverse_deform_path": None,  # ~112MB
                "skull_intact_path": None
            }

            for file_name in files:
                file_name_parts = file_name.split("_")
                file_path = f"{path}/{file_name}"

                if file_name_parts[2] == "space-Ixi549Space":
                    record["skull_intact_path"] = file_path
                elif file_name_parts[3] == "segm-csf":
                    record["csf_path"] = file_path
                elif file_name_parts[3] == "segm-graymatter":
                    record["gray_matter_path"] = file_path
                elif file_name_parts[3] == "segm-whitematter":
                    record["white_matter_path"] = file_path
                elif file_name_parts[4] == "transformation-inverse":
                    record["inverse_deform_path"] = file_path
                elif file_name_parts[4] == "transformation-forward":
                    record["forward_deform_path"] = file_path
                else:
                    raise Exception(f"Unrecognized file format: {file_name}")

            # values might not be in order
            # data.append([*record.values()])
            data.append([record[key] for key in record.keys()])

    return pd.DataFrame(data, columns=record.keys())


def add_adni_merge(df, adni_merge_path):
    '''Add adni_merge
    '''
    adni_merge = pd.read_csv(adni_merge_path)

    # create new columns for metadata
    df["DX"] = None
    df["VISNUM"] = None
    df["AGE"] = None
    df["PTGENDER"] = None
    df["PTEDUCAT"] = None
    df["PTETHCAT"] = None
    df["PTRACCAT"] = None
    df["PTMARRY"] = None
    df["MMSE"] = None
    df["RAVLT_immediate"] = None
    df["RAVLT_forgetting"] = None
    df["ADAS11"] = None
    df["ADAS13"] = None
    df["APOE4"] = None
    df["CDRSB"] = None

    for idx, row in df.iterrows():
        # "127S0259" => "127_S_0259"
        # patient_id = "_S_".join(row["patient_id"].split("S"))
        patient_id = "_S_".join(row["PTID"].split("S"))
        # visit_code = row["visit_code"].lower()
        visit_code = row["VISCODE"]

        visit_code = "bl" if visit_code == "m00" else visit_code
        ptid_match = adni_merge["PTID"] == patient_id
        vc_match = adni_merge["VISCODE"] == visit_code
        match = adni_merge[ptid_match & vc_match]

        if len(match.index) != 1:
            raise Exception("Not exactly one match. Sumthin' not right.")

        row["DX"] = match["DX"].values[0]
        # row["VISNUM"] = (int(row["visit_code"][1:])//6)+1 # Nobody has an eighth visit it goes from M36 -> M48
        row["VISNUM"] = (int(row["VISCODE"][1:]) // 6)  # + 1

        row["AGE"] = match["AGE"].values[0]
        row["PTGENDER"] = match["PTGENDER"].values[0]
        row["PTEDUCAT"] = match["PTEDUCAT"].values[0]
        row["PTETHCAT"] = match["PTETHCAT"].values[0]
        row["PTRACCAT"] = match["PTRACCAT"].values[0]
        row["PTMARRY"] = match["PTMARRY"].values[0]
        row["MMSE"] = match["MMSE"].values[0]
        row["RAVLT_immediate"] = match["RAVLT_immediate"].values[0]
        row["RAVLT_forgetting"] = match["RAVLT_forgetting"].values[0]
        row["ADAS11"] = match["ADAS11"].values[0]
        row["ADAS13"] = match["ADAS13"].values[0]
        row["APOE4"] = match["APOE4"].values[0]
        row["CDRSB"] = match["CDRSB"].values[0]

    df.reset_index(drop=True, inplace=True)
    return df


def add_img_features(df, cnn_root_path, tadpole_path):
    # Load dataframes
    df_tadpole = pd.read_csv(tadpole_path)

    # Initialize column for image feature path
    df["image_feature_path"] = None

    # Initialize columns for tadpole features
    ucsf_names = []
    for name in df_tadpole.columns.values:
        if ('UCSFFSX' in name or 'UCSFFSL' in name):
            if (name.startswith('ST') and 'STATUS' not in name):
                ucsf_names.append(name)
                df[name] = None

    for idx in df.index.values:
        pid = df.loc[idx, "PTID"]
        pid_tadpole = f"{pid[:3]}_{pid[3]}_{pid[4:]}"

        visit = df.loc[idx, "VISCODE"]

        if visit == "m00":  # NOTE: originally M00, changed to lowercase
            visit_tadpole = "bl"
        else:
            visit_tadpole = f"m{visit[1:]}"  # TODO: Why not just visit?

        tadpole_match = df_tadpole[
            (df_tadpole['PTID'] == pid_tadpole) &
            (df_tadpole['VISCODE'] == visit_tadpole)]

        # tadpole features
        for name in ucsf_names:
            df.at[idx, name] = tadpole_match[name].values[0]

        # image features
        df.at[idx, "image_feature_path"] = \
            f"{cnn_root_path}/{pid}/{visit}/features.pckl"

    df.reset_index(drop=True, inplace=True)
    return df


def impute_nan(df, threshold=None):
    tadpole_cols = df.columns[df.dtypes.eq('object')]
    tadpole_cols = [col for col in tadpole_cols if 'UCSF' in col]
    misc_cols = [
        "CDRSB", "ADAS11", "ADAS13", "MMSE", "RAVLT_immediate", "RAVLT_forgetting", "AGE", "APOE4"
    ]

    # Convert tadpole column format from object to float
    df[tadpole_cols] = df[tadpole_cols].apply(pd.to_numeric, errors='coerce')
    df[misc_cols] = df[misc_cols].apply(pd.to_numeric, errors='coerce')

    # Impute values with previous value if previous value exists
    df.sort_values(by=['PTID', 'VISNUM'], inplace=True)
    df = df.groupby('PTID').ffill()

    # Fill in leftover NaN values with column mean
    values = tadpole_cols + \
             ['ADAS13', 'MMSE', 'ADAS11', 'RAVLT_immediate', 'RAVLT_forgetting', 'AGE',
              'CDRSB', 'APOE4', ]
    all_nan_cols = []
    for v in values:
        df[v].fillna(df[v].mean(), inplace=True)
        if np.sum(np.isnan(df[v].values)) > 0:
            all_nan_cols.append(v)
    df = df.drop(all_nan_cols, axis=1)

    # Drop columns that have an amount of NaN values above the threshold
    if threshold:
        for col in tadpole_cols:
            num_missing = df[col].isna().sum()

            if num_missing > threshold - 1:
                try:
                    df.drop(col, axis=1, inplace=True)
                except KeyError:
                    print("KeyError, cannot drop column: ", col)

    df.reset_index(drop=True, inplace=True)

    return df


def main(args):
    start_time = time()
    print("Starting get_image_paths")
    image_df = get_image_paths(args.scans_path, max_proc=args.max_proc)
    result_df = image_df
    print(f"Finished get_image_paths in {round(time() - start_time)} seconds.")

    start_time = time()
    print("Starting add_adni_merge")
    adni_merge_df = add_adni_merge(image_df, args.adnimerge_path)
    result_df = adni_merge_df
    print(f"Finished add_adni_merge in {round(time() - start_time)} seconds.")

    skip_image_features = args.cnn_path is None or args.tadpole_path is None
    if skip_image_features:
        print("Skipping image features.")
    else:
        start_time = time()
        print("Starting add_img_features")
        tadpole_features_df = add_img_features(adni_merge_df,
                                               args.cnn_path,
                                               args.tadpole_path)
        result_df = tadpole_features_df
        print(f"Finished add_img_features in {round(time() - start_time)} seconds.")

    formats = args.output_format.split(",")
    for f in formats:
        assert f in OUTPUT_FORMATS, f"Unrecognized format: {f}."

    if "json" in formats:
        result_df.to_json(f"{args.output_path}.json")

    if "pickle" in formats:
        with open(f"{args.output_path}.pickle", "wb") as file:
            pickle.dump(result_df, file)


if __name__ == "__main__":
    '''
    Script mode
    python3 mapping.py --output_format="pickle" --run_main --max_proc=12

    Interactive mode
    python3 -i mapping.py
    '''
    parser = ArgumentParser()

    parser.add_argument("--scans_path",
                        default="path/to/clinica_data/subjects",
                        type=str,
                        help=f"Root folder for the subject images/scans.")

    parser.add_argument("--adnimerge_path",
                        default="../files/ADNIMERGE_relabeled.csv",
                        type=str,
                        help=f"Path to the ADNIMERGE.csv file.")

    parser.add_argument("--tadpole_path",
                        default=None,
                        type=str,
                        help=f"Path to the tadpole_features.csv file. Optional.")

    parser.add_argument("--cnn_path",
                        default=None,
                        type=str,
                        help=f"Root of the directory that contains the CNN features. Optional.")

    parser.add_argument("--output_path",
                        default="../data/data_mapping",
                        type=str,
                        help=f"Path of the output file.")

    parser.add_argument("--output_format",
                        default="json,pickle",
                        type=str,
                        help=f"JSON or pickle format, delimited by comma.")

    parser.add_argument("--run_main",
                        default=False,
                        action="store_true",
                        help=f"Whether to run the main method or not. Main does not run impute_nan.")

    parser.add_argument("--max_proc",
                        default=NUM_CPU - 1,
                        type=int,
                        help=f"Maximum number of parallel processes to spawn, set to #CPU - 1 by default. Set to 1 for debugging.")

    args = parser.parse_args()

    if args.run_main:
        main(args)
    else:
        with open(f"{args.output_path}.pickle", "rb") as file:
            df = pickle.load(file)
        tadpole_cols = [col for col in df.columns if "UCSF" in col]

        df_imputed = impute_nan(df.copy(deep=True))

        df_imputed.to_json(f"ultimate_final_imputed_mapping.json")
        with open(f"ultimate_final_imputed_mapping.pickle", "wb") as file:
            pickle.dump(df_imputed, file)
