'''Script to generate label mapping for normalized images.

/mnt/nfs/work1/mfiterau/riteshkumar/processed_segmentation_data/caps_tissue/subjects/sub-ADNI[SUBJECT_ID]/ses-[VISCODE]/t1/spm/segmentation
    -> dartel_input
    -> native_space
    -> normalized_space
'''
import os
import pickle
import multiprocessing as mp
import pandas as pd

from fnmatch import fnmatch
from argparse import ArgumentParser
from dir import lsdir
from pdb import set_trace

class FileMapping:
    EXTENSION = "*.nii.gz"
    FILE_COLUMNS = ["patient_id", "visit_code", "csf_path", "gray_matter_path", "white_matter_path", "forward_deform", "inverse_deform", "misc"]

    def __init__(self, **kwargs):
        self.img_path = kwargs.get("img_path")
        self.max_proc = kwargs.get("max_proc")
        self.feat_path = kwargs.get("feat_path")
        self.save_path = kwargs.get("save_path")

        self.features = pd.read_csv(self.feat_path)
        self.mapping = None

    def load_mapping(self):
        if self.mapping is not None:
            return self.mapping
        else:
            self.mapping = self._get_paths(self.img_path)

        self._add_metadata(self.mapping)

        return self.mapping

    def save_mapping(self):
        if self.mapping is None:
            print("No mapping detected, generating.")
            self.load_mapping()

        print("Saving mapping to file.")
        self._save_df(self.mapping, self.save_path)

    def _get_paths(self, root):
        '''Generates and return a pandas.DataFrame containing all of the subjects with their respective image paths.
        '''
        subjects = lsdir(root)

        if self.max_proc > 1:
            num_proc = min(max(1, mp.cpu_count() - 2), self.max_proc)

            with mp.Pool(num_proc) as pool:
                paths = pool.map(self._get_path, subjects)
        else:
            paths = list(map(self._get_path, subjects))

        return pd.concat(paths)

    def _get_path(self, subject_dir):
        '''Returns one pandas.DataFrame for the give subject's directory.
        '''
        # split "sub-ADNI127S0259" into "sub-" and "127S0259"
        subject_id = subject_dir.split("ADNI")[1]
        subject_path = self.img_path + "/" + subject_dir
        data = []

        # there may be multiple sessions for each patient
        sessions = lsdir(subject_path)

        for sess in sessions:
            # split "ses-M48" into "ses" and "M48"
            visit_code = sess.split("-")[1]
            sess_path = subject_path + "/" + sess

            for path, subdirs, files in os.walk(sess_path):
                # should be in the normalized_space/ dir
                if not path.split("/")[-1] == "normalized_space":
                    continue

                csf_path = None # ~500KB
                gray_matter_path = None # ~400KB
                white_matter_path = None # ~300KB
                forward_deform = None # ~22MB
                inverse_deform = None # ~112MB
                # Not sure what to call this, example file name is
                # sub-ADNI005S0222_ses-M18_space-Ixi549Space_T1w.nii.gz
                misc = None

                for file_name in files:
                    # should end in .nii.gz
                    if not fnmatch(file_name, self.EXTENSION):
                        continue

                    fn_parts = file_name.split("_")
                    file_path = "{}/{}".format(path, file_name)
                    if fn_parts[2] == "space-Ixi549Space":
                        misc = file_path
                    elif fn_parts[3] == "segm-csf":
                        csf_path = file_path
                    elif fn_parts[3] == "segm-graymatter":
                        gray_matter_path = file_path
                    elif fn_parts[3] == "segm-whitematter":
                        white_matter_path = file_path
                    elif fn_parts[4] == "transformation-inverse":
                        inverse_deform = file_path
                    elif fn_parts[4] == "transformation-forward":
                        forward_deform = file_path
                    else:
                        raise Exception("Unrecognized file name: {}"
                                            .format(file_name))

                # ORDER SHOULD MATCH self.columns
                data.append([subject_id, visit_code, csf_path, gray_matter_path,
                             white_matter_path, forward_deform, inverse_deform,
                             misc])

        df = pd.DataFrame(data, columns=self.FILE_COLUMNS)

        return df

    def _add_metadata(self, df):
        # create new columns for metadata
        df["label"] = None

        feats = self.features

        for idx, row in df.iterrows():
            # "127S0259" => "127_S_0259"
            patient_id = "_S_".join(row["patient_id"].split("S"))
            visit_code = row["visit_code"].lower()

            # "bl" in image path, m00 in ADNIMERGE
            visit_code = "bl" if visit_code == "m00" else visit_code

            ptid_match = feats["PTID"] == patient_id
            vc_match = feats["VISCODE"] == visit_code
            match = feats[ptid_match & vc_match]

            if len(match.index) != 1:
                raise Exception("Not exactly one match. Sumthin' not right.")

            row["label"] = match["DX"].values[0]

    def _save_df(self, df, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(df, file)

if __name__ == "__main__":
    parser = ArgumentParser()

    # The level below this path should be all of the subjects with IDs
    parser.add_argument("--img_path", type=str,
                        default="/mnt/nfs/work1/mfiterau/zguan/clinica_data/subjects")
    parser.add_argument("--max_proc", type=int, default=12)
    parser.add_argument("--feat_path", type=str,
                        default="../outputs/ADNIMERGE.csv")
    parser.add_argument("--save_path", type=str,
                        default="../outputs/normalized_mapping.pickle")

    args = parser.parse_args()
    mapping_object = FileMapping(img_path=args.img_path,
                                 max_proc=args.max_proc,
                                 feat_path=args.feat_path,
                                 save_path=args.save_path)

    mapping_object.save_mapping()
