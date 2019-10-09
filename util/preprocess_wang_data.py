import os
import pandas as pd


ADNI_merge = pd.read_csv("/mnt/nfs/work1/mfiterau/yfung/MRI/src/outputs/ADNIMERGE.csv", header=0)
MRI_list = pd.read_csv("/mnt/nfs/work1/mfiterau/zguan/alzheimers-cnn-study/data/MRILIST.csv", header=0)
VISITS_list = pd.read_csv("/mnt/nfs/work1/mfiterau/zguan/alzheimers-cnn-study/data/VISITS.csv", header=0)

subdir_num = "0"
source_MRI = "/mnt/nfs/work1/mfiterau/ADNI_data/wang_et_al/" + "data" + subdir_num + "/"

bet_cropped_dir = "/mnt/nfs/work1/mfiterau/ADNI_data/wang_et_al/cropped"
bet_skullstripped_dir = "/mnt/nfs/work1/mfiterau/ADNI_data/wang_et_al/skullstripped"
bet_registered_dir = "/mnt/nfs/work1/mfiterau/ADNI_data/wang_et_al/registered"

BET_data_mapping = []

def match_viscode(PTID, series_ID):
    PTID_info = MRI_list[(MRI_list["SUBJECT"] == PTID) & (MRI_list["SERIESID"] == int(series_ID[1:]))]
    VIS_info = PTID_info["VISIT"]
    if VIS_info.shape[0] > 0:
        VIS_info = VIS_info.values[0]
        if VIS_info[:5] == "ADNI ":
            visname = VIS_info.split(" ")[1]
        elif VIS_info[:9] == "ADNI1/GO ":
            visname = " ".join(VIS_info.split(" ")[1:])
        else: 
            visname = VIS_info
        VISCODE = VISITS_list[VISITS_list["VISNAME"] == visname]["VISCODE"].iloc[0]
        VISCODE = VISCODE if VISCODE != "sc" else "bl"
        if ADNI_merge[(ADNI_merge["PTID"] == PTID) & (ADNI_merge["VISCODE"] == VISCODE)].empty == False:
            return VISCODE
    return None

def preprocess_wang(PTID, VISCODE, MRI_path):
    if os.path.exists(bet_cropped_dir + "/" + PTID) == False:
        os.mkdir(bet_cropped_dir + "/" + PTID)
        os.mkdir(bet_skullstripped_dir + "/" + PTID)
        os.mkdir(bet_registered_dir + "/" + PTID)
    file_name = os.listdir(MRI_path)[0]
    file_orig_name = MRI_path + "/" + file_name
    file_cropped_name = bet_cropped_dir + "/" + PTID + "/" + VISCODE + "/" + file_name 
    file_skullstripped_name = bet_skullstripped_dir + "/" + PTID + "/" + VISCODE + "/" + file_name
    file_registered_name = bet_registered_dir + "/" + PTID + "/" + VISCODE + "/" + file_name 
    crop_neck_command = "robustfov -i " + file_orig_name + " -r " + file_cropped_name
    bet_command = "bet " + file_cropped_name + " " + file_skullstripped_name + " -R"
    flirt_command = "flirt -in " + file_skullstripped_name
    flirt_command += " -ref /mnt/nfs/work1/mfiterau/yfung/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"
    flirt_command += " -out " + file_registered_name + " -omat " + PTID +"_"+VISCODE+".mat"
    if os.path.exists(bet_cropped_dir + "/" + PTID + "/" + VISCODE) == False:
        os.mkdir(bet_cropped_dir + "/" + PTID + "/" + VISCODE)
        os.system(crop_neck_command)
    if os.path.exists(bet_skullstripped_dir + "/" + PTID + "/" + VISCODE) == False:
        os.mkdir(bet_skullstripped_dir + "/" + PTID + "/" + VISCODE)
        os.system(bet_command)
    if os.path.exists(bet_registered_dir + "/" + PTID + "/" + VISCODE) == False:
        os.mkdir(bet_registered_dir + "/" + PTID + "/" + VISCODE)
        os.system(flirt_command)
        label = ADNI_merge[(ADNI_merge["PTID"] == PTID) & (ADNI_merge["VISCODE"] == VISCODE)]["DX"].iloc[0]
        BET_data_mapping.append([PTID, VISCODE, label, MRI_path])

for subj in os.listdir(source_MRI):
    PTID = subj
    dir_path = source_MRI + subj + "/"
    for preprocessing_type in os.listdir(dir_path):
        dir_subj_path = dir_path + preprocessing_type 
        for vis_date in os.listdir(dir_subj_path):
            dir_subj_vis_path = dir_subj_path + "/" + vis_date 
            for series_ID in os.listdir(dir_subj_vis_path):
                dir_subj_vis_path += "/" + series_ID
                VISCODE = match_viscode(PTID, series_ID)
                if VISCODE is not None:
                    preprocess_wang(PTID, VISCODE, dir_subj_vis_path)

mri_file_mapping = "/mnt/nfs/work1/mfiterau/yfung/alzheimers-cnn-study/data/wang_et_al_mri_mapping.csv"
if os.path.exists(mri_file_mapping):
    BET_data_map = pd.read_csv(mri_file_mapping, skiprows=1)
    BET_data_map2 = pd.DataFrame(BET_data_mapping)
    BET_data_map = pd.concat([BET_data_map, BET_data_map2])
else:
    BET_data_map = pd.DataFrame(BET_data_mapping)
BET_data_map.to_csv(mri_file_mapping, index=False, header=["PTID", "VISCODE", "DX", "MRI_path"])
