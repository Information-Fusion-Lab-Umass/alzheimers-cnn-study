import os
import pandas as pd


ADNI_merge = pd.read_csv("/mnt/nfs/work1/mfiterau/yfung/MRI/src/outputs/ADNIMERGE.csv", header=0)
MRI_list = pd.read_csv("/mnt/nfs/work1/mfiterau/zguan/alzheimers-cnn-study/data/MRILIST.csv", header=0)
VISITS_list = pd.read_csv("/mnt/nfs/work1/mfiterau/zguan/alzheimers-cnn-study/data/VISITS.csv", header=0)

subdir_num = "0"
source_MRI = "/mnt/nfs/work1/mfiterau/ADNI_data/wang_et_al/" + "data" + subdir_num + "/"

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
    pass #`print(PTID, VISCODE, MRI_path)

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
