import os
import pandas as pd


ADNI_merge = pd.read_csv("/mnt/nfs/work1/mfiterau/yfung/MRI/src/outputs/ADNIMERGE.csv", header=0)
MRI_list = pd.read_csv("/mnt/nfs/work1/mfiterau/zguan/alzheimers-cnn-study/data/MRILIST.csv", header=0)
VISITS_list = pd.read_csv("/mnt/nfs/work1/mfiterau/zguan/alzheimers-cnn-study/data/VISITS.csv", header=0)

subdir_num = "1"
source_MRI = "/mnt/nfs/work1/mfiterau/ADNI_data/soes_et_al/" + "data" + subdir_num + "/"

bet_skullstripped_dir = "/mnt/nfs/work1/mfiterau/ADNI_data/soes_et_al/skullstripped"

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
        elif "Month" in VIS_info: 
            num = "03" if "3" in VIS_info else "06"
            return "m" + num
        elif "Year" in VIS_info:
            num = VIS_info.split(" ")[2]
            return "m" + str(int(num)*12)
        elif "Screening" in VIS_info:
            return "bl"
        else: 
            visname = VIS_info
        VISCODE = VISITS_list[VISITS_list["VISNAME"] == visname]["VISCODE"]
        if VISCODE.empty:
            return None
        VISCODE = VISCODE.iloc[0]
        VISCODE = VISCODE if VISCODE != "sc" else "bl"
        if ADNI_merge[(ADNI_merge["PTID"] == PTID) & (ADNI_merge["VISCODE"] == VISCODE)].empty == False:
            return VISCODE
    return None

def dcm2niix():
    pass

def preprocess_soes(PTID, VISCODE, MRI_path):
    if os.path.exists(bet_skullstripped_dir + "/" + PTID) == False:
        os.mkdir(bet_skullstripped_dir + "/" + PTID)
    file_name = MRI_path.split("/")[-1]
    file_orig_name = MRI_path 
    file_skullstripped_name = bet_skullstripped_dir + "/" + PTID + "/" + VISCODE + "/" + file_name
    bet_command = "bet " + file_orig_name + " " + file_skullstripped_name + " -R"
    if os.path.exists(bet_skullstripped_dir + "/" + PTID + "/" + VISCODE) == False:
        os.mkdir(bet_skullstripped_dir + "/" + PTID + "/" + VISCODE)
    if os.listdir(bet_skullstripped_dir + "/" + PTID + "/" + VISCODE) == []:
        print(bet_command)
        label = ADNI_merge[(ADNI_merge["PTID"] == PTID) & (ADNI_merge["VISCODE"] == VISCODE)]["DX"].iloc[0]
        os.system(bet_command)
        BET_data_mapping.append([PTID, VISCODE, label, MRI_path])


for subj in os.listdir(source_MRI):
    PTID = subj
    dir_path = source_MRI + subj + "/"
    for preprocessing_type in os.listdir(dir_path):
        dir_subj_path = dir_path + preprocessing_type 
        for vis_date in os.listdir(dir_subj_path):
            dir_subj_vis_path, series_ID = dir_subj_path + "/" + vis_date, "" 
            for data in os.listdir(dir_subj_vis_path): 
                if data[-3:] == ".gz":
                    dir_subj_vis_path += "/" + data
                if data[0] == "S":
                    series_ID = data
            try:
                VISCODE = match_viscode(PTID, series_ID)
                print(PTID, series_ID, VISCODE)
                if VISCODE is not None:
                    preprocess_soes(PTID, VISCODE, dir_subj_vis_path)
            except:
                print("Failed for: " + dir_subj_vis_path)
mri_file_mapping = "/mnt/nfs/work1/mfiterau/yfung/alzheimers-cnn-study/data/soes_et_al_mri_mapping.csv"
if os.path.exists(mri_file_mapping):
    BET_data_map = pd.read_csv(mri_file_mapping, skiprows=1)
    BET_data_map2 = pd.DataFrame(BET_data_mapping)
    BET_data_map = pd.concat([BET_data_map, BET_data_map2])
else:
    BET_data_map = pd.DataFrame(BET_data_mapping)
BET_data_map.to_csv(mri_file_mapping, index=False, header=["PTID", "VISCODE", "DX", "MRI_path"])
