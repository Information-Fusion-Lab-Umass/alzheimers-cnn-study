import nibabel as nib
import pickle
import torch
import os
import cv2
import torchvision
import matplotlib.pyplot as plt

with open('outputs/normalized_mapping.pickle', 'rb') as f:
    x = pickle.load(f)

dest_dir = "/mnt/nfs/work1/mfiterau/ADNI_data/slice_subsample_no_seg/axial_noskullstrip/"

def help_process_data(data, label):
    patient_id = data["patient_id"]
    visit_code = data["visit_code"]
    misc_path = data["misc"]
    if not os.path.exists(dest_dir+patient_id+"_"+visit_code+"_"+label):
        os.mkdir(dest_dir+patient_id+"_"+visit_code+"_"+label)
    mri = nib.load(misc_path).get_fdata().squeeze()  
    mri = torch.tensor(mri).permute([0,2,1])  
    mri = mri.flip(1)
    mri = mri.flip(2)
    for idx in range(5,50,10):
        middle_index = mri.shape[1] // 2
        plt.imshow(mri[:, middle_index-25+idx, :])
        plt.axis('off')
        plt.savefig(dest_dir+patient_id+"_"+visit_code+"_"+label+'/normalized_seg_'+str(idx)+".tiff", dpi='figure', bbox_inches='tight')
    with open(dest_dir+r'subjectID_label_match.txt', 'a') as f:
        f.write(dest_dir+patient_id + "_" + visit_code + "_" + label + "\n")
    print("sucess")
  
NC_count = 0
MCI_count = 0
AD_count = 0

patient_ids = []
patient_labels = []

idx = 535
while not (AD_count >= 100):
    try:
        label = x.iloc[idx]['label']
        #if label == "CN" and NC_count < 100:
        #    help_process_data(x.iloc[idx], label)
        #    NC_count += 1
        #    patient_ids.append(x.iloc[idx]['patient_id'])
        #    patient_labels.append(label)
        #elif (label == "LMCI" or label == "EMCI") and MCI_count < 100:
        #    help_process_data(x.iloc[idx], "MCI")
        #    MCI_count += 1
        #    patient_ids.append(x.iloc[idx]['patient_id'])
        #    patient_labels.append(label)
        if (label == "AD") and AD_count < 100:
            help_process_data(x.iloc[idx], label)
            AD_count += 1
            patient_ids.append(x.iloc[idx]['patient_id'])
            patient_labels.append(label)
    except:
        print("failed:" + str(idx))
    idx += 1

