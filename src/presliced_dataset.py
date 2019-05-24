import os
import torch
import random
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from pdb import set_trace

class PreslicedDataset(Dataset):
    VALID_MODES = ["train", "valid", "test"]

    def __init__(self, **kwargs):
        super().__init__()
        self.mode = kwargs.get("mode", "train")

        transforms = kwargs.get("transforms", [
            T.ToTensor()
        ])

        self.transforms = T.Compose(transforms)
        # self.data_path = kwargs.get("data_path", "/mnt/nfs/work1/mfiterau/ADNI_data/slice_subsample_no_seg/coronal_skullstrip")
        # self.data_path = kwargs.get("data_path", "/mnt/nfs/work1/mfiterau/ADNI_data/slice_all_no_seg/coronal_skullstrip")

        # self.data_path = kwargs.get("data_path", "/mnt/nfs/work1/mfiterau/ADNI_data/slice_all_spm_no_seg_NEW/coronal_skullstrip")
        self.data_path = kwargs.get("data_path", "/mnt/nfs/work1/mfiterau/ADNI_data/slice_all_spm_no_seg_NEW/sagital_skullstrip")



        subject_paths = os.listdir(self.data_path)
        cn = list(filter(lambda x: "CN" in x, subject_paths))
        ad = list(filter(lambda x: "AD" in x, subject_paths))
        mci = list(filter(lambda x: "MCI" in x, subject_paths))

        count = min(len(cn), len(ad), len(mci))
        cn = cn[:count]
        ad = ad[:count]
        mci = mci[:count]

        valid_split = kwargs.get("valid_split", 0.1)
        test_split = kwargs.get("test_split", 0.1)
        valid_count = int(count * valid_split)
        test_count = int(count * test_split)
        train_count = count - valid_count - test_count

        if self.mode == "train":
            self.cn = cn[:train_count]
            self.ad = ad[:train_count]
            self.mci = mci[:train_count]
        elif self.mode == "valid":
            self.cn = cn[train_count:train_count+valid_count]
            self.ad = ad[train_count:train_count+valid_count]
            self.mci = mci[train_count:train_count+valid_count]
        elif self.mode == "test":
            self.cn = cn[train_count+valid_count:]
            self.ad = ad[train_count+valid_count:]
            self.mci = mci[train_count+valid_count:]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["CN", "AD", "MCI"])

        print("{} split: CN {}, AD {}, MCI {}"
                    .format(self.mode,
                        len(self.cn), len(self.ad), len(self.mci)))

        assert len(self.cn) == len(self.ad) == len(self.mci), \
            "Imbalanced classes: {} vs. {} vs. {}" \
                .format(len(self.cn), len(self.ad), len(self.mci))

    def __len__(self):
        return len(self.cn) + len(self.ad) + len(self.mci)

    def __getitem__(self, idx):
        class_len = len(self.cn)
        class_idx = int(idx / class_len)
        data_idx = idx % class_len

        if class_idx == 0:
            subject_path = self.cn[data_idx]
            label = self.label_encoder.transform(["CN"])[0]
        elif class_idx == 1:
            subject_path = self.ad[data_idx]
            label = self.label_encoder.transform(["AD"])[0]
        elif class_idx == 2:
            subject_path = self.mci[data_idx]
            label = self.label_encoder.transform(["MCI"])[0]
        else:
            raise Exception("Invalid class index computed: {}"
                    .format(class_idx))

        # coronal
        # full_path = "{}/{}/normalized_seg_33.tiff" \
        #                 .format(self.data_path, subject_path)

        # sagital
        full_path = "{}/{}/normalized_seg_25.tiff" \
                        .format(self.data_path, subject_path)

        try:
            image = Image.open(full_path)
        except Exception as e:
            print("Failed to load image at: {}".format(full_path))
            print("\t{}".format(e))
            return None, None

        if image is None:
            print("Failed to load image at: {}".format(full_path))
            return None, None
        else:
            return self.transforms(image)[:3], label

if __name__ ==  "__main__":
    dataset = PreslicedDataset()
