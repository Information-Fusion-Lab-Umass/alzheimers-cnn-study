import os
import pickle
import traceback
import pandas as pd
import nibabel as nib

from torchvision import transforms as T
from torch.utils.data import Dataset
from pdb import set_trace
from scipy.ndimage import zoom

class DatasetBase(Dataset):
    LABEL_MAPPING = ["CN", "MCI", "AD"]
    VALID_SPLIT = ["train", "val", "test", "all"]
    VALID_TASKS = ["pretrain", "classify"]
    SPLIT_METHOD = [
        "split_all_data",
        "split_balanced_data"
    ]

    def __init__(self, config, logger, **kwargs):
        super().__init__()
        self.config = config
        self.logger = logger
        self.image_columns = config.image_columns

        shuffle = kwargs.get("shuffle", False)
        mapping_path = config.data_path

        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file \"{mapping_path}\" does not exist! Run \"mapping.py\" script in the \"util/\" directory to generate the mapping pickle file.")

        self.dataframe = self._load_data(mapping_path)
        self.fold_dataframe = None

        self.split_config = {
            "test_split": config.testing_split
        }

        if os.path.exists(config.brain_mask_path):
            self.brain_mask = nib.load(config.brain_mask_path) \
                                 .get_fdata() \
                                 .squeeze()
        else:
            self.brain_mask = None

        if shuffle:
            self.dataframe = self.dataframe.sample(frac=1)

        self.data_idx = []

    def __len__(self):
        return self.fold_dataframe.shape[0] 

    def __getitem__(self, idx):
        return None

    def process_item(self, idx):
        paths = [ self.fold_dataframe.iloc[idx][col] for col in self.image_columns ]
        images = []
        label = self.fold_dataframe.iloc[idx][self.config.label_column]

        for path in paths:
            try:
                image = nib.load(path) \
                           .get_fdata() \
                           .squeeze()

                if self.config.engine == "soes_3d":
                    x, y, z = image.shape
                    image = zoom(image, (116./x, 130./y, 83./z))
                
                if self.brain_mask is not None:
                    image *= self.brain_mask

            except Exception as e:
                print(f"Failed to load #{idx}: {paths[idx]}")
                print(f"Errors encountered: {e}")
                print(traceback.format_exc())
                return None

            images.append(image)

        return images, label

    def encode_label(self, label):
        """Transforms a string label into integer.
        """
        return self.LABEL_MAPPING.index(label)

    def decode_label(self, num):
        """Transforms an integer label into string.
        """
        return self.LABEL_MAPPING[num]

    def _load_data(self, path):
        """Load data from the mapping file.
        """
        df = pd.read_csv(path)
        #with open(path, "rb") as f:
        #    df = pickle.load(f)

        # filter out rows with empty image path
        for i in range(len(self.image_columns)):
            df = df[df[self.image_columns[i]].notnull()].reset_index(drop=True)

        # change LMCI and EMCI to MCI
        target = (df[self.config.label_column] == "LMCI") | \
                 (df[self.config.label_column] == "EMCI")
        df.loc[target, self.config.label_column] = "MCI"

        return df

    def load_split(self, split, fold_i=4):
        assert split in self.VALID_SPLIT, "splitting methodology not valid"

        df_len = self.dataframe.shape[0]
        test_ratio = self.split_config["test_split"]
        k = self.config.training_crossval_folds

        if split == "train" or split == "val":
            self.fold_dataframe = self.dataframe[0 : int((1-test_ratio) * df_len)]
            df_len = self.fold_dataframe.shape[0]
            if split == "train":
                self.fold_dataframe = self.fold_dataframe[0 : int(fold_i * df_len / k)].append(\
				self.fold_dataframe[int((fold_i+1) * df_len / k) : -1], ignore_index=True)
            elif split == "val":
                self.fold_dataframe = self.fold_dataframe[int(fold_i * df_len / k) : int((fold_i+1) * df_len / k)]
        elif split == "test":
            self.fold_dataframe = self.dataframe[int((1-test_ratio) * df_len) : -1]
        if self.config.dataset_size_limit != -1:
            self.logger.warn(f"ENFORCING DATASET SIZE LIMIT OF {self.config.dataset_size_limit}.")
            self.fold_dataframe = self.fold_dataframe[:self.config.dataset_size_limit]
        return 
        self.logger.debug(
            f"\n\tTraining size - {len(train_split)}"
            f"\n\t\tCN: {len(list(filter(lambda x: x[1]=='CN',train_split)))}, "
            f"MCI: {len(list(filter(lambda x: x[1]=='MCI',train_split)))}, "
            f"AD: {len(list(filter(lambda x: x[1]=='AD',train_split)))}"
            f"\n\tValidation size - {len(valid_split)}"
            f"\n\t\tCN: {len(list(filter(lambda x: x[1]=='CN',valid_split)))}, "
            f"MCI: {len(list(filter(lambda x: x[1]=='MCI',valid_split)))}, "
            f"AD: {len(list(filter(lambda x: x[1]=='AD',valid_split)))}"
            f"\n\tTesting size - {len(test_split)}"
            f"\n\t\tCN: {len(list(filter(lambda x: x[1]=='CN',test_split)))}, "
            f"MCI: {len(list(filter(lambda x: x[1]=='MCI',test_split)))}, "
            f"AD: {len(list(filter(lambda x: x[1]=='AD',test_split)))}"
        )
