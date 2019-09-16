import pickle
from abc import ABC
from collections.abc import Collection
from math import floor
from typing import List

import pandas as pd
from pandas.core.series import Series

from lib import Object, ImageRecord


class Mapping(Object, Collection, ABC):
    def __init__(self, mapping: List[ImageRecord] = None, mapping_path: str = None):
        Object.__init__(self)
        Collection.__init__(self)
        ABC.__init__(self)
        self.mapping: List[ImageRecord] = mapping or self._load_mapping(mapping_path)

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int) -> ImageRecord:
        return self.mapping[index]

    def __contains__(self, item) -> bool:
        return item in self.mapping

    def split(self, ratios: List[float]) -> List["Mapping"]:
        """This method splits at the 3D-image level.
        """
        assert sum(ratios) == 1.0, "Split ratio must add up to 1.0"
        num_total = len(self.mapping)
        num_per_split = list(map(lambda x: floor(num_total * x), ratios))
        splits = list(map(lambda x: Mapping(mapping=self.mapping[x]), num_per_split))

        return splits

    # ==================================================================================================================
    # Helper methods
    # ==================================================================================================================

    def _series_to_record(self, series: Series) -> ImageRecord:
        patient_id = series["PTID"]
        visit_code = series["VISCODE"]
        image_paths = list(map(lambda x: series[x], self.config.image_columns))
        label = series["DX"]

        return ImageRecord(patient_id, visit_code, image_paths, label)

    def _load_mapping(self, mapping_path: str) -> List[ImageRecord]:
        assert mapping_path is not None, "Cannot load empty path!"

        if mapping_path[-3:] == "csv":
            df = pd.read_csv(mapping_path)
        else:
            with open(mapping_path, "rb") as f:
                df = pickle.load(f)

        if self.config.num_classes == 2:
            df = df[(df["DX"] == "AD") | (df["DX"] == "CN")]

        # name of the column containing image path
        image_columns: str = self.config.image_columns
        # name of the column containing label
        label_columns: str = self.config.label_column

        # filter out rows with empty image path
        for i in range(len(image_columns)):
            df = df[df[image_columns[i]].notnull()].reset_index(drop=True)

            # change LMCI and EMCI to MCI
        target = (df[label_columns] == "LMCI") | (df[label_columns] == "EMCI")
        df.loc[target, label_columns] = "MCI"

        return list(map(self._series_to_record, df.iterrows()))
