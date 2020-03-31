import pickle
from collections import Iterable
from copy import deepcopy
from math import floor
from typing import List

import pandas as pd
from pandas.core.series import Series

from lib import Object, ImageRecord


class Mapping(Object, Iterable):
    def __init__(self, mapping: List[ImageRecord] = None, mapping_path: str = None):
        Object.__init__(self)
        Iterable.__init__(self)

        assert mapping is not None or mapping_path is not None, \
            "Must provide either a list of ImageRecord objects or a mapping path where such list can " \
            "be constructed from."

        self.mapping: List[ImageRecord] = mapping or self._load_mapping(mapping_path)
        self.iter_idx = 0

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int) -> ImageRecord:
        return self.mapping[index]

    def __iter__(self):
        return (record for record in self.mapping)

    def __contains__(self, item) -> bool:
        return item in self.mapping

    def __add__(self, other: 'Mapping') -> 'Mapping':
        return Mapping(mapping=deepcopy(self.mapping + other.mapping))

    def __deepcopy__(self, memodict={}) -> 'Mapping':
        return Mapping(mapping=deepcopy(self.mapping))

    def split_by_ratio(self, ratios: List[float]) -> List["Mapping"]:
        """This method splits the mapped images into sets based on the specified ratio.
        """
        assert abs(sum(ratios) - 1.0) < 0.00000001, "Split ratio must add up to 1.0"
        num_total = len(self.mapping)
        num_per_split = list(map(lambda x: floor(num_total * x), ratios))
        splits = []
        if len(num_per_split) == 2:
            splits = [Mapping(mapping=deepcopy(self.mapping[0:num_per_split[0]])), \
                      Mapping(mapping=deepcopy(self.mapping[-1*num_per_split[1]:-1]))]
        elif len(num_per_split) == 3:
            [Mapping(mapping=deepcopy(self.mapping[0:num_per_split[0]])), \
                     Mapping(mapping=deepcopy(self.mapping[num_per_split[0]:-1*num_per_split[2]])), \
                      Mapping(mapping=deepcopy(self.mapping[-1*num_per_split[2]:-1]))]
        else:
            quit()
        return splits

    def shuffle(self) -> 'Mapping':
        """Returns a cloned object containing shuffled records.
        """
        mapping_copy = deepcopy(self.mapping)
        return Mapping(mapping=mapping_copy)

    @classmethod
    def merge(cls, mappings: List['Mapping']) -> 'Mapping':
        result = None

        for idx in range(len(mappings)):
            if result is None:
                result = mappings[idx]
            else:
                result += mappings[idx]

        return deepcopy(result)

    # ==================================================================================================================
    # Helper methods
    # ==================================================================================================================

    def _series_to_record(self, series: Series) -> ImageRecord:
        patient_id = series[1]["PTID"]
        visit_code = series[1]["VISCODE"]
        image_path = series[1][self.config.image_column]
        label = series[1]["DX"]
        age = series[1]["AGE"]
        return ImageRecord(patient_id, visit_code, image_path, label, age)

    def _load_mapping(self, mapping_path: str) -> List[ImageRecord]:
        assert mapping_path is not None, "Cannot load empty path!"
        # name of the column containing image path
        image_column: str = self.config.image_column
        # name of the column containing label
        label_column: str = self.config.label_column
        if mapping_path[-3:] == "csv":
            df = pd.read_csv(mapping_path,names=["PTID", "VISCODE", label_column, image_column, "AGE"])
            if df["PTID"].iloc[0] == "PTID":
                df = pd.read_csv(mapping_path,
                             dtype={image_column: str, label_column: str, "PTID": str, "VISCODE": str})
        else:
            with open(mapping_path, "rb") as f:
                df = pickle.load(f)
        
        if self.config.num_classes == 2:
            df = df[(df["DX"] == "AD") | (df["DX"] == "CN")]

        # filter out rows with empty image path
        df = df[df[image_column].notnull()].reset_index(drop=True)

        return list(map(self._series_to_record, df.iterrows()))

    def two_class(self, memodict={}) -> 'Mapping':
        return Mapping(mapping=deepcopy(list(filter(lambda x: x.label != "MCI", self.mapping))))
