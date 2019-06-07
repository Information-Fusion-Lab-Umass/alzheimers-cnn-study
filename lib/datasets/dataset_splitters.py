
class Splitter(object):
    def __init__(self, dataset, split_ratio, shuffle=False):
        """Base class for all Splitter classes.

        Args:
            dataset (List): A List in which each item is a tuple with the
                format (index, label, ...),
            split_ratio ([List]): A List containing the split ratios, for
                example, [ 0.8, 0.1, 0.1 ],
            shuffle ([Bool]): Whether to shuffle the data before splitting.
        """
        self.dataset = dataset
        self.dataset_size = len(self.dataset)
        self.split_ratio = split_ratio

    def __call__(self, split):
        return None

    def _accum_idx(self, num_list):
        """For every number in num_list, replace it with the sum of all of the numbers before it, including itself.

        Example:
            _accum_idx([3, 3, 3]) => [3, 6, 9]
        """
        return [ sum(num_list[idx + 1]) for idx, _ in enumerate(num_list) ]

class AllDataSplitter(Splitter):
    def __init__(self, dataset, split_ratio):
        """Initializes a Splitter that splits the dataset without balancing for
        classes.
        """
        super().__init__(dataset, split_ratio)

    def __call__(self):
        idx = list(range(self.dataset_size))
        split_sizes = [
            round(self.dataset_size * split) for split in self.split_ratio ]
        split_idx = self._accum_idx(split_sizes)
        splits = []

        for i, current_idx in enumerate(split_idx):
            if i == 0:
                splits.append(self.dataset[:current_idx])
            else:
                previous_idx = split_idx[i-1]
                data_split = self.dataset[previous_idx:current_idx]
                split_idx = list(map(lambda x: x[0], data_split))
                splits.append(split_idx)

        return tuple(splits)
