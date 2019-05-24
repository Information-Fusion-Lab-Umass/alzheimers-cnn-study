# Utility functions for DataLoader
from torch.utils.data.dataloader import default_collate

from pdb import set_trace

def invalid_collate(batch):
    '''
    Collate function for DataLoader that handles the scenario where Dataset returns (None, None) due to erraneous image files.
    '''
    batch = list(filter(lambda X: X[0] is not None and X[1] is not None, batch))

    if len(batch) == 0:
        return batch

    return default_collate(batch)
