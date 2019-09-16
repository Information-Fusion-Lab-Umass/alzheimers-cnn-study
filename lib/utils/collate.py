from torch.utils.data.dataloader import default_collate


def invalid_collate(batch):
    """Collate function for DataLoader that handles the scenario where Dataset returns (None, None), such as when the
    entity or relation do not exist in the vocab file.
    """
    batch = list(filter(lambda x: x[0] is not None, batch))

    if len(batch) == 0:
        return batch

    return default_collate(batch)
