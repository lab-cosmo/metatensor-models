import itertools
from typing import List

import numpy as np
import torch


class CombinedIterableDataset(torch.utils.data.IterableDataset):
    """
    Combines multiple dataloaders into a single iterable dataset.
    This is useful for combining multiple dataloaders into a single
    dataloader. The new dataloader can be shuffled or not.

    :param dataloaders: list of dataloaders to combine
    :param shuffle: whether to shuffle the combined dataloader

    :return: combined dataloader
    """

    def __init__(self, dataloaders, shuffle):
        self.dataloaders = dataloaders
        self.shuffle = shuffle

        # Create the indices:
        indices = [
            (i, dl_idx)
            for dl_idx, dl in enumerate(self.dataloaders)
            for i in range(len(dl))
        ]

        # Shuffle the indices if requested
        if self.shuffle:
            np.random.shuffle(indices)

        self.indices = indices

    def __iter__(self):
        for idx, dataloader_idx in self.indices:
            yield next(itertools.islice(self.dataloaders[dataloader_idx], idx, None))

    def __len__(self):
        return len(self.indices)


def combine_dataloaders(
    dataloaders: List[torch.utils.data.DataLoader], shuffle: bool = True
):
    """
    Combines multiple dataloaders into a single dataloader.

    :param dataloaders: list of dataloaders to combine
    :param shuffle: whether to shuffle the combined dataloader

    :return: combined dataloader
    """
    combined_dataset = CombinedIterableDataset(dataloaders, shuffle)
    return torch.utils.data.DataLoader(combined_dataset, batch_size=None)
