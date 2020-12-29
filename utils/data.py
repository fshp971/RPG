import numpy as np
import torch


class EvalLoader():
    def __init__(self, dataset, batch_size):
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=False, drop_last=False, num_workers=4)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class TrainLoader():
    def __init__(self, dataset, batch_size):
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
        self.iterator = None

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples
