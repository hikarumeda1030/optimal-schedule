import random
from torch.utils.data import Sampler


class DynamicBatchSampler(Sampler):
    def __init__(self, data_len, scheduler, shuffle=True, drop_last=False):
        self.data_len = data_len
        self.scheduler = scheduler
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(self.data_len))
        if self.shuffle:
            random.shuffle(indices)

        i = 0
        while i < self.data_len:
            batch_size = self.scheduler.batch_size
            end = min(i + batch_size, self.data_len)

            # drop_last処理
            if self.drop_last and (end - i < batch_size):
                break

            yield indices[i:end]
            i = end

    def __len__(self):
        return (self.data_len + self.scheduler.batch_size - 1) // self.scheduler.batch_size
