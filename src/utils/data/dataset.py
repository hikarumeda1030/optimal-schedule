import torch
from torch.utils.data import IterableDataset, get_worker_info


class DynamicBatchDataset(IterableDataset):
    def __init__(self, dataset, scheduler, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.scheduler = scheduler
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            torch.manual_seed(torch.initial_seed() + worker_id)

        data_len = len(self.dataset)
        indices = torch.randperm(data_len) if self.shuffle else torch.arange(data_len)
        i = 0
        while i < data_len:
            batch_size = self.scheduler.batch_size
            end = min(i + batch_size, data_len)

            # drop_last logic
            if self.drop_last and (end - i < batch_size):
                break

            batch_indices = indices[i:end]
            batch = [self.dataset[j] for j in batch_indices]
            inputs, targets = zip(*batch)
            yield torch.stack(inputs), torch.tensor(targets)
            i = end
