import random
import numpy as np


class SimpleBatcher:
    def __init__(self, batch_size, dataset):
        self.batch_size = batch_size
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        shuffeled_data = list(self.dataset)
        while len(shuffeled_data) < self.batch_size:
            shuffeled_data += shuffeled_data
        random.shuffle(shuffeled_data)
        mod = len(shuffeled_data) % self.batch_size
        if mod != 0:
            shuffeled_data += shuffeled_data[:self.batch_size - mod]
            random.shuffle(shuffeled_data)
        batched_points = []
        batch_size = self.batch_size
        n = int(len(self.dataset) / batch_size)
        for i in range(n):
            batch_x = [p[0] for p in shuffeled_data[i * batch_size:(i + 1) * batch_size]]
            batch_y = [p[1] for p in shuffeled_data[i * batch_size:(i + 1) * batch_size]]
            batched_points.append((np.array(batch_x).T, np.array(batch_y)))
        return iter(batched_points)

    def subset_percent(self, percent, batch_size=None):
        assert 0 < percent and percent <= 1
        shuffeled_data = list(self.dataset)
        random.shuffle(shuffeled_data)
        length = int(len(self) * percent)
        if batch_size is None:
            batch_size = self.batch_size
        return SimpleBatcher(batch_size, shuffeled_data[: length])
