import random
import numpy as np


class SimpleBatcher:
    def __init__(self, batch_size, dataset):
        self.batch_size = batch_size
        self.dataset = dataset
        assert len(dataset)%batch_size == 0, "The length of the dataset has to be divisible by the batch size."

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        shuffeled_data = list(self.dataset)
        random.shuffle(shuffeled_data)
        batched_points = []
        batch_size = self.batch_size
        n = int(len(self.dataset)/batch_size)
        for i in range(n):
            batch = shuffeled_data[i*batch_size:(i+1)*batch_size]
            batched_points.append(np.array(batch))
        return iter(batched_points)
