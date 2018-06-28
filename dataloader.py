import math
import torch
from dataset_file import DatasetFile
from converter import ImageConverter


class CustomDataloader(object):
    def __init__(self, dataset='mnist.dataset', batch_size=16, fold='train', shuffle=True, last_batch=False, example_count=None, img_format=None, **kwargs):
        self.dsf = DatasetFile(dataset, example_count=example_count)
        if img_format is None:
            img_format = ImageConverter
        self.img_conv = img_format(self.dsf, **kwargs)
        self.batch_size = batch_size
        self.fold = fold
        self.last_batch = last_batch
        self.shuffle = shuffle

    def get_batch(self, **kwargs):
        batch = self.dsf.get_batch(fold=self.fold, batch_size=self.batch_size, **kwargs)
        images, labels = self.convert(batch)
        return images, labels

    def __iter__(self):
        self.batcher = self.dsf.get_all_batches(
                fold=self.fold,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                last_batch=self.last_batch)
        return self

    def __next__(self):
        batch = next(self.batcher)
        images, labels = self.convert(batch)
        return images, labels

    def convert(self, batch):
        images = self.img_conv(batch)
        images = torch.FloatTensor(images).cuda()
        labels = None
        return images, labels

    def __len__(self):
        return math.floor(self.dsf.count(self.fold) / self.batch_size)

    def count(self):
        return self.dsf.count(self.fold)
