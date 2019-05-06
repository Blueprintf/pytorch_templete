import os
import pdb
import sys
import gzip
import torch
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import torch.utils.data as data
import torchvision.transforms as transforms

sys.path.append("..")


class BaseFeeder(data.Dataset):

    def __init__(self, data_path, label_path, phase="train"):
        self.phase = phase
        self.data = self.extract_images(data_path)
        self.label = self.extract_labels(label_path)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        img = self.transform(img)
        return img, self.label[index]

    def __len__(self):
        return len(self.label)

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def extract_images(self, filename):
        """Extract the images into a 4D uint8 numpy array [index, depth,y, x]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)
            return data

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def extract_labels(self, filename, one_hot=False):
        """Extract the labels into a 1D uint8 numpy array [index]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            if one_hot:
                return self.dense_to_one_hot(labels)
            return labels


if __name__ == "__main__":
    data_path = "/home/myc/Datasets/MNIST/train-images-idx3-ubyte.gz"
    label_path = "/home/myc/Datasets/MNIST/train-labels-idx1-ubyte.gz"
    dataloader = BaseFeeder(data_path, label_path)
    mnist = torch.utils.data.DataLoader(
        dataset=dataloader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    plt.ion()
    for batch in mnist:
        img = batch[0][0].numpy()
        pdb.set_trace()
        label = batch[1][0].numpy()
        print(img.shape)
        plt.imshow(img[0])
        plt.title(str(label))
        plt.pause(0.5)
        plt.cla()
