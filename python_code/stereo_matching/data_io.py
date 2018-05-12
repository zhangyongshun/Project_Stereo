# coding=utf-8
import os
import torch
import pandas as pd
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
import numpy as np

input_height = 172
input_width = 576
output_height = 27
output_width = 142


class data(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv('./train.csv')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        dep_name = os.path.join(self.root_dir, self.frame.iloc[idx, 1])
        print(img_name)
        image = cv.imread(img_name, cv.IMREAD_COLOR)
        depth = cv.imread(dep_name, cv.IMREAD_GRAYSCALE)
        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, depth_size):
        assert isinstance(output_size, (int, tuple))
        assert isinstance(depth_size, (int, tuple))
        self.output_size = output_size
        self.depth_size = depth_size

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv.resize(image, (new_h, new_w))

        h, w = depth.shape[:2]
        if isinstance(self.depth_size, int):
            if h > w:
                new_h, new_w = self.depth_size * h / w, self.depth_size
            else:
                new_h, new_w = self.depth_size, self.depth_size * w / h
        else:
            new_h, new_w = self.depth_size

        dep = cv.resize(depth, (new_h, new_w))

        return {'image': img, 'depth': dep}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(np.float32(image)),
                'depth': torch.from_numpy(np.float32(depth))}

