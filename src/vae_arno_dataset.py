import torchvision.datasets as datasets
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import avg_pool2d
import sys

sys.path.insert(0, '/home/ubuntu/pycharm/arno-victor/Dataprocessing')
from imagePreprocessing import resize_im


class FlatDirectoryImageDataset(Dataset):
    """ pyTorch Dataset wrapper for the generic flat directory images dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, data_dir, transform=None, im_size=128):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()
        self.im_size=im_size

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img)
        else:
            img = Image.open(img_name)

        # downsample the image
        img = resize_im(np.float32(img), self.im_size)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # return the image:
        return img

class ArnoDataset(object):

    def __init__(self, batch_size, path, shuffle_dataset=True, num_workers=6, im_size=128):
        if not os.path.isdir(path):
            os.mkdir(path)

        # create a data source:
        data_source = FlatDirectoryImageDataset

        self._training_data = data_source(
            data_dir=path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            im_size=im_size)

        self.num_images = self._training_data.__len__()

        self._training_loader = DataLoader(
            self._training_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle_dataset,
            pin_memory=True
        )


        self._train_data_variance = self.estimate_variance(10)
        #self._train_data_variance = 1.0

    def estimate_variance(self, num_examples=10):
        samples = []
        for _ in range(num_examples):
            samples.append(self.training_data.__getitem__(np.random.randint(self.num_images)))

        samples = np.vstack(samples).reshape(-1)
        return np.var(samples / 255.0)


    @property
    def training_data(self):
        return self._training_data

    @property
    def validation_data(self):
        return self._training_data

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def validation_loader(self):
        return self._training_loader

    @property
    def train_data_variance(self):
        return self._train_data_variance
