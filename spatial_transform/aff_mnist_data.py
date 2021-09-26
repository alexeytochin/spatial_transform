import os
from typing import Tuple
import numpy as np
from scipy.io import loadmat
from glob import glob


from config import PATH_TO_AFF_MNIST_DATA


IMAGE_SIZE = 40
IMAGE_SHAPE = (40, 40)
IMAGE_NUM_CHANNELS = 1


# Borrowed from https://www.kaggle.com/kmader/repackage-dataset-into-competition


def read_many_aff(in_paths):
    img_out, label_out = [], []
    for c_path in in_paths:
        a, b = read_affdata(c_path)
        img_out += [a]
        label_out += [b]
    return np.concatenate(img_out, 0), np.concatenate(label_out, 0)


def read_affdata(in_path) -> Tuple[np.ndarray, np.ndarray]:
    v = loadmat(in_path)['affNISTdata'][0][0]
    img = v[2].reshape((40, 40, -1)).swapaxes(0, 2).swapaxes(1, 2)
    label = v[5][0].astype(np.int32)
    return img, label


def get_aff_mnist_data(path_to_data_dir: str=PATH_TO_AFF_MNIST_DATA) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :param path_to_data_dir: path to directoru with affMNIST data *.mat files
    :return: 6 np.ndarray objects of images and labels for 3 datasets partitions
    """
    train_img_data, train_img_label = \
        read_many_aff(glob(os.path.join(path_to_data_dir, 'training_batches/training_batches/*.mat')))
    validation_img_data, validation_img_label = \
        read_affdata(os.path.join(path_to_data_dir, 'validation.mat'))
    test_img_data, test_img_label = \
        read_affdata(os.path.join(path_to_data_dir, 'test.mat'))

    return train_img_data, train_img_label, validation_img_data, validation_img_label, test_img_data, test_img_label