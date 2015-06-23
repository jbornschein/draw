
from __future__ import division

supported_datasets = ['bmnist', 'mnist', 'silhouettes']
# 'tfd' is missing but needs normalization


def get_data(data_name):
    if data_name == 'mnist':
        from fuel.datasets import MNIST

        img_size = (28, 28)

        data_train = MNIST(which_sets=["train"], sources=['features'])
        data_valid = MNIST(which_sets=["test"], sources=['features'])
        data_test  = MNIST(which_sets=["test"], sources=['features'])
    elif data_name == 'bmnist':
        from fuel.datasets.binarized_mnist import BinarizedMNIST

        img_size = (28, 28)

        data_train = BinarizedMNIST(which_sets=['train'], sources=['features'])
        data_valid = BinarizedMNIST(which_sets=['valid'], sources=['features'])
        data_test  = BinarizedMNIST(which_sets=['test'], sources=['features'])
    elif data_name == 'silhouettes':
        from fuel.datasets.caltech101_silhouettes import CalTech101Silhouettes

        size = 28 
        img_size = (size, size)

        data_train = CalTech101Silhouettes(which_sets=['train'], size=size, sources=['features'])
        data_valid = CalTech101Silhouettes(which_sets=['valid'], size=size, sources=['features'])
        data_test  = CalTech101Silhouettes(which_sets=['test'], size=size, sources=['features'])
    elif data_name == 'tfd':
        from fuel.datasets.toronto_face_database import TorontoFaceDatabase

        size = 28
        img_size = (size, size)

        data_train = TorontoFaceDatabase(which_sets=['unlabeled'], size=size, sources=['features'])
        data_valid = TorontoFaceDatabase(which_sets=['valid'], size=size, sources=['features'])
        data_test  = TorontoFaceDatabase(which_sets=['test'], size=size, sources=['features'])
    else:
        raise ValueError("Unknown dataset %s" % data_name)

    return img_size, data_train, data_valid, data_test
