
from __future__ import division

supported_datasets = ['bmnist', 'silhouettes']
# ToDo: # 'mnist' and 'tfd' are not normalized (0<= x <=1.)


def get_data(data_name):
    if data_name == 'mnist':
        from fuel.datasets import MNIST
        img_size = (28, 28)
        channels = 1
        data_train = MNIST(which_sets=["train"], sources=['features'])
        data_valid = MNIST(which_sets=["test"], sources=['features'])
        data_test  = MNIST(which_sets=["test"], sources=['features'])
    elif data_name == 'bmnist':
        from fuel.datasets.binarized_mnist import BinarizedMNIST
        img_size = (28, 28)
        channels = 1
        data_train = BinarizedMNIST(which_sets=['train'], sources=['features'])
        data_valid = BinarizedMNIST(which_sets=['valid'], sources=['features'])
        data_test  = BinarizedMNIST(which_sets=['test'], sources=['features'])
    # TODO: make a generic catch-all for loading custom datasets like "colormnist"
    elif data_name == 'colormnist':
        from draw.colormnist import ColorMNIST
        img_size = (28, 28)
        channels = 3
        data_train = ColorMNIST(which_sets=['train'], sources=['features'])
        data_valid = ColorMNIST(which_sets=['test'], sources=['features'])
        data_test  = ColorMNIST(which_sets=['test'], sources=['features'])
    elif data_name == 'cifar10':
        from fuel.datasets.cifar10 import CIFAR10
        img_size = (32, 32)
        channels = 3
        data_train = CIFAR10(which_sets=['train'], sources=['features'])
        data_valid = CIFAR10(which_sets=['test'], sources=['features'])
        data_test  = CIFAR10(which_sets=['test'], sources=['features'])
    elif data_name == 'svhn2':
        from fuel.datasets.svhn import SVHN
        img_size = (32, 32)
        channels = 3
        data_train = SVHN(which_format=2,which_sets=['train'], sources=['features'])
        data_valid = SVHN(which_format=2,which_sets=['test'], sources=['features'])
        data_test  = SVHN(which_format=2,which_sets=['test'], sources=['features'])
    elif data_name == 'silhouettes':
        from fuel.datasets.caltech101_silhouettes import CalTech101Silhouettes
        size = 28
        img_size = (size, size)
        channels = 1
        data_train = CalTech101Silhouettes(which_sets=['train'], size=size, sources=['features'])
        data_valid = CalTech101Silhouettes(which_sets=['valid'], size=size, sources=['features'])
        data_test  = CalTech101Silhouettes(which_sets=['test'], size=size, sources=['features'])
    elif data_name == 'tfd':
        from fuel.datasets.toronto_face_database import TorontoFaceDatabase
        img_size = (28, 28)
        channels = 1
        data_train = TorontoFaceDatabase(which_sets=['unlabeled'], size=size, sources=['features'])
        data_valid = TorontoFaceDatabase(which_sets=['valid'], size=size, sources=['features'])
        data_test  = TorontoFaceDatabase(which_sets=['test'], size=size, sources=['features'])
    else:
        raise ValueError("Unknown dataset %s" % data_name)

    return img_size, channels, data_train, data_valid, data_test
