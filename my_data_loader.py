"""
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
[2]: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""

import torch
import numpy as np

from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_data_transforms(dataset, augment=True, scale=False):
    print("Getting",dataset,"Transforms")
    if dataset == 'cifar10':
        return _data_transforms_cifar10(augment, scale)
    if dataset == 'mnist':
        return _data_transforms_mnist(augment, scale)
    if dataset == 'emnist':
        return _data_transforms_emnist(augment, scale)
    if dataset == 'fashion':
        return _data_transforms_fashion(augment, scale)
    if dataset == 'svhn':
        return _data_transforms_svhn(augment, scale)
    if dataset == 'stl10':
        return _data_transforms_stl10(augment, scale)
    if dataset == 'devanagari':
        return _data_transforms_devanagari(augment)
    assert False, "Cannot get Transform for dataset"

# Transform defined for cifar-10
def _data_transforms_cifar10(augment, scale):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    elif scale:
        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        valid_transform=transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform


# Transform defined for mnist
def _data_transforms_mnist(augment, scale):
    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
    elif scale:
        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(28),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
        valid_transform=transforms.Compose([
            transforms.Scale(28),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
    return train_transform, valid_transform


# Transform defined for fashion mnist
def _data_transforms_fashion(augment, scale):
    FASHION_MEAN = (0.2860405969887955,)
    FASHION_STD = (0.35302424825650003,)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD),
        ])
    elif scale:
        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(28),
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD),
        ])
        valid_transform=transforms.Compose([
            transforms.Scale(28),
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD),
        ])
    return train_transform, valid_transform


# Transform defined for emnist
def _data_transforms_emnist(augment, scale):
    EMNIST_MEAN = (0.17510417052459282,)
    EMNIST_STD = (0.33323714976320795,)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
        ])
    elif scale:
        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(28),
            transforms.ToTensor(),
            transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
        ])
        valid_transform=transforms.Compose([
            transforms.Scale(28),
            transforms.ToTensor(),
            transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
        ])
    return train_transform, valid_transform


# Transform defined for svhn
def _data_transforms_svhn(augment, scale):
    SVHN_MEAN = [ 0.4376821,   0.4437697,   0.47280442]
    SVHN_STD = [ 0.19803012,  0.20101562,  0.19703614]

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
    elif scale:
        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
        valid_transform=transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
    return train_transform, valid_transform


# Transform defined for stl10
def _data_transforms_stl10(augment, scale):
    STL_MEAN = [ 0.44671062,  0.43980984,  0.40664645]
    STL_STD = [ 0.26034098,  0.25657727,  0.27126738]

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(STL_MEAN, STL_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(STL_MEAN, STL_STD),
        ])
    elif scale:
        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(96),
            transforms.ToTensor(),
            transforms.Normalize(STL_MEAN, STL_STD),
        ])
        valid_transform=transforms.Compose([
            transforms.Scale(96),
            transforms.ToTensor(),
            transforms.Normalize(STL_MEAN, STL_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(STL_MEAN, STL_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(STL_MEAN, STL_STD),
        ])
    return train_transform, valid_transform


# Transform defined for devanagari hand written symbols
def _data_transforms_devanagari(augment, scale):
    DEVANAGARI_MEAN = (0.240004663268,)
    DEVANAGARI_STD = (0.386530114768,)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=2), #Already has padding 2 and size is 32x32
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(DEVANAGARI_MEAN, DEVANAGARI_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(DEVANAGARI_MEAN, DEVANAGARI_STD),
        ])
    elif scale:
        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize(DEVANAGARI_MEAN, DEVANAGARI_STD),
        ])
        valid_transform=transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize(DEVANAGARI_MEAN, DEVANAGARI_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(DEVANAGARI_MEAN, DEVANAGARI_STD),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(DEVANAGARI_MEAN, DEVANAGARI_STD),
        ])
    return train_transform, valid_transform

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False,
                           dataset='cifar10'):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    ##
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )

    # define transforms
    # valid_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    # ])
    # if augment:
    #     train_transform = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # else:
    #     train_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    train_transform, valid_transform = get_data_transforms(augment, scale=False)

    # # load the dataset
    # train_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=train_transform,
    # )

    # valid_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=valid_transform,
    # )
    ##

    if dataset == 'cifar10':
        print("Using CIFAR10")
        train_dataset = dset.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        valid_dataset = dset.CIFAR10(root=data_dir, train=True, download=True, transform=valid_transform)
    elif dataset == 'mnist':
        print("Using MNIST")
        train_dataset = dset.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        valid_dataset = dset.MNIST(root=data_dir, train=True, download=True, transform=valid_transform)
    elif dataset == 'emnist':
        print("Using EMNIST")
        train_dataset = dset.EMNIST(root=data_dir, split='balanced', train=True, download=True, transform=train_transform)
        valid_dataset = dset.EMNIST(root=data_dir, split='balanced', train=True, download=True, transform=valid_transform)
    elif dataset == 'fashion':
        print("Using Fashion")
        train_dataset = dset.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
        valid_dataset = dset.FashionMNIST(root=data_dir, train=True, download=True, transform=valid_transform)
    elif dataset == 'svhn':
        print("Using SVHN")
        train_dataset = dset.SVHN(root=data_dir, split='train', download=True, transform=train_transform)
        valid_dataset = dset.SVHN(root=data_dir, split='train', download=True, transform=valid_transform)
    elif dataset == 'stl10':
        print("Using STL10")
        stl10_data = os.path.join(data_dir, "STL-10")
        train_dataset = dset.STL10(root=stl10_data, split='train', download=True, transform=train_transform)
        valid_dataset = dset.STL10(root=stl10_data, split='train', download=True, transform=valid_transform)
    elif dataset == 'devanagari':
        print("Using DEVANAGARI")
        shuffle = True
        train_dir = os.path.join(data_dir, "DevanagariHandwrittenCharacterDataset", "Train")
        test_dir = os.path.join(data_dir, "DevanagariHandwrittenCharacterDataset", "Test")
        def grey_pil_loader(path):
          # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
          with open(path, 'rb') as f:
              img = Image.open(f)
              img = img.convert('L')
              return img
        # Ensure dataset is present in the directory args.data. Does not support auto download
        train_dataset = dset.ImageFolder(root=train_dir, transform=train_transform, loader = grey_pil_loader)
        valid_dataset = dset.ImageFolder(root=test_dir, transform=valid_transform, loader = grey_pil_loader)
    else:
        assert False, "Cannot get training queue for dataset"

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    print("Total Training size",num_train)
    print("Training set size",split)
    print("Validation set size",num_train-split)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_train_test_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False,
                           dataset='cifar10'):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    ##
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )

    # define transforms
    # valid_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    # ])
    # if augment:
    #     train_transform = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # else:
    #     train_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    train_transform, test_transform = get_data_transforms(augment=False, scale=True)

    # # load the dataset
    # train_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=train_transform,
    # )

    # valid_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=valid_transform,
    # )
    ##

    if dataset == 'cifar10':
        print("Using CIFAR10")
        train_dataset = dset.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = dset.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    elif dataset == 'mnist':
        print("Using MNIST")
        train_dataset = dset.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = dset.MNIST(root=data_dir, train=False, download=True, transform=test_transform)
    elif dataset == 'emnist':
        print("Using EMNIST")
        train_dataset = dset.EMNIST(root=data_dir, split='balanced', train=True, download=True, transform=train_transform)
        test_dataset = dset.EMNIST(root=data_dir, split='balanced', train=False, download=True, transform=test_transform)
    elif dataset == 'fashion':
        print("Using Fashion")
        train_dataset = dset.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = dset.FashionMNIST(root=data_dir, train=False, download=True, transform=test_transform)
    elif dataset == 'svhn':
        print("Using SVHN")
        train_dataset = dset.SVHN(root=data_dir, split='train', download=True, transform=train_transform)
        test_dataset = dset.SVHN(root=data_dir, split='test', download=True, transform=test_transform)
    elif dataset == 'stl10':
        print("Using STL10")
        stl10_data = os.path.join(data_dir, "STL-10")
        train_dataset = dset.STL10(root=stl10_data, split='train', download=True, transform=train_transform)
        test_dataset = dset.STL10(root=stl10_data, split='test', download=True, transform=test_transform)
    elif dataset == 'devanagari':
        print("Using DEVANAGARI")
        train_dir = os.path.join(data_dir, "DevanagariHandwrittenCharacterDataset", "Train")
        test_dir = os.path.join(data_dir, "DevanagariHandwrittenCharacterDataset", "Test")
        shuffle = True
        def grey_pil_loader(path):
          # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
          with open(path, 'rb') as f:
              img = Image.open(f)
              img = img.convert('L')
              return img
        # Ensure dataset is present in the directory args.data. Does not support auto download
        train_dataset = dset.ImageFolder(root=train_dir, transform=train_transform, loader = grey_pil_loader)
        test_dataset = dset.ImageFolder(root=test_dir, transform=test_transform, loader = grey_pil_loader)
    else:
        assert False, "Cannot get training queue for dataset"

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=int(2))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=int(2))

    return (dataloader, test_dataloader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False
                    dataset_type='cifar10'):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )

    # # define transform
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    _, transform = get_data_transforms(dataset_type)

    if dataset_type == 'cifar10':
        print("Using CIFAR10")
        dataset = dset.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_type == 'mnist':
        print("Using MNIST")
        dataset = dset.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_type == 'emnist':
        print("Using EMNIST")
        dataset = dset.EMNIST(root=data_dir, split='balanced', train=False, download=True, transform=transform)
    elif dataset_type == 'fashion':
        print("Using Fashion")
        dataset = dset.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_type == 'svhn':
        print("Using SVHN")
        dataset = dset.SVHN(root=data_dir, split='test', download=True, transform=transform)
    elif dataset_type == 'stl10':
        print("Using STL10")
        dataset = dset.STL10(root=data_dir, split='test', download=True, transform=transform)
    elif dataset_type == 'devanagari':
        print("Using DEVANAGARI")
        shuffle = True
        def grey_pil_loader(path):
          # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
          with open(path, 'rb') as f:
              img = Image.open(f)
              img = img.convert('L')
              return img
        # Ensure dataset is present in the directory args.data. Does not support auto download
        dataset = dset.ImageFolder(root=data_dir, transform=transform, loader = grey_pil_loader)
    else:
        assert False, "Cannot get training queue for dataset"

    # dataset = datasets.CIFAR10(
    #     root=data_dir, train=False,
    #     download=True, transform=transform,
    # )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
