from datasets.load import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import transforms

import torch
import torchvision
from typing import Dict, Any, List

import os
import random
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        image, label = super(CIFAR10, self).__getitem__(index)
        return index, image, label


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, *args, **kwargs):
        super(SVHN, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        image, label = super(SVHN, self).__getitem__(index)
        return index, image, label


class ImageNet100(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        return idx, image, label


def create_dataset(dataset_name: str, batch_size: int) -> Dict[str, Any]:
    """Creates the train/test datasets.
    Args:
        dataset_name: Name of the dataset (e.g. cifar10).
        batch_size: Batch Size.
    Returns:
        A dictionary with keys 'train_dl' and 'test_dl' whose values are the train and test data loader respectively.
    """
    logging.info('Creating the train and test datasets for %s', dataset_name)
    if dataset_name == 'cifar10':
        num_classes = 10
        transform = transforms.Compose([transforms.ToTensor(), ])

        train_ds = CIFAR10(root='./data',
                           train=True,
                           download=True,
                           transform=transform)
        test_ds = CIFAR10(root='./data',
                          train=False,
                          download=True,
                          transform=transform)
        train_ds.targets = train_ds.targets

    elif dataset_name == 'svhn':
        num_classes = 10
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = SVHN(root='./data', split='train', download=True, transform=transform)
        test_ds = SVHN(root='./data', split='test', download=True, transform=transform)
        train_ds.targets = train_ds.labels

    elif dataset_name == 'imagenet100':
        num_classes = 100
        train_ds = load_dataset("clane9/imagenet-100", split="train", cache_dir='./data')
        test_ds = load_dataset("clane9/imagenet-100", split="validation", cache_dir='./data')

        def train_transform(examples):
            t_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.RandomResizedCrop(size=(128, 128)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [t_transform(x) for x in examples["image"]]
            return examples

        def validation_transform(examples):
            v_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(size=160),
                    transforms.CenterCrop(size=128),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [v_transform(x) for x in examples["image"]]
            return examples

        train_ds.set_transform(transform=train_transform)
        test_ds.set_transform(transform=validation_transform)
        train_ds = ImageNet100(train_ds)
        test_ds = ImageNet100(test_ds)

    else:
        raise ValueError('Dataset %s is not implemented yet...', dataset_name)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    return {'train_dl': train_dl, 'test_dl': test_dl, 'num_classes': num_classes}


def create_unlearning_dataset(dataset_name: str, batch_size: int,
                              forget_ratio: float, forget_mode: str, forget_classes: List[int],
                              forget_data_dir: str) -> Dict[str, Any]:
    """Creates the train/test datasets.
    Args:
        dataset_name: Name of the dataset (e.g. cifar10).
        batch_size: Batch Size.
        forget_ratio: Ratio of forget samples (e.g. 10% ).
        forget_mode: Mode of creating forget set (e.g. iid)
        forget_classes: Forget classes.
        forget_data_dir: Directory of forget data indices.
    Returns:
        A dictionary with keys 'train_dl' and 'test_dl' whose values are the train and test data loader respectively.
    """
    logging.info('Creating the train and test datasets for %s', dataset_name)
    if dataset_name == 'cifar10':
        num_classes = 10
        transform = transforms.Compose([transforms.ToTensor(), ])

        train_ds = CIFAR10(root='./data',
                           train=True,
                           download=True,
                           transform=transform)
        test_ds = CIFAR10(root='./data',
                          train=False,
                          download=True,
                          transform=transform)
        train_ds.targets = train_ds.targets

    elif dataset_name == 'svhn':
        num_classes = 10
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = SVHN(root='./data', split='train', download=True, transform=transform)
        test_ds = SVHN(root='./data', split='test', download=True, transform=transform)
        train_ds.targets = train_ds.labels

    elif dataset_name == 'imagenet100':
        num_classes = 100
        train_ds = load_dataset("clane9/imagenet-100", split="train", cache_dir='./data')
        test_ds = load_dataset("clane9/imagenet-100", split="validation", cache_dir='./data')

        def train_transform(examples):
            t_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.RandomResizedCrop(size=(128, 128)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [t_transform(x) for x in examples["image"]]
            return examples

        def validation_transform(examples):
            v_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(size=160),
                    transforms.CenterCrop(size=128),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            examples["image"] = [v_transform(x) for x in examples["image"]]
            return examples

        train_ds.set_transform(transform=train_transform)
        test_ds.set_transform(transform=validation_transform)
        train_ds = ImageNet100(train_ds)
        test_ds = ImageNet100(test_ds)
        train_ds.targets = [train_ds[i][1] for i in range(len(train_ds))]

    else:
        raise ValueError('Dataset %s is not implemented yet...', dataset_name)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    if os.path.exists(forget_data_dir):
        forget_indices = torch.load(forget_data_dir)
        logging.info('Loaded forget indices from %s', forget_data_dir)
    else:
        if forget_mode == 'iid':
            forget_classes = list(range(num_classes))
            class_indices = [i for i, label in enumerate(train_ds.targets) if label in forget_classes]
        if forget_mode == 'non-iid':
            class_indices = [i for i, label in enumerate(train_ds.targets) if str(label) in forget_classes]
        n_forget = int(forget_ratio * len(train_ds))
        forget_indices = random.sample(class_indices, n_forget)
        torch.save(forget_indices, forget_data_dir)
        logging.info('Saved forget indices at %s', forget_data_dir)

    logging.info(f'The first five forget indices are: {forget_indices[0:5]}')
    logging.info(f'Number of forget indices are :{len(forget_indices)}')

    all_indices = set(range(len(train_ds)))
    retain_indices = list(all_indices - set(forget_indices))
    forget_subset = Subset(train_ds, forget_indices)
    retain_subset = Subset(train_ds, retain_indices)

    forget_dl = DataLoader(forget_subset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    retain_dl = DataLoader(retain_subset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    return {'train_dl': train_dl, 'test_dl': test_dl, 'forget_dl': forget_dl, 'retain_dl': retain_dl,
            'forget_ds': forget_subset,
            'retain_ds': retain_subset,
            'num_classes': num_classes}
