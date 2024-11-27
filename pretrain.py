"""Pretrains a ResNet18/ViT/ResNet50 model on CIFAR10/SVHN/ImageNet100 dataset."""
import os

from absl import flags, app
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt
import random
from unlearning_models import *
from unlearning_datasets import *

random.seed(25)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

FLAGS = flags.FLAGS
_DATASET = flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'svhn', 'cifar100', 'imagenet100'], 'Dataset')
_MODEL = flags.DEFINE_enum('model', 'resnet18', ['resnet18', 'vit', 'resnet50'], 'Model')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'Batch Size')
_LR = flags.DEFINE_float('learning_rate', 0.1, 'Learning Rate')
_WD = flags.DEFINE_float('weight_decay', 0, 'Weight Decay')
_N_EPOCHS = flags.DEFINE_integer('epochs', 50, 'Number of Epochs')
_BASE_DIR = flags.DEFINE_string('base_dir', './data/', 'The base directory to save the model.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_accuracy_dataset(model: nn.Module, dataset: DataLoader) -> Dict[str, float]:
    """ Computes loss and accuracy of the model on a given set.
    Args:
        model: The model to be evaluated.
        dataset: The dataset to evaluated the model on.
    Returns:
        A dict with 'loss' and 'acc' as the keys whose values are the loss and accuracy of the model on the dataset.
    """
    loss_all_batches = []
    acc_all_batches = []
    num_all_samples = 0
    model.eval()
    with torch.no_grad():
        for _, images, labels in dataset:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_batch = nn.CrossEntropyLoss()(logits, labels)
            loss_all_batches.append(loss_batch)

            prediction = torch.argmax(logits, 1)
            acc_all_batches.append((prediction == labels).sum().item())
            num_all_samples += len(labels)

    loss_epoch = sum(loss_all_batches) / len(loss_all_batches)
    acc_epoch = sum(acc_all_batches) / num_all_samples
    return {'loss': loss_epoch.item(), 'acc': acc_epoch}


def train_epoch(model: nn.Module,
                train_dl: DataLoader,
                optimizer: optim.Optimizer,
                scheduler: lr_scheduler) -> Dict[str, float]:
    """Trains the model for only one epoch.
    Args:
        model: The model.
        train_dl: The training data loader.
        optimizer: The optimizer.
        scheduler: Learning rate scheduler.
    Returns:
        A dict with 'loss_train', 'accuracy_train', 'loss_test', 'accuracy_test' keys whose values are
        loss and accuracy on the train and test test respectively.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    loss_all_batches = []
    acc_all_batches = []
    num_all_samples = 0
    logging.info(f'learning rate:  {scheduler.get_last_lr()[0]}')

    for _, images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)
        loss_batch = loss_fn(logits, labels)
        loss_batch.backward()
        optimizer.step()

        loss_all_batches.append(loss_batch)
        prediction = torch.argmax(logits, 1)
        acc_all_batches.append((prediction == labels).sum().item())
        num_all_samples += len(labels)

    scheduler.step()
    loss_epoch = sum(loss_all_batches) / len(loss_all_batches)
    acc_epoch = sum(acc_all_batches) / num_all_samples
    return {'loss_train_epoch': loss_epoch.item(), 'acc_train_epoch': acc_epoch}


def train(model: nn.Module,
          train_dl: DataLoader,
          test_dl: DataLoader,
          lr: float,
          weight_decay: float,
          num_epochs: int) -> Dict[str, List[float]]:
    """Trains the model.
        Args:
            model: The model.
            train_dl: The training data loader.
            test_dl: Test dataset used for evaluation.
            lr: Learning rate.
            weight_decay: Weight decay.
            num_epochs: Number of training epochs
        Returns:
            A dict with 'train_loss', 'train_acc', 'test_loss', 'test_acc' keys whose values are
            loss and accuracy on the train and test test respectively.
    """
    train_loss_all_epochs = []
    train_accuracy_all_epochs = []
    test_loss_all_epochs = []
    test_accuracy_all_epochs = []

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.01 * lr)
    with tqdm(total=num_epochs, desc='Training', unit='epoch') as pbar:
        for _ in range(num_epochs):
            train_loss_acc = train_epoch(model,
                                         train_dl,
                                         optimizer,
                                         scheduler
                                         )
            train_loss_all_epochs.append(train_loss_acc['loss_train_epoch'])
            train_accuracy_all_epochs.append(train_loss_acc['acc_train_epoch'])

            test_loss_acc = loss_accuracy_dataset(model, test_dl)
            test_loss_all_epochs.append(test_loss_acc['loss'])
            test_accuracy_all_epochs.append(test_loss_acc['acc'])

            pbar.update(1)
            pbar.set_postfix({'train_loss': train_loss_all_epochs[-1],
                              'train_acc': train_accuracy_all_epochs[-1],
                              'test_loss': test_loss_all_epochs[-1],
                              'test_acc': test_accuracy_all_epochs[-1]
                              })

    return {'train_loss': train_loss_all_epochs, 'train_acc': train_accuracy_all_epochs,
            'test_loss': test_loss_all_epochs, 'test_acc': test_accuracy_all_epochs}


def save_stats(stats_dict: Dict[str, List[float]], save_dir: str) -> None:
    """Saves the evaluation statistics.
    Args:
        stats_dict: The stats dictionary to be saved.
        save_dir: The directory to save the dict at.
    Returns:
        None.
    """
    torch.save(stats_dict, save_dir)
    logging.info('Saved the evaluation stats at %s', save_dir)


def plot_save_stats(stats_dict: Dict[str, List[float]], save_dir: str) -> None:
    """Plot accuracy and loss on the train and test datasets.
    Args:
        stats_dict: A dict with 'train_loss', 'train_acc', 'test_loss', 'test_acc' keys and the corresponding values.
        save_dir: The directory to save the plot.
    Returns:
        None.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))

    # plotting the train and test accuracies
    axes[0].plot(stats_dict['train_acc'], color='green', label='train', linewidth=2)
    axes[0].plot(stats_dict['test_acc'], color='blue', label='test', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # plotting the train and test losses
    axes[1].plot(stats_dict['train_loss'], color='green', label='train', linewidth=2)
    axes[1].plot(stats_dict['test_loss'], color='blue', label='test', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir)
    logging.info('Saved the visualization of the evaluation stats at %s', save_dir)
    # plt.show()


def main(argv):
    train_test_dl = create_dataset(_DATASET.value, _BATCH_SIZE.value)
    logging.info('Created the train and test datasets!')

    model = initialize_model(_MODEL.value, train_test_dl['num_classes'], device)
    logging.info('Initialized the model!')

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Model {_MODEL.value} contains {n_param} parameters ....')

    loss_acc_train_test = train(model,
                                train_test_dl['train_dl'],
                                train_test_dl['test_dl'],
                                _LR.value,
                                _WD.value,
                                _N_EPOCHS.value)

    logging.info('Trained the model and evaluated it on both train and test set!')

    logging.info('Saving the pretrained model...')

    save_dir = os.path.join(_BASE_DIR.value, 'pretrained_model_{}_{}.pth'.format(_MODEL.value, _DATASET.value))
    save_model(model, save_dir)

    logging.info('Saving the pretrained evaluation statistics ...')
    save_dir = os.path.join(_BASE_DIR.value, 'pretrained_stats_{}_{}.pth'.format(_MODEL.value, _DATASET.value))

    save_stats(loss_acc_train_test, save_dir)
    logging.info('Plotting the evaluation stats...')
    save_dir = os.path.join(_BASE_DIR.value, 'pretrained_stats_vis_{}_{}.pdf'.format(_MODEL.value, _DATASET.value))
    plot_save_stats(loss_acc_train_test, save_dir)


if __name__ == '__main__':
    app.run(main)
