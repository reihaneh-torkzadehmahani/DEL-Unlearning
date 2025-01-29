"""
MIT License

Copyright (c) 2025 Reihaneh Torkzadehmahani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from absl import flags, app
from itertools import cycle

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from utils import *
from unlearning_models import *
from unlearning_datasets import *

FLAGS = flags.FLAGS
_DATASET = flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'svhn', 'cifar100', 'imagenet100'], 'Dataset')
_MODEL = flags.DEFINE_enum('model', 'resnet18', ['resnet18', 'vit', 'resnet50'], 'Model')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'Batch Size')
_LR = flags.DEFINE_float('learning_rate', 0.015, 'Learning Rate')
_WD = flags.DEFINE_float('weight_decay', 0, 'Weight Decay')
_N_EPOCHS = flags.DEFINE_integer('epochs', 50, 'Number of Epochs')
_BASE_DIR = flags.DEFINE_string('base_dir', './data/', 'The base directory to save the model.')

_FORGET_DATA_DIR = flags.DEFINE_string('forget_data_dir',
                                       './data/cifar10_forget_indices.pth',
                                       'The directory of the forget indices and classes.')
_FORGET_RATIO = flags.DEFINE_float('forget_ratio', 0.1, 'Ratio of forget samples')
_FORGET_MODE = flags.DEFINE_enum('forget_mode', 'iid', ['iid', 'non-iid'], 'Mode of creating forget set.')
_FORGET_CLASSES = flags.DEFINE_list('forget_classes', [-1], 'Classes of forget samples')

_PRETRAINED_DIR = flags.DEFINE_string('pretrained_dir',
                                      './data/pretrained_model_resnet18_cifar10.pth',
                                      'The directory where the pretrained model is.')

_MASK_DIR = flags.DEFINE_string('mask_dir',
                                './data/mask_resnet18_cifar10_iid_0.1_weighted_grad_channel_thr_0.3.pt',
                                'The directory of mask for critical parameters.')
_UNLEARNING_ALG = flags.DEFINE_enum('unlearning_alg', 'reset+finetune',
                               ['negrad+', 'relabeling', 'reset+finetune', 'l1_sparse', 'wfisher', 'ga', 'ssd'],
                               'Unlearning algorithm.')

_RUN = flags.DEFINE_integer('run', 0, 'Run number')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def loss_accuracy_dataset(model: nn.Module, dataset: DataLoader) -> Dict[str, float]:
    """ Computes loss and accuracy of the model on a given set.
    Args:
        model: The model to be evaluated.
        dataset: The dataset to evaluate the model on.
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

def reset_params(mask: nn.Module, model: nn.Module, init_model: nn.Module):
    reinit_model = copy.deepcopy(model)
    with torch.no_grad():
        for (name, param), (init_name, init_param) in zip(reinit_model.named_parameters(),
                                                          init_model.named_parameters()):
            param.data = (param.data * (1 - mask[name])) + (init_param.data * mask[name])

    return reinit_model


def fine_tune_epoch(model: nn.Module,
                    retain_dl: DataLoader,
                    forget_dl: DataLoader,
                    optimizer: optim.Optimizer,
                    scheduler: lr_scheduler,
                    mask=None,
                    unlearning_alg='reset+finetune',
                    epoch=0,
                    num_epochs=0
                    ) -> Dict[str, float]:
    """Fine_tunes the model for only one epoch.
    Args:
        model: The model.
        retain_dl: The training data loader.
        forget_dl: The forget data loader.
        optimizer: The optimizer.
        scheduler: Learning rate scheduler.
        mask: Mask of critical parameters.
        unlearning_alg: The unlearning algorithm (e.g. 'reset+finetune').
        epoch: Current epoch.
        num_epochs: Total number of epochs.
    Returns:
        A dict with 'loss_train_epoch' and 'acc_retian_epoch' keys whose values are loss and accuracy on the retain set
        in the current epoch.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    acc_all_batches = []
    num_all_samples = 0
    sum_loss = 0

    for (indices, images, labels), (_, forget_images, forget_labels) in zip(retain_dl, cycle(forget_dl)):
        images, labels = images.to(device), labels.to(device)
        forget_images, forget_labels = forget_images.to(device), forget_labels.to(device)

        logits = model(images)
        loss_retain = loss_fn(logits, labels)

        if unlearning_alg == 'relabeling' or unlearning_alg == 'reset+finetune':
            loss_batch = loss_retain
        elif unlearning_alg == 'ga':
            forget_logits = model(forget_images)
            loss_forget = loss_fn(forget_logits, forget_labels)
            loss_batch = -loss_forget
        elif unlearning_alg == 'negrad+':
            forget_logits = model(forget_images)
            loss_forget = loss_fn(forget_logits, forget_labels)
            alpha = 0.99
            loss_batch = (alpha * loss_retain) - ((1 - alpha) * loss_forget)
        elif unlearning_alg == 'l1_sparse':
            alpha = 7e-5
            current_alpha = alpha * (1 - epoch / num_epochs)

            params_vec = []
            for param in model.parameters():
                params_vec.append(param.view(-1))
            l1_regularization = torch.linalg.norm(torch.cat(params_vec), ord=1)

            loss_batch = loss_retain + (current_alpha * l1_regularization)

        optimizer.zero_grad()
        loss_batch.backward()

        if mask is not None:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()

        sum_loss += loss_batch * len(labels)

        prediction = torch.argmax(logits, 1)
        acc_all_batches.append((prediction == labels).sum().item())
        num_all_samples += len(labels)

    scheduler.step()
    loss_epoch = sum_loss / num_all_samples
    acc_epoch = sum(acc_all_batches) / num_all_samples
    return {'loss_retain_epoch': loss_epoch.item(), 'acc_retain_epoch': acc_epoch}

def fine_tune(model: nn.Module,
              retain_dl: DataLoader,
              test_dl: DataLoader,
              forget_dl: DataLoader,
              lr: float,
              weight_decay: float,
              num_epochs: int,
              update_layers: List[str] = [],
              mask=None,
              unlearning_alg='reset+finetune'
              ) -> Dict[str, List[float]]:
    """Trains the model.
        Args:
            model: The model.
            retain_dl: The retain data loader.
            forget_dl: The forget data loader.
            test_dl: Test dataset used for evaluation.
            lr: Learning rate.
            weight_decay: Weight decay.
            num_epochs: Number of training epochs.
            update_layers: Layers to update and not freeze.
            mask: Mask of critical parameters.
            unlearning_alg: The unlearning algorithm (e.g. 'reset+finetune').

        Returns:
            A dict with 'retain_loss', 'retain_acc', 'test_loss','forget_loss', 'forget_acc', 'test_acc' keys whose
            values are loss and accuracy on the train and test sets respectively.
    """
    retain_loss_all_epochs = []
    retain_accuracy_all_epochs = []
    forget_loss_all_epochs = []
    forget_accuracy_all_epochs = []
    test_loss_all_epochs = []
    test_accuracy_all_epochs = []
    all_steps = num_epochs
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if  unlearning_alg == 'l1_sparse' or unlearning_alg == 'reset+finetune':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=all_steps, eta_min=0.01 * lr)
    elif unlearning_alg == 'relabeling':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=all_steps, eta_min=0.5 * lr)
    elif unlearning_alg == 'negrad+' or unlearning_alg == 'ga':
        constant_lr = lambda epoch: 1.0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=constant_lr, verbose=True)
    # Computes the retain, forget and test statistics right after resetting
    retain_loss_acc = loss_accuracy_dataset(model, retain_dl)
    retain_loss_all_epochs.append(retain_loss_acc['loss'])
    retain_accuracy_all_epochs.append(retain_loss_acc['acc'])

    test_loss_acc = loss_accuracy_dataset(model, test_dl)
    test_loss_all_epochs.append(test_loss_acc['loss'])
    test_accuracy_all_epochs.append(test_loss_acc['acc'])

    forget_loss_acc = loss_accuracy_dataset(model, forget_dl)
    forget_loss_all_epochs.append(forget_loss_acc['loss'])
    forget_accuracy_all_epochs.append(forget_loss_acc['acc'])

    if len(update_layers) > 0:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            name_no_weight_no_bias = name
            name_no_weight_no_bias = name_no_weight_no_bias.rstrip('.weight')
            name_no_weight_no_bias = name_no_weight_no_bias.rstrip('.bias')
            if name_no_weight_no_bias in update_layers:
                param.requires_grad = True

    with tqdm(total=num_epochs, desc='Training', unit='epoch') as pbar:
        for epoch in range(num_epochs):
            retain_loss_acc = fine_tune_epoch(model,
                                              retain_dl,
                                              forget_dl,
                                              optimizer,
                                              scheduler,
                                              mask=mask,
                                              unlearning_alg=unlearning_alg,
                                              epoch=epoch,
                                              num_epochs=num_epochs
                                              )
            retain_loss_all_epochs.append(retain_loss_acc['loss_retain_epoch'])
            retain_accuracy_all_epochs.append(retain_loss_acc['acc_retain_epoch'])

            test_loss_acc = loss_accuracy_dataset(model, test_dl)
            test_loss_all_epochs.append(test_loss_acc['loss'])
            test_accuracy_all_epochs.append(test_loss_acc['acc'])

            forget_loss_acc = loss_accuracy_dataset(model, forget_dl)
            forget_loss_all_epochs.append(forget_loss_acc['loss'])
            forget_accuracy_all_epochs.append(forget_loss_acc['acc'])

            pbar.update(1)
            pbar.set_postfix({'train': retain_accuracy_all_epochs[-1],
                              'test': test_accuracy_all_epochs[-1],
                              'forget': forget_accuracy_all_epochs[-1],
                              })

    return {'retain_loss': retain_loss_all_epochs, 'retain_acc': retain_accuracy_all_epochs,
            'test_loss': test_loss_all_epochs, 'test_acc': test_accuracy_all_epochs,
            'forget_loss': forget_loss_all_epochs, 'forget_acc': forget_accuracy_all_epochs,}


def main(argv):
    forget_retain_test_dl = create_unlearning_dataset(_DATASET.value, _BATCH_SIZE.value,
                                                      _FORGET_RATIO.value, _FORGET_MODE.value, _FORGET_CLASSES.value,
                                                      _FORGET_DATA_DIR.value)

    logging.info('Created the retain, forget, train and test datasets!')

    model, init_model = load_model(_MODEL.value, forget_retain_test_dl['num_classes'], _PRETRAINED_DIR.value, device)
    logging.info('Loaded the pretrained model!')

    # ------------------------------- Retrain-free Unlearning Algorithms: WFisher(IU) and SSD --------------------------
    if _UNLEARNING_ALG.value == 'wfisher':
        pretrained_model = copy.deepcopy(model)
        # -- for cifar10/resnet18 IID alpha=42000, non-IID: alpha=1900
        # -- for svhn/vit IID alpha = 4000, non-IID: alpha=6000
        # -- for imagenet100/resnet50 IID alpha=2000
        alpha = 2000
        criterion = nn.CrossEntropyLoss()
        reinitialized_model = Wfisher(
            {'forget': forget_retain_test_dl['forget_ds'], 'retain': forget_retain_test_dl['retain_ds']},
            pretrained_model,
            criterion,
            alpha,
            _BATCH_SIZE.value, device)

        loss_acc_test = loss_accuracy_dataset(reinitialized_model, forget_retain_test_dl['test_dl'])
        loss_acc_forget = loss_accuracy_dataset(reinitialized_model, forget_retain_test_dl['forget_dl'])
        loss_acc_retain = loss_accuracy_dataset(reinitialized_model, forget_retain_test_dl['retain_dl'])
        loss_acc_retain_test_forget = {'retain_loss': [loss_acc_test['loss']], 'retain_acc': [loss_acc_retain['acc']],
                                       'test_loss': [loss_acc_test['loss']], 'test_acc': [loss_acc_test['acc']],
                                       'forget_loss': [loss_acc_forget['loss']], 'forget_acc': [loss_acc_forget['acc']]}
    elif _UNLEARNING_ALG.value == 'ssd':
        pretrained_model = copy.deepcopy(model)
        # -- n_forget = 100 : selection_weighting = 5, n_forget = 5000,
        # -- (cifar10 dampening = 1 and selection weighting = 1 for iid, 16 for non-iid)
        # -- (svhn dampening = 3, selection_weighting = 5) (imagenet dampening = 1, selection weighting = 2)
        dampening_constant = 1
        selection_weighting = 2

        parameters = {
            "lower_bound": 1,
            "exponent": 1,
            "magnitude_diff": None,
            "min_layer": -1,
            "max_layer": -1,
            "forget_threshold": 1,
            "dampening_constant": dampening_constant,
            "selection_weighting": selection_weighting, }
        # load the trained model
        optimizer = torch.optim.SGD(pretrained_model.parameters(), lr=_LR.value, momentum=0.9, weight_decay=0)
        pdr = ParameterPerturber(pretrained_model, optimizer, device, parameters)
        pretrained_model = pretrained_model.eval()
        sample_importances = pdr.calc_importance(forget_retain_test_dl['forget_dl'])
        original_importances = pdr.calc_importance(forget_retain_test_dl['train_dl'])
        pdr.modify_weight(original_importances, sample_importances)
        reinitialized_model = copy.deepcopy(pretrained_model)

        loss_acc_test = loss_accuracy_dataset(reinitialized_model, forget_retain_test_dl['test_dl'])
        loss_acc_forget = loss_accuracy_dataset(reinitialized_model, forget_retain_test_dl['forget_dl'])
        loss_acc_retain = loss_accuracy_dataset(reinitialized_model, forget_retain_test_dl['retain_dl'])
        loss_acc_retain_test_forget = {'retain_loss': [loss_acc_test['loss']], 'retain_acc': [loss_acc_retain['acc']],
                                       'test_loss': [loss_acc_test['loss']], 'test_acc': [loss_acc_test['acc']],
                                       'forget_loss': [loss_acc_forget['loss']], 'forget_acc': [loss_acc_forget['acc']]}
        logging.info(f'The evaluation metric for SSD are : {loss_acc_retain_test_forget}')

    # ------------------------------------------------------------------------------------------------------------------

    else:
        mask_dir = _MASK_DIR.value
        salun_mask = torch.load(mask_dir, weights_only=True)
        logging.info('Loaded the mask from %s', mask_dir)
        # --------------------- Unlearning steps : reset critical params, finetune only on the retain
        if _UNLEARNING_ALG.value == 'reset+finetune':
            reinitialized_model = reset_params(salun_mask, model, init_model)
            desired_layers_keys = []
            grad_mask = salun_mask
            for name, param in model.named_parameters():
                if 'fc' in name or 'mlp_head' in name:
                    logging.info('Setting mask of linear probe to 1!')
                    grad_mask[name] = torch.ones(grad_mask[name].shape).to(device)
        else:
            logging.info('Not resetting the params and no freezing (update all layers)!!!!')
            reinitialized_model = model
            desired_layers_keys = []
            grad_mask = None

        if _UNLEARNING_ALG.value == 'relabeling':
            logging.info('Relabeling is applied!!!!')
            #### Unlearning steps : relabel F, create F`+R=T`,finetune on T` while grads=grads*mask (update only critical)
            #### Step 1: relabel the forget set s.t. all the labels are flipped
            forget_dataset = copy.deepcopy(forget_retain_test_dl['forget_dl'].dataset)
            if _DATASET.value == 'cifar10' or _DATASET.value == 'svhn':
                num_classes = 10
                forget_dataset.dataset.targets = np.random.randint(0, num_classes, len(forget_dataset.dataset.targets))
            elif dataset_name == 'imagenet100':
                forget_dataset = ImageNet100Random(forget_dataset)
            # step 2: create the new train loader containing the relabeled forget set and the retain set
            retain_dataset = forget_retain_test_dl['retain_dl'].dataset
            train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
            fine_tune_loader = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=_BATCH_SIZE.value,
                                                           shuffle=True,
                                                           num_workers=10)
            # step 3: finetune the new model on the new training loader while only finetuning the params with value 1 in mask
        else:
            fine_tune_loader = forget_retain_test_dl['retain_dl']

        loss_acc_retain_test_forget = fine_tune(reinitialized_model,
                                                fine_tune_loader,
                                                forget_retain_test_dl['test_dl'],
                                                forget_retain_test_dl['forget_dl'],
                                                _LR.value,
                                                _WD.value,
                                                _N_EPOCHS.value,
                                                update_layers=desired_layers_keys,
                                                mask=grad_mask,
                                                unlearning_alg=_UNLEARNING_ALG.value,
                                                )

    unlearning_model = copy.deepcopy(reinitialized_model)
    print('Unlearning forget accuracy is', loss_accuracy_dataset(unlearning_model,
                                                                 forget_retain_test_dl['forget_dl']))

    logging.info('Fine_tuned the model and evaluated it on retain, test and forget set!')
    # =================================================================================================================

    logging.info('Saving the unlearning evaluation statistics and model ...')
    save_dir = os.path.join(_BASE_DIR.value,
                            '{}_{}_reset_finetune_stats_{}_{}_run_{}.pth'.format(
                                _MODEL.value,
                                _DATASET.value,
                                _FORGET_MODE.value,
                                _UNLEARNING_ALG.value,
                                _RUN.value))

    save_dir_model = os.path.join(_BASE_DIR.value,
                                  '{}_{}_reset_finetune_model_{}_{}_run_{}.pth'.format(
                                      _MODEL.value,
                                      _DATASET.value,
                                      _FORGET_MODE.value,
                                      _UNLEARNING_ALG.value,
                                      _RUN.value))
    save_stats(loss_acc_retain_test_forget, save_dir)
    save_model(unlearning_model, save_dir_model)


if __name__ == '__main__':
    app.run(main)
