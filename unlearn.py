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
_LR = flags.DEFINE_float('learning_rate', 0.1, 'Learning Rate')
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
                                      './data/base/pretrained_model_resnet18_cifar10_noisy_indices_0.pth',
                                      'The directory where the pretrained model is.')

_RUN = flags.DEFINE_integer('run', 0, 'Run number')

#  =================================================================================================

_THRESHOLD = flags.DEFINE_float('threshold', 0.16874820229, 'Salun threshold')
_SALUN_BASE_NAME = flags.DEFINE_string('salun_base_name',
                                       'our',
                                       'The base name for salun mask.')

_RESET_METHOD = flags.DEFINE_enum('reset_method',
                                  'init', ['init', 'perturb', 'init_random', 'none_random', 'none'],
                                  'Parameter resetting strategy.')

_UPDATE_LAYERS = flags.DEFINE_enum('update_layers',
                                   'critical', ['all', 'fc', 'critical', 'critical_fc'],
                                   'Layers to update (the rest will be frozen).')
_RELABEL = flags.DEFINE_bool('relabel', False, 'applying relabeling on forget set(salun unlearning algo)')

_ALPHA = flags.DEFINE_float('alpha', 0.7, 'perturbation alpha')

# _GRAD_MODE = flags.DEFINE_enum('grad_mode', 'descent',
#                                ['negrad+', 'descent', 'l1_sparse', 'wfisher', 'ga', 'ssd'],
#                                'The forget set mode.')
_GRAD_MODE = flags.DEFINE_enum('grad_mode', 'ssd',
                               ['negrad+', 'descent', 'l1_sparse', 'wfisher', 'ga', 'ssd'],
                               'The forget set mode.')
np.random.seed(0)
random.seed(25)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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


def reset_params(mask: nn.Module, model: nn.Module, init_model: nn.Module):
    reinit_model = copy.deepcopy(model)
    with torch.no_grad():
        for (name, param), (init_name, init_param) in zip(reinit_model.named_parameters(),
                                                          init_model.named_parameters()):
            param.data = (param.data * (1 - mask[name])) + (init_param.data * mask[name])

    return reinit_model


def perturb_params(mask: nn.Module, model: nn.Module, init_model: nn.Module,
                   alpha: float = 1.0):
    reinit_model = copy.deepcopy(model)
    all_reset_params = 0
    with torch.no_grad():
        for (name, param), (init_name, init_param) in zip(reinit_model.named_parameters(),
                                                          init_model.named_parameters()):
            # mean_param = init_param.data.mean()
            # var_param = init_param.data.var()
            # rnd_param = torch.normal(mean=mean_param, std=torch.sqrt(var_param), size=init_param.size(), device=init_param.device)
            # rnd_param = init_param.data
            rnd_param = torch.normal(mean=0, std=1, size=init_param.size(), device=init_param.device)
            param.data = (param.data * (1 - mask[name])) + (
                    ((alpha * param.data) + ((1 - alpha) * rnd_param)) * mask[name])

    return reinit_model


def count_params(layer_names: list[str], num_channels: list[int], model: nn.Module):
    """

    :param layer_names:
    :param num_channels:
    :param model:
    :return:
        num_params: total numebr of parameters for the input list
        sample_params_dict: a dict whose keys is all the layer names and values is a list of all the channels for that layer
        num_params_list: List of number of parameters for each element of input lists
    """
    copy_model = copy.deepcopy(model)
    num_params = 0
    num_params_list = []
    sample_params_dict = {}
    for name, param in copy_model.named_parameters():
        if name in layer_names:
            indices = [index for index, layer in enumerate(layer_names) if layer == name]
            sample_params_dict[name] = []
            for i in indices:
                if isinstance(num_channels[i][0], torch.Tensor):
                    param_ind = num_channels[i][0].item()
                else:
                    param_ind = num_channels[i][0]
                sample_params_dict[name].append(param_ind)
                if 'fc' in name:
                    params_i = param[:, param_ind].numel()
                    num_params += params_i
                else:
                    params_i = param[param_ind].numel()
                    num_params += params_i
                num_params_list.append(params_i)

    return num_params, sample_params_dict, num_params_list


def fine_tune_epoch(model: nn.Module,
                    retain_dl: DataLoader,
                    forget_dl: DataLoader,
                    optimizer: optim.Optimizer,
                    scheduler: lr_scheduler,
                    mask=None,
                    grad_mode='descent',
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
    Returns:
        A dict with 'loss_train', 'accuracy_train', 'loss_test', 'accuracy_test' keys whose values are
        loss and accuracy on the train and test test respectively.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    acc_all_batches = []
    num_all_samples = 0
    sum_loss = 0

    print(len(retain_dl))
    print(len(forget_dl))
    for (indices, images, labels), (_, forget_images, forget_labels) in zip(retain_dl, cycle(forget_dl)):
        # TODO: the original version uses zip(retian_dl, cycle(forget_dl))
        images, labels = images.to(device), labels.to(device)
        forget_images, forget_labels = forget_images.to(device), forget_labels.to(device)

        logits = model(images)
        loss_retain = loss_fn(logits, labels)

        if grad_mode == 'descent' or grad_mode == 'ft':
            loss_batch = loss_retain
        elif grad_mode == 'ga':
            forget_logits = model(forget_images)
            loss_forget = loss_fn(forget_logits, forget_labels)
            loss_batch = -loss_forget
        elif grad_mode == 'negrad+':
            forget_logits = model(forget_images)
            loss_forget = loss_fn(forget_logits, forget_labels)
            ## class 25 5k
            alpha = 0.99
            ## all 5k
            # alpha= 0.95
            loss_batch = (alpha * loss_retain) - ((1 - alpha) * loss_forget)
        elif grad_mode == 'l1_sparse':
            # TODO
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


def loss_accuracy_dataset(model: nn.Module, dataset: DataLoader) -> Dict[str, float]:
    """ Computes loss and accuracy of the model on a given set.
    Args:
        model: The model to be evaluated.
        dataset: The dataset to evaluated the model on.
    Returns:
        A dict with 'loss' and 'acc' as the keys whose values are the loss and accuracy of the model on the dataset.
    """
    acc_all_batches = []
    num_all_samples = 0
    model.eval()
    sum_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for _, images, labels in dataset:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_batch = loss_fn(logits, labels)
            sum_loss += loss_batch * len(labels)

            prediction = torch.argmax(logits, 1)
            acc_all_batches.append((prediction == labels).sum().item())
            num_all_samples += len(labels)

    loss_epoch = sum_loss / num_all_samples
    acc_epoch = sum(acc_all_batches) / num_all_samples
    return {'loss': loss_epoch.item(), 'acc': acc_epoch}


def fine_tune(model: nn.Module,
              retain_dl: DataLoader,
              test_dl: DataLoader,
              forget_dl: DataLoader,
              lr: float,
              weight_decay: float,
              num_epochs: int,
              noisy_forget_dl: DataLoader = None,
              forget_data_dir: str = None,
              pred_depth_dir: str = None,
              update_layers: List[str] = [],
              mask=None,
              relabel=False,
              grad_mode='descent'
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
            noisy_forget_dl: The noisy forget data loader.
            update_layers: Layers to update and not freeze.
            forget_data_dir: Path to the forget indices.
            pred_depth_dir: Path to the prediction depths so that we can evaluate the finetuned model on different
                            subgroups made according to prediction depth.
        Returns:
            A dict with 'retain_loss', 'retain_acc', 'test_loss','forget_loss', 'forget_acc', 'test_acc' keys whose
            values are loss and accuracy on the train and test test respectively.
    """
    retain_loss_all_epochs = []
    retain_accuracy_all_epochs = []
    forget_loss_all_epochs = []
    forget_accuracy_all_epochs = []
    noisy_forget_loss_all_epochs = []
    noisy_forget_accuracy_all_epochs = []
    test_loss_all_epochs = []
    test_accuracy_all_epochs = []
    pd_accuracy_all_epochs = {}
    all_steps = num_epochs
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if not relabel and (grad_mode == 'descent' or grad_mode == 'l1_sparse'):
        # TODO------------
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=all_steps, eta_min=0.01 * lr)
    elif relabel:
        # constant_lr = lambda epoch: 1.0
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=constant_lr, verbose=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=all_steps, eta_min=0.5 * lr, verbose=True)
    elif grad_mode == 'negrad+' or grad_mode == 'ga':
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
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            name_no_weight_no_bias = name
            name_no_weight_no_bias = name_no_weight_no_bias.rstrip('.weight')
            name_no_weight_no_bias = name_no_weight_no_bias.rstrip('.bias')
            if name_no_weight_no_bias in update_layers:
                param.requires_grad = True
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    with tqdm(total=num_epochs, desc='Training', unit='epoch') as pbar:
        for epoch in range(num_epochs):
            retain_loss_acc = fine_tune_epoch(model,
                                              retain_dl,
                                              forget_dl,
                                              optimizer,
                                              scheduler,
                                              mask=mask,
                                              grad_mode=grad_mode,
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
            'forget_loss': forget_loss_all_epochs, 'forget_acc': forget_accuracy_all_epochs,
            'noisy_forget_loss': noisy_forget_loss_all_epochs, 'noisy_forget_acc': noisy_forget_accuracy_all_epochs,
            'pred_depth_acc': pd_accuracy_all_epochs}


def main(argv):
    forget_retain_test_dl = create_unlearning_dataset(_DATASET.value, _BATCH_SIZE.value,
                                                      _FORGET_RATIO.value, _FORGET_MODE.value, _FORGET_CLASSES.value,
                                                      _FORGET_DATA_DIR.value)

    logging.info('Created the retain, forget, train and test datasets!')

    model, init_model = load_model(_MODEL.value, forget_retain_test_dl['num_classes'], _PRETRAINED_DIR.value, device)
    logging.info('Loaded the pretrained model!')

    # ------------------------------- Retrain-free Unlearning Algorithms: WFisher(IU) and SSD --------------------------
    if _GRAD_MODE.value == 'wfisher':
        pretrained_model = copy.deepcopy(model)
        # -- for cifar10/resnet18 IID alpha=42000, non-IID: alpha=1900
        # -- for svhn/vit IID alpha = 4000, non-IID: alpha=6000
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
    elif _GRAD_MODE.value == 'ssd':
        pretrained_model = copy.deepcopy(model)
        # -- n_forget = 100 : selection_weighting = 5, n_forget = 5000,
        # -- (cifar10 dampening = 1 and selection weighting = 1 for iid, 16 for non-iid)??
        # -- (svhn dampening = 3, selection_weighting = 5)
        # selection_weighting = 16
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
        salun_mask_name = _SALUN_BASE_NAME.value + '_with_' + str(_THRESHOLD.value) + '.pt'
        salun_mask = torch.load(salun_mask_name)
        logging.info('Loaded Salun mask from %s', salun_mask_name)
        # --------------------- Unlearning steps : reset critical params, finetune only on the retain
        if _RESET_METHOD.value == 'init':
            reinitialized_model = reset_params(salun_mask, model, init_model)
        elif _RESET_METHOD.value == 'perturb':
            reinitialized_model = perturb_params(salun_mask, model, init_model, _ALPHA.value)
        elif 'random' in _RESET_METHOD.value:
            salun_params_per_layer = {}
            random_mask = {}
            all_random_params = 0
            all_mask_params = 0
            for name, param in model.named_parameters():
                print(salun_mask[name].ndim)
                dims_to_sum = list(range(0, salun_mask[name].ndim))
                salun_params_per_layer[name] = torch.sum(salun_mask[name].cpu(), dims_to_sum)
                all_mask_params += salun_params_per_layer[name]
                random_mask[name] = torch.zeros(param.shape).to(device)
                random_mask_layer_params = 0
                all_output_channels = list(range((param.shape[0])))
                while random_mask_layer_params < salun_params_per_layer[name]:
                    random_ch = random.sample(all_output_channels, 1)[0]
                    all_output_channels.remove(random_ch)
                    random_mask[name][random_ch] = torch.ones(param[random_ch].shape).to(device)
                    random_mask_layer_params += torch.sum(random_mask[name][random_ch].cpu())
                all_random_params += random_mask_layer_params

            print('All random params are :', all_random_params)
            print('All input mask params were: ', all_mask_params)
            salun_mask = random_mask
            if 'init' in _RESET_METHOD.value:
                reinitialized_model = reset_params(random_mask, model, init_model)
            elif 'none' in _RESET_METHOD.value:
                reinitialized_model = model
        elif _RESET_METHOD.value == 'none':
            print('Not resetting the params!!!!')
            reinitialized_model = model

        if _UPDATE_LAYERS.value == 'all':
            desired_layers_keys = []
            grad_mask = None
        elif _UPDATE_LAYERS.value == 'fc':
            if model_name == 'resnet18':
                desired_layers_keys = 'resnet18.fc'
                grad_mask = None
            else:
                raise ValueError('update fc not implemented for model'.format(model_name))
        elif _UPDATE_LAYERS.value == 'critical':
            desired_layers_keys = []
            grad_mask = salun_mask
        elif _UPDATE_LAYERS.value == 'critical_fc':
            desired_layers_keys = []
            grad_mask = salun_mask
            # grad_mask = random_mask
            for name, param in model.named_parameters():
                if 'fc' in name or 'mlp_head' in name:
                    print('Setting mask of linear probel to 1!')
                    grad_mask[name] = torch.ones(grad_mask[name].shape).to(device)
        else:
            raise ValueError('Resetting strategy or update layer are not valid!!!!!!')
        logging.info('The layers that are being %s are %s', _RESET_METHOD.value, _UPDATE_LAYERS.value)
        logging.info('Computing the prediction depths....')
        pred_depth_dir = './data/base/prediction_depth'
        # forget_data_dir = './data/forget_indices_vertical_0.pth'
        forget_data_dir = _FORGET_DATA_DIR.value

        if _RELABEL.value:
            print('relabeling is avtive!!!!')
            #### Unlearning steps : relabel F, create F`+R=T`,finetune on T` while grads=grads*mask (update only critical)
            #### Step 1: relabel the forget set s.t. all the labels are flipped
            forget_dataset = copy.deepcopy(forget_retain_test_dl['clean_forget_dl'].dataset)
            if dataset_name == 'cifar10' or dataset_name == 'svhn':
                forget_dataset.dataset.targets = np.random.randint(0, num_classes, len(forget_dataset.dataset.targets))
                print(torch.unique(torch.tensor(forget_dataset.dataset.targets), return_counts=True))
            elif dataset_name == 'imagenet100':
                print(forget_dataset.dataset)
                print(len(forget_dataset))
                forget_dataset = TinyImageNetRandom(forget_dataset)
            # step 2: create the new train loader containing the relabeled forget set and the retain set
            retain_dataset = forget_retain_test_dl['retain_dl'].dataset
            train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
            fine_tune_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_BATCH_SIZE.value, shuffle=True,
                                                           num_workers=10)
            # step 3: fine tune the new model on the new training loader while only finetuning the params with value 1 in mask
        else:
            fine_tune_loader = forget_retain_test_dl['retain_dl']

        print(_GRAD_MODE.value)
        loss_acc_retain_test_forget = fine_tune(reinitialized_model,
                                                fine_tune_loader,
                                                forget_retain_test_dl['test_dl'],
                                                forget_retain_test_dl['forget_dl'],
                                                _LR.value,
                                                _WD.value,
                                                _N_EPOCHS.value,
                                                forget_retain_test_dl['noisy_forget_dl'],
                                                forget_data_dir,
                                                pred_depth_dir,
                                                update_layers=desired_layers_keys,
                                                mask=grad_mask,
                                                relabel=_RELABEL.value,
                                                grad_mode=_GRAD_MODE.value,
                                                )

    unlearning_model = copy.deepcopy(reinitialized_model)
    print('Unlearning forget accuracy is', loss_accuracy_dataset(unlearning_model,
                                                                 forget_retain_test_dl['forget_dl']))

    logging.info('Fine_tuned the model and evaluated it on retain, test and forget set!')
    # =================================================================================================================

    logging.info('Saving the unlearning evaluation statistics and model ...')
    save_dir = os.path.join(_BASE_DIR.value,
                            '{}_{}_reset_finetune_stats_{}_{}_{}_{}_{}_relabel_{}_{}_run_{}.pth'.format(
                                _SALUN_BASE_NAME.value,
                                _THRESHOLD.value,
                                _MODEL.value,
                                _DATASET.value,
                                _FORGET_MODE.value,
                                _RESET_METHOD.value,
                                _UPDATE_LAYERS.value,
                                _RELABEL.value,
                                _GRAD_MODE.value,
                                _RUN.value))

    save_dir_model = os.path.join(_BASE_DIR.value,
                                  '{}_{}_reset_finetune_model_{}_{}_{}_{}_{}_relabel_{}_{}_run_{}.pth'.format(
                                      _SALUN_BASE_NAME.value,
                                      _THRESHOLD.value,
                                      _MODEL.value,
                                      _DATASET.value,
                                      _FORGET_MODE.value,
                                      _RESET_METHOD.value,
                                      _UPDATE_LAYERS.value,
                                      _RELABEL.value,
                                      _GRAD_MODE.value,
                                      _RUN.value))
    save_stats(loss_acc_retain_test_forget, save_dir)
    save_model(unlearning_model, save_dir_model)


if __name__ == '__main__':
    app.run(main)
