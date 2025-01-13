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

from absl import flags, app

from collections import OrderedDict

import torch.optim
import torch.utils.data

from unlearning_models import *
from unlearning_datasets import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
FLAGS = flags.FLAGS
_DATASET = flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'svhn', 'cifar100', 'imagenet100'], 'Dataset')
_MODEL = flags.DEFINE_enum('model', 'resnet18', ['resnet18', 'vit', 'resnet50'], 'Model')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'Batch Size')
_LR = flags.DEFINE_float('learning_rate', 0.1, 'Learning Rate')
_WD = flags.DEFINE_float('weight_decay', 0, 'Weight Decay')
_N_EPOCHS = flags.DEFINE_integer('epochs', 50, 'Number of Epochs')
_BASE_DIR = flags.DEFINE_string('mask_save_dir', './data', 'The base directory to save the mask.')

_FORGET_DATA_DIR = flags.DEFINE_string('forget_data_dir',
                                       './data/cifar10_forget_indices.pth',
                                       'The directory of the forget indices and classes.')
_FORGET_RATIO = flags.DEFINE_float('forget_ratio', 0.1, 'Ratio of forget samples')
_FORGET_MODE = flags.DEFINE_enum('forget_mode', 'iid', ['iid', 'non-iid'], 'Mode of creating forget set.')
_FORGET_CLASSES = flags.DEFINE_list('forget_classes', [-1], 'Classes of forget samples')

_PRETRAINED_DIR = flags.DEFINE_string('pretrained_dir',
                                      './data/base/pretrained_model_resnet18_cifar10_noisy_indices_0.pth',
                                      'The directory where the pretrained model is.')
_CRITIC_CRITERIA = flags.DEFINE_enum('critic_criteria', 'grad',
                                     ['grad', 'weight', 'weighted_grad', 'weighted_grad_all'],
                                     'Criticality criteria for the model params. (salun: grad, ours: weighted grad)')
_GRANULARITY = flags.DEFINE_enum('granularity', 'param', ['channel', 'param'],
                                 'Granularity of  critical model params. (salun: params, ours: channel)')
_THRESHOLD = flags.DEFINE_list('threshold', None, 'Threshold for the ratio of parameters for the mask.')

_RUN = flags.DEFINE_integer('run', 0, 'Run number')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_gradient_ratio(data_loaders, model, criterion, mask_save_dir, mode='grad', granularity='channel',
                        threshold_list=(0.3, 1.0)):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    gradients = {}
    if mode == 'weighted_grad_all':
        forget_loader = data_loaders['train']
    else:
        forget_loader = data_loaders["forget"]
    model.eval()
    for name, param in model.named_parameters():
        gradients[name] = 0
    for i, (_, image, target) in enumerate(forget_loader):
        image = image.cuda()
        target = target.cuda()
        # compute output
        output_clean = model(image)
        loss = - criterion(output_clean, target)
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if mode == 'grad':
                        gradients[name] += param.grad.data
                    elif mode == 'weighted_grad' or mode == 'weighted_grad_all':
                        gradients[name] += param.data * param.grad.data
                    elif mode == 'weight':
                        gradients[name] += param.data

    if granularity == 'param':
        with torch.no_grad():
            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])
    elif granularity == 'channel':
        # ---------------------- Our mask : Per channel experiments, sum, l1 per channel -------------------------------
        grads_ch_out = dict()
        with torch.no_grad():
            for name in gradients:
                num_dims = gradients[name].ndim
                if num_dims > 1:
                    flatten_grads = torch.abs_(gradients[name]).view(gradients[name].shape[0], -1)

                    if _MODEL.value == 'resnet18' or _MODEL.value == 'resnet50':
                        top5_values, _ = torch.topk(flatten_grads, int(0.1 * len(flatten_grads)), dim=1)
                        grads_ch_out[name] = top5_values.mean(dim=1)
                    elif _MODEL.value == 'vit':
                        grads_ch_out[name] = flatten_grads.mean(dim=1)
                    else:
                        logging.info(f'Model {_MODEL.value} is not supported!!')
                        exit()
                else:
                    grads_ch_out[name] = torch.max(torch.abs_(gradients[name]))

        with torch.no_grad():
            for name in gradients:
                num_dims = gradients[name].ndim
                if num_dims > 1:
                    for i in range(grads_ch_out[name].shape[0]):
                        gradients[name][i, :] = grads_ch_out[name][i]
        #  -------------------------------------------------------------------------------------------------------------
    else:
        logging.info(f'Granularity {granularity} is not supported!!')

    for i in threshold_list:
        i = float(i)
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index: start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements
        save_dir = os.path.join(_BASE_DIR.value, mask_save_dir.format(i))
        torch.save(hard_dict, save_dir)
        logging.info('Saved the mask at: %s', save_dir)


def main(argv):
    print(_THRESHOLD.value)
    forget_retain_test_dl = create_unlearning_dataset(_DATASET.value, _BATCH_SIZE.value,
                                                      _FORGET_RATIO.value, _FORGET_MODE.value, _FORGET_CLASSES.value,
                                                      _FORGET_DATA_DIR.value)

    logging.info('Created the retain, forget, train and test datasets!')

    model, init_model = load_model(_MODEL.value, forget_retain_test_dl['num_classes'], _PRETRAINED_DIR.value, device)
    logging.info('Loaded the pretrained model!')

    tmp_save_dir = "mask_{}_{}_{}_{}_with_".format(_FORGET_MODE.value,
                                                   _FORGET_RATIO.value,
                                                   _CRITIC_CRITERIA.value,
                                                   _GRANULARITY.value)
    mask_save_dir = tmp_save_dir + "{}.pt"

    unlearn_data_loaders = OrderedDict(
        retain=forget_retain_test_dl['retain_dl'], forget=forget_retain_test_dl['forget_dl'],
        test=forget_retain_test_dl['test_dl'], train=forget_retain_test_dl['train_dl'],
    )

    criterion = nn.CrossEntropyLoss()
    save_gradient_ratio(unlearn_data_loaders, model, criterion, mask_save_dir,
                        _CRITIC_CRITERIA.value, _GRANULARITY.value, _THRESHOLD.value)


if __name__ == '__main__':
    app.run(main)
