# References : SSD and SalUn github repos

import torch
from torch.autograd import grad
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Subset, dataset
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List


def get_require_grad_params(model: torch.nn.Module):
    return [param for param in model.parameters() if param.requires_grad]


def sam_grad(model, loss):
    params = []

    for param in get_require_grad_params(model):
        params.append(param)

    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]

    return torch.cat(sample_grad)


def apply_perturb(model, v):
    curr = 0
    for param in get_require_grad_params(model):
        length = param.view(-1).shape[0]
        param.view(-1).data += v[curr: curr + length].data
        curr += length
    return model


def woodfisher(model, train_dl, device, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 1000
    o_vec = None
    for idx, (_, data, label) in enumerate(tqdm(train_dl)):
        model.zero_grad()
        data = data.to(device)
        label = label.to(device)
        output = model(data)

        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec


def Wfisher(data_sets, model, criterion, alpha, batch_size, device):
    retain_set = data_sets["retain"]
    forget_set = data_sets["forget"]
    retain_grad_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=batch_size, shuffle=False
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=1, shuffle=False
    )
    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=batch_size, shuffle=False
    )
    # device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    params = []

    for param in get_require_grad_params(model):
        params.append(param.view(-1))

    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params)).to(device)
    total = 0
    model.eval()
    if True:
        for i, (_, data, label) in enumerate(tqdm(forget_loader)):
            model.zero_grad()
            real_num = data.shape[0]
            data = data.to(device)
            label = label.to(device)
            output = model(data)

            loss = criterion(output, label)
            f_grad = sam_grad(model, loss) * real_num
            forget_grad += f_grad
            total += real_num
        total_2 = 0
        for i, (_, data, label) in enumerate(tqdm(retain_grad_loader)):
            model.zero_grad()
            real_num = data.shape[0]
            data = data.to(device)
            label = label.to(device)
            output = model(data)

            loss = criterion(output, label)
            r_grad = sam_grad(model, loss) * real_num
            retain_grad += r_grad
            total_2 += real_num

    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2

    perturb = woodfisher(
        model,
        retain_loader,
        device=device,
        criterion=criterion,
        v=forget_grad - retain_grad
    )

    for name, param in model.named_parameters():
        if name == 'resnet18.conv1.weight':
            print("Still the pretrained model:", param)
        break
    model = apply_perturb(model, alpha * perturb)

    return model


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
class ParameterPerturber:
    def __init__(
            self,
            model,
            opt,
            device="cuda" if torch.cuda.is_available() else "cpu",
            parameters=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        self.forget_threshold = parameters["forget_threshold"]
        self.dampening_constant = parameters["dampening_constant"]
        self.selection_weighting = parameters["selection_weighting"]

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def fulllike_params_dict(
            self, model: torch.nn, fill_value, as_tensor: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict like named_parameters(), with parameter values replaced with fill_value

        Parameters:
        model (torch.nn): model to get param dict from
        fill_value: value to fill dict with
        Returns:
        dict(str,torch.Tensor): dict of named_parameters() with filled in values
        """

        def full_like_tensor(fillval, shape: list) -> list:
            """
            recursively builds nd list of shape shape, filled with fillval
            Parameters:
            fillval: value to fill matrix with
            shape: shape of target tensor
            Returns:
            list of shape shape, filled with fillval at each index
            """
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            tmp = [fillval for _ in range(shape[0])]
            return tmp

        dictionary = {}

        for n, p in model.named_parameters():
            _p = (
                torch.tensor(full_like_tensor(fill_value, p.shape), device=self.device)
                if as_tensor
                else full_like_tensor(fill_value, p.shape)
            )
            dictionary[n] = _p
        return dictionary

    def subsample_dataset(self, dataset: dataset, sample_perc: float) -> Subset:
        """
        Take a subset of the dataset

        Parameters:
        dataset (dataset): dataset to be subsampled
        sample_perc (float): percentage of dataset to sample. range(0,1)
        Returns:
        Subset (float): requested subset of the dataset
        """
        sample_idxs = np.arange(0, len(dataset), step=int((1 / sample_perc)))
        return Subset(dataset, sample_idxs)

    def split_dataset_by_class(self, dataset: dataset) -> List[Subset]:
        """
        Split dataset into list of subsets
            each idx corresponds to samples from that class

        Parameters:
        dataset (dataset): dataset to be split
        Returns:
        subsets (List[Subset]): list of subsets of the dataset,
            each containing only the samples belonging to that class
        """
        n_classes = len(set([target for _, target in dataset]))
        subset_idxs = [[] for _ in range(n_classes)]
        for idx, (x, y) in enumerate(dataset):
            subset_idxs[y].append(idx)

        return [Subset(dataset, subset_idxs[idx]) for idx in range(n_classes)]

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        print(len(dataloader))
        for batch in dataloader:
            # x, _, y = batch
            _, x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
            self,
            original_importance: List[Dict[str, torch.Tensor]],
            forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                    self.model.named_parameters(),
                    original_importance.items(),
                    forget_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)
                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                if 'temperature' in n:
                    continue
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)


###############################################

def ssd_tuning(
        model,
        forget_train_dl,
        dampening_constant,
        selection_weighting,
        full_train_dl,
        device,
        weight_decay=0,
        lr=0.1,
        **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(),  lr=lr)

    ssd = ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()
    sample_importances = ssd.calc_importance(forget_train_dl)

    original_importances = ssd.calc_importance(full_train_dl)
    ssd.modify_weight(original_importances, sample_importances)
    return model
