"""
 Including adapted forward using adapted parameters, such that the gradients can bp to
 gating layers that change the original model parameters
"""

from typing import Optional, Callable, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from typing import Union


class AdaptedLeafCNN1(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """

    def __init__(self, num_classes):
        super(AdaptedLeafCNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)

        # adapted_model_para is used to make self-model a non-leaf computational graph,
        # such that other trainable components using self-model can track the grad passing self-model,
        # e.g. a gating layer that changes the weights of self-model
        self.adapted_model_para = {name: None for name, val in self.named_parameters()}
        # ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
        # 'fc1.weight', 'fc1.bias', 'output.weight', 'output.bias']

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

    def adapted_forward(self, x):
        # forward using the adapted parameters
        x = self.pool(F.relu(self.conv1._conv_forward(
            x, weight=self.adapted_model_para["conv1.weight"], bias=self.adapted_model_para["conv1.bias"])))
        x = self.pool(F.relu(self.conv2._conv_forward(
            x, weight=self.adapted_model_para["conv2.weight"], bias=self.adapted_model_para["conv2.bias"])))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["fc1.weight"], bias=self.adapted_model_para["fc1.bias"]))
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["output.weight"], bias=self.adapted_model_para["output.bias"]))
        return x

    def set_adapted_para(self, name, val):
        self.adapted_model_para[name] = val

    def del_adapted_para(self):
        for key, val in self.adapted_model_para.items():
            if self.adapted_model_para[key] is not None:
                self.adapted_model_para[key].grad = None
                self.adapted_model_para[key] = None


class AdaptedLeafCNN3(AdaptedLeafCNN1):
    def __init__(self, num_classes, n_kernels=32, in_channels=3, fc_factor=1):
        super(AdaptedLeafCNN3, self).__init__(num_classes)
        self.n_kernels = n_kernels
        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 2048 * fc_factor)
        self.output = nn.Linear(2048 * fc_factor, num_classes)

        # adapted_model_para is used to make self-model a non-leaf computational graph,
        # such that other trainable components using self-model can track the grad passing self-model,
        # e.g. a gating layer that changes the weights of self-model
        self.adapted_model_para = {name: None for name, val in self.named_parameters()}

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.n_kernels * 2 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

    def adapted_forward(self, x):
        # forward using the adapted parameters
        x = self.pool(F.relu(self.conv1._conv_forward(
            x, weight=self.adapted_model_para["conv1.weight"], bias=self.adapted_model_para["conv1.bias"])))
        x = self.pool(F.relu(self.conv2._conv_forward(
            x, weight=self.adapted_model_para["conv2.weight"], bias=self.adapted_model_para["conv2.bias"])))
        x = x.view(-1, self.n_kernels * 2 * 5 * 5)
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["fc1.weight"], bias=self.adapted_model_para["fc1.bias"]))
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["output.weight"], bias=self.adapted_model_para["output.bias"]))
        return x


class AdaptedLeNet(AdaptedLeafCNN1):
    """
    CNN model used in "(ICML 21)  Personalized Federated Learning using Hypernetworks":
    a LeNet-based (LeCun et al., 1998) network with two convolution and two fully connected layers.
    """

    def __init__(self, num_classes, n_kernels=32, in_channels=3, fc_factor=1, fc_factor2=1):
        super(AdaptedLeNet, self).__init__(num_classes)
        in_channels = in_channels
        self.n_kernels = n_kernels
        self.fc_factor = fc_factor
        self.fc_factor2 = fc_factor2
        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120 * self.fc_factor)
        self.fc2 = nn.Linear(120 * self.fc_factor, 84 * self.fc_factor2)
        self.output = nn.Linear(84 * self.fc_factor2, num_classes)

        # def __init__(self, num_classes, n_kernels=32, in_channels=3, fc_factor=1):
        #     super(AdaptedLeafCNN3, self).__init__(num_classes)
        #     self.n_kernels = n_kernels
        #     self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        #     self.pool = nn.MaxPool2d(2, 2)
        #     self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        #     self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 2048 * fc_factor)
        #     self.output = nn.Linear(2048 * fc_factor, num_classes)

        # adapted_model_para is used to make self-model a non-leaf computational graph,
        # such that other trainable components using self-model can track the grad passing self-model,
        # e.g. a gating layer that changes the weights of self-model
        self.adapted_model_para = {name: None for name, val in self.named_parameters()}

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2 * self.n_kernels * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

    def adapted_forward(self, x):
        # forward using the adapted parameters
        x = self.pool(F.relu(self.conv1._conv_forward(
            x, weight=self.adapted_model_para["conv1.weight"], bias=self.adapted_model_para["conv1.bias"])))
        x = self.pool(F.relu(self.conv2._conv_forward(
            x, weight=self.adapted_model_para["conv2.weight"], bias=self.adapted_model_para["conv2.bias"])))
        x = x.view(-1, 2 * self.n_kernels * 5 * 5)
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["fc1.weight"], bias=self.adapted_model_para["fc1.bias"]))
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["fc2.weight"], bias=self.adapted_model_para["fc2.bias"]))
        x = F.relu(F.linear(
            x, weight=self.adapted_model_para["output.weight"], bias=self.adapted_model_para["output.bias"]))
        return x


class AdaptedLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AdaptedLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.weight_adapted = None
        self.bias_adapted = None
        self.forward_with_adapt_para = False

    def forward(self, input: Tensor) -> Tensor:
        if self.forward_with_adapt_para:
            weight = self.weight_adapted
            bias = self.bias_adapted
        else:
            weight = self.weight
            bias = self.bias
        return F.linear(input, weight, bias)


def map_module_name(name):
    mapped_name = [f"_modules.[{m_name}]" if m_name.isnumeric() else m_name for m_name in name.split(".")]
    return ".".join(mapped_name)


def deepgetattr(obj, attr):
    ""
    "Recurses through an attribute chain to get the ultimate value."
    ""
    for sub_attr in attr.split("."):
        if sub_attr[0] == "[" and sub_attr[-1] == "]":
            obj = obj[sub_attr[1:-1]]
        else:
            obj = getattr(obj, sub_attr)
    # return reduce(getattr, attr.split('.'), obj)
    return obj
