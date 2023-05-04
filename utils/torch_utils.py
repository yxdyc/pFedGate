import collections.abc
import copy
import pickle
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn


def average_model_of_learners(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()


def average_torch_modules(
        ori_modules,
        target_module,
        weights=None, ):
    ori_device = next(ori_modules[0].parameters()).device

    if weights is None:
        n_modules = len(ori_modules)
        weights = (1 / n_modules) * torch.ones(n_modules, device=ori_device)
    else:
        weights = weights.to(ori_device)

    target_state_dict = target_module.state_dict(keep_vars=True)

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            target_state_dict[key].data.fill_(0.)

            for ori_module_id, ori_module in enumerate(ori_modules):
                state_dict = ori_module.state_dict(keep_vars=True)
                target_state_dict[key].data += weights[ori_module_id] * state_dict[key].data.clone()


def average_torch_state_dict_online(
        ori_state_dict,
        ori_module_weight,
        target_state_dict,
):
    # if ori_module_weight > 1 or ori_module_weight < 0:
    #     raise ValueError(f"In online averaging mode, ori_module_weight should be (0,1], while got {ori_module_weight}")

    for key in target_state_dict:
        target_state_dict[key].data += ori_module_weight * ori_state_dict[key].data.clone()


def mean_torch_state_dict(
        ori_state_dict,
        total_weights,
):
    for key in ori_state_dict:
        ori_state_dict[key].data = ori_state_dict[key].data.clone() / total_weights


def partial_average(learners, average_learner, alpha):
    """
    performs a step towards aggregation for learners, i.e.

    .. math::
        \forall i,~x_{i}^{k+1} = (1-\alpha) x_{i}^{k} + \alpha \bar{x}^{k}

    :param learners:
    :type learners: List[Learner]
    :param average_learner:
    :type average_learner: Learner
    :param alpha:  expected to be in the range [0, 1]
    :type: float

    """
    source_state_dict = average_learner.model.state_dict()

    target_state_dicts = [learner.model.state_dict() for learner in learners]

    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            for target_state_dict in target_state_dicts:
                target_state_dict[key].data = \
                    (1 - alpha) * target_state_dict[key].data + alpha * source_state_dict[key].data


def differentiate_learner(target, reference_state_dict, coeff=1.):
    """
    set the gradient of the model to be the difference between `target` and `reference` multiplied by `coeff`

    :param target:
    :type target: Learner
    :param reference_state_dict:
    :type reference_state_dict: OrderedDict[str, Tensor]
    :param coeff: default is 1.
    :type: float

    """
    target_state_dict = target.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].grad = \
                coeff * (target_state_dict[key].data.clone() - reference_state_dict[key].data.clone())


def copy_model(target, source):
    """
    Copy learners_weights from source to target
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    if isinstance(source, (OrderedDict, dict)):
        # TODO: source -->  deepcopy
        target.load_state_dict(source)
    elif isinstance(source, torch.nn.Module):
        target.load_state_dict(source.state_dict())
    else:
        raise ValueError(f"Un-expected type of source when copy_model, got {type(source)}")


def copy_side_info(target, source, filter_keys=None):
    """
    Copy side-info stored in dict from source to target
    :param target:
    :type target: dict
    :param source:
    :type source: dict
    :return: None

    """
    # target.update(source)
    for key, val in source.items():
        if filter_keys and key in filter_keys:
            continue
        if isinstance(val, torch.nn.Module):
            target[key].load_state_dict(val.state_dict())
        else:
            target[key] = pickle.loads(pickle.dumps(val))


def simplex_projection(v, s=1):
    """
    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):

    .. math::
        min_w 0.5 * || w - v ||_2^2,~s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) torch tensor,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) torch tensor,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.

    References
    ----------
    [1] Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection
        onto the probability simplex: An efficient algorithm with a
        simple proof, and an application." arXiv preprint
        arXiv:1309.1541 (2013)
        https://arxiv.org/pdf/1309.1541.pdf

    """

    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape

    u, _ = torch.sort(v, descending=True)

    cssv = torch.cumsum(u, dim=0)

    rho = int(torch.nonzero(u * torch.arange(1, n + 1) > (cssv - s))[-1][0])

    lambda_ = - float(cssv[rho] - s) / (1 + rho)

    w = v + lambda_

    w = (w * (w > 0)).clip(min=0)

    return w


def get_model_diff_norm(model_a_param, model_b_param, norm_type="fro"):
    res = 0
    if isinstance(model_a_param, dict) and isinstance(model_b_param, dict):
        for key, val in model_a_param.items():
            tmp_res = torch.norm(val - model_b_param[key], p=norm_type)
            if torch.isnan(tmp_res) or torch.isinf(tmp_res):
                continue
            else:
                res += tmp_res
    elif isinstance(model_a_param, collections.abc.Iterable) and isinstance(model_b_param, collections.abc.Iterable):
        for para1, para2 in zip(model_a_param, model_b_param):
            res += torch.norm(para1 - para2, p=norm_type)
    else:
        raise NotImplementedError(
            "Unsupported type pair when calculating model parameters difference. Expect both Dict or Iterable, "
            f"but got {type(model_a_param)} for model a and {type(model_b_param)} for model b")
    return res
