import torch
import math


class Sequential(torch.nn.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for computing the output of
    the function alongside with the log-det-Jacobian of such transformation.
    """

    def forward(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """

        log_det_jacobian = 0.0
        for i, module in enumerate(self._modules.values()):
            inputs, log_det_jacobian_ = module(inputs)
            log_det_jacobian = log_det_jacobian + log_det_jacobian_
        return inputs, log_det_jacobian


class BNAF(torch.nn.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for constructing a Block Neural
    Normalizing Flow.
    """

    def __init__(self, *args, res: str = None):
        """
        Parameters
        ----------
        *args : ``Iterable[torch.nn.Module]``, required.
            The modules to use.
        res : ``str``, optional (default = None).
            Which kind of residual connection to use. ``res = None`` is no residual
            connection, ``res = 'normal'`` is ``x + f(x)`` and ``res = 'gated'`` is
            ``a * x + (1 - a) * f(x)`` where ``a`` is a learnable parameter.
        """

        super(BNAF, self).__init__(*args)

        self.res = res

        if res == "gated":
            self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))

    def forward(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """

        outputs = inputs
        grad = None

        for module in self._modules.values():
            outputs, grad = module(outputs, grad)

            grad = grad if len(grad.shape) == 4 else grad.view(grad.shape + [1, 1])

        assert inputs.shape[-1] == outputs.shape[-1]

        if self.res == "normal":
            return inputs + outputs, torch.nn.functional.softplus(grad.squeeze()).sum(
                -1
            )
        elif self.res == "gated":
            return self.gate.sigmoid() * outputs + (1 - self.gate.sigmoid()) * inputs, (
                torch.nn.functional.softplus(grad.squeeze() + self.gate)
                - torch.nn.functional.softplus(self.gate)
            ).sum(-1)
        else:
            return outputs, grad.squeeze().sum(-1)

    def _get_name(self):
        return "BNAF(res={})".format(self.res)


class Permutation(torch.nn.Module):
    """
    Module that outputs a permutation of its input.
    """

    def __init__(self, in_features: int, p: list = None):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features.
        p : ``list`` or ``str``, optional (default = None)
            The list of indeces that indicate the permutation. When ``p`` is not a
            list, if ``p = 'flip'``the tensor is reversed, if ``p = None`` a random
            permutation is applied.
        """

        super(Permutation, self).__init__()

        self.in_features = in_features

        if p is None:
            self.p = np.random.permutation(in_features)
        elif p == "flip":
            self.p = list(reversed(range(in_features)))
        else:
            self.p = p

    def forward(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The permuted tensor and the log-det-Jacobian of this permutation.
        """

        return inputs[:, self.p], 0

    def __repr__(self):
        return "Permutation(in_features={}, p={})".format(self.in_features, self.p)


class MaskedWeight(torch.nn.Module):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.
    """

    def __init__(
        self, in_features: int, out_features: int, dim: int, bias: bool = True
    ):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features per each dimension ``dim``.
        out_features : ``int``, required.
            The number of output features per each dimension ``dim``.
        dim : ``int``, required.
            The number of dimensions of the input of the flow.
        bias : ``bool``, optional (default = True).
            Whether to add a parametrizable bias.
        """

        super(MaskedWeight, self).__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim

        weight = torch.zeros(out_features, in_features)
        for i in range(dim):
            weight[
                i * out_features // dim : (i + 1) * out_features // dim,
                0 : (i + 1) * in_features // dim,
            ] = torch.nn.init.xavier_uniform_(
                torch.Tensor(out_features // dim, (i + 1) * in_features // dim)
            )

        self._weight = torch.nn.Parameter(weight)
        self._diag_weight = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.Tensor(out_features, 1)).log()
        )

        self.bias = (
            torch.nn.Parameter(
                torch.nn.init.uniform_(
                    torch.Tensor(out_features),
                    -1 / math.sqrt(out_features),
                    1 / math.sqrt(out_features),
                )
            )
            if bias
            else 0
        )

        mask_d = torch.zeros_like(weight)
        for i in range(dim):
            mask_d[
                i * (out_features // dim) : (i + 1) * (out_features // dim),
                i * (in_features // dim) : (i + 1) * (in_features // dim),
            ] = 1

        self.register_buffer("mask_d", mask_d)

        mask_o = torch.ones_like(weight)
        for i in range(dim):
            mask_o[
                i * (out_features // dim) : (i + 1) * (out_features // dim),
                i * (in_features // dim) :,
            ] = 0

        self.register_buffer("mask_o", mask_o)

    def get_weights(self):
        """
        Computes the weight matrix using masks and weight normalization.
        It also compute the log diagonal blocks of it.
        """

        w = torch.exp(self._weight) * self.mask_d + self._weight * self.mask_o

        w_squared_norm = (w ** 2).sum(-1, keepdim=True)

        w = self._diag_weight.exp() * w / w_squared_norm.sqrt()

        wpl = self._diag_weight + self._weight - 0.5 * torch.log(w_squared_norm)

        return w.t(), wpl.t()[self.mask_d.bool().t()].view(
            self.dim, self.in_features // self.dim, self.out_features // self.dim
        )

    def forward(self, inputs, grad: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal block of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """

        w, wpl = self.get_weights()

        g = wpl.transpose(-2, -1).unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)

        return (
            inputs.matmul(w) + self.bias,
            torch.logsumexp(g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1)
            if grad is not None
            else g,
        )

    def __repr__(self):
        return "MaskedWeight(in_features={}, out_features={}, dim={}, bias={})".format(
            self.in_features,
            self.out_features,
            self.dim,
            not isinstance(self.bias, int),
        )


class Tanh(torch.nn.Tanh):
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """

    def forward(self, inputs, grad: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal blocks of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """

        g = -2 * (inputs - math.log(2) + torch.nn.functional.softplus(-2 * inputs))
        return (
            torch.tanh(inputs),
            (g.view(grad.shape) + grad) if grad is not None else g,
        )

import torch.nn as nn
class BNAFModel(nn.Module):
    def __init__(self, num_inputs, n_flows=5, n_layers=0, hidden_dim=10, residual='gated'):
        super(BNAFModel, self).__init__()
        flows = []
        for f in range(n_flows):
            layers = []
            for _ in range(n_layers - 1):
                layers.append(
                    MaskedWeight(
                        num_inputs * hidden_dim,
                        num_inputs * hidden_dim,
                        dim=num_inputs,
                    )
                )
                layers.append(Tanh())

            flows.append(
                BNAF(
                    *(
                            [
                                MaskedWeight(
                                    num_inputs, num_inputs * hidden_dim, dim=num_inputs
                                ),
                                Tanh(),
                            ]
                            + layers
                            + [
                                MaskedWeight(
                                    num_inputs * hidden_dim, num_inputs, dim=num_inputs
                                )
                            ]
                    ),
                    res=residual if f < n_flows - 1 else None
                )
            )

            if f < n_flows - 1:
                flows.append(Permutation(num_inputs, "flip"))
        self.model = Sequential(*flows)

    def log_probs(self, x_mb):
        y_mb, log_diag_j_mb = self.model(x_mb)
        log_p_y_mb = (
        torch.distributions.Normal(torch.zeros_like(y_mb), torch.ones_like(y_mb))
            .log_prob(y_mb)
            .sum(-1)
        )
        return log_p_y_mb + log_diag_j_mb

import math
import torch


class Adam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        polyak=0.998,
        ramp=0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= polyak <= 1.0:
            raise ValueError("Invalid polyak decay term: {}".format(polyak))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            polyak=polyak,
        )
        self.ramp_up = ramp
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # Exponential moving average of param
                    state["exp_avg_param"] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                polyak = self.defaults["polyak"]
                #polyak = 0.5**(1/(min(10000, state['step']*self.ramp_up+1e-8)))
                #polyak = min(0.5, 0.5**(1/(state['step']*self.ramp_up+1e-8)))
                #if state['step'] % 1000 == 0:
                #    print(state['step'], '  *** ', polyak)
                state["exp_avg_param"] = (
                    polyak * state["exp_avg_param"] + (1 - polyak) * p.data
                )

        return loss

    def swap(self):
        """
        Swapping the running average of params and the current params for saving parameters using polyak averaging
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                new = p.data
                p.data = state["exp_avg_param"]
                state["exp_avg_param"] = new

    def substitute(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["exp_avg_param"]



if __name__ == '__main__':
    import random
    import numpy as np
    import os
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    seed=123
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    flow = BNAFModel(10, 1, n_layers=0, hidden_dim=10)
    x = np.random.randn(10000, 10)
    x = torch.from_numpy(x.astype(np.float32))
    print(x[:10])
    weight = torch.from_numpy(np.random.randn(10, 10).astype(np.float32))
    x = torch.sigmoid(torch.matmul(x, weight)).detach()
    #+ torch.sigmoid(x).detach()
    #x = x*2.
    print(x.view(-1)[:10])
    opt = Adam(flow.parameters(), lr=1e-3, amsgrad=True, polyak=0.998)
    train_data = x[:8000]
    test_data = x[8000:]
    for it in range(60000):
        ids = np.random.choice(8000, 256)
        loss = -flow.log_probs(train_data[ids]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=0.1)
        opt.step()
        if it % 100 == 0:
            test_log_prob = flow.log_probs(test_data).mean()
            print('[%d] log_prob: %.3f' % (it, test_log_prob.item()))
            opt.swap()
            test_log_prob = flow.log_probs(test_data).mean()
            print('[%d] Moving Average log_prob: %.3f' % (it, test_log_prob.item()))
            opt.swap()
