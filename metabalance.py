# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.




import math
import torch
from torch.optim.optimizer import Optimizer
import time
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class MetaBalance(Optimizer):
    r"""Implements MetaBalance algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        relax factor: the hyper-parameter to control the magnitude proximity
        beta: the hyper-parameter to control the moving averages of magnitudes, set as 0.9 empirically

    """
    def __init__(self, params, relax_factor=0.7, beta=0.9):
        if not 0.0 <= relax_factor < 1.0:
            raise ValueError("Invalid relax factor: {}".format(relax_factor))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {}".format(beta))
        defaults = dict(relax_factor=relax_factor, beta=beta)
        super(MetaBalance, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss_array):#, closure=None
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()

        self.balance_GradMagnitudes(loss_array)

        #return loss

    def balance_GradMagnitudes(self, loss_array):

      for loss_index, loss in enumerate(loss_array):
        loss.backward(retain_graph=True)
        for group in self.param_groups:
          for p in group['params']:

            if p.grad is None:
              print("breaking")
              break

            if p.grad.is_sparse:
              raise RuntimeError('MetaBalance does not support sparse gradients')

            state = self.state[p]

            # State initialization
            if len(state) == 0:
              for j, _ in enumerate(loss_array):
                if j == 0: p.norms = [torch.zeros(1).cuda()]
                else: p.norms.append(torch.zeros(1).cuda())

            # calculate moving averages of gradient magnitudes
            beta = group['beta']
            p.norms[loss_index] = (p.norms[loss_index]*beta) + ((1-beta)*torch.norm(p.grad))

            # narrow the magnitude gap between the main gradient and each auxilary gradient
            relax_factor = group['relax_factor']
            p.grad = (p.norms[0] * p.grad/ p.norms[loss_index]) * relax_factor + p.grad * (1.0 - relax_factor)

            if loss_index == 0:
              state['sum_gradient'] = torch.zeros_like(p.data)
              state['sum_gradient'] += p.grad
            else:
              state['sum_gradient'] += p.grad

            # have to empty p.grad, otherwise the gradient will be accumulated
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

            if loss_index==len(loss_array) - 1:

              p.grad = state['sum_gradient']
