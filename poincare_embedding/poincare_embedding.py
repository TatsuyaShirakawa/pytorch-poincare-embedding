from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import torch
from torch.autograd.function import Function

class PoincareEmbedding(Function):

    def __init__(self):
        super(PoincareEmbedding, self).__init__()

    def forward(self, input):
        self.save_for_backward(input)
        return input

    def backward(self, grad_output):
        x = self.saved_tensors

        # modify gradient by metric
        xx = torch.norm(x, 2, 1)
        scale = ((1-xx) ** 2)/4

        return scale * grad_output
