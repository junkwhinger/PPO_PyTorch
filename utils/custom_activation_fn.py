import torch
import torch.nn as nn

def swish(input):
    """
    f(x) = x * sigmoid(x)
    """
    f_input = input * torch.sigmoid(input)
    return f_input


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return swish(input)