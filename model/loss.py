import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_Loss(output, target):
    return F.cross_entropy(output, target)  

def bce_logit_loss(ouput, target):
    return F.binary_cross_entropy_with_logits(ouput, target)