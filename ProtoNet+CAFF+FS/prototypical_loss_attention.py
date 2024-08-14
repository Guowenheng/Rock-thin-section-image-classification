# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y,dev):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dev = dev.unsqueeze(0).expand(n,m,d)
    diff=x-y
    dist_norm=diff*torch.sqrt(dev)
    dist_norm = torch.pow(dist_norm,2).sum(2)

    return torch.pow(x - y, 2).sum(2), torch.sqrt(dist_norm)

# def trans_sigma(sigma_raw):
#     offset = 1.0
#     scale = 1.0
#     sigma = offset +scale * torch.nn.functional.softplus(sigma_raw)
#     return sigma

def prototypical_loss(input, target, n_support,FA):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    embedding_size = int(input_cpu.size(1) / 2)
    sigma_size = int(input_cpu.size(1) / 2)
    input_cpu_x, sigma_raw = torch.split(input_cpu, [embedding_size, sigma_size], dim=1)
    # sigma=trans_sigma(sigma_raw)
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    # for idx_list in support_idxs:
    #     temp=(input_cpu_x[idx_list] * sigma[idx_list]).sum(0)
    dev_list = torch.stack([input_cpu[idx_list] for idx_list in support_idxs], dim=0)
    feature_weight=FA(dev_list.unsqueeze(1))
    feature_weight=feature_weight.to('cpu')
    # dev=torch.stack([sigma_raw[idx_list].sum(0) for idx_list in support_idxs])
    # pro=torch.stack([(input_cpu_x[idx_list]*sigma[idx_list]).sum(0)/sigma[idx_list].sum(0) for idx_list in support_idxs])
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    query_encode, query_sigma=torch.split(query_samples, [embedding_size, sigma_size], dim=1)
    dists,dist_norm = euclidean_dist(query_samples, prototypes,feature_weight)

    log_p_y = F.log_softmax(-dist_norm , dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val,  acc_val
