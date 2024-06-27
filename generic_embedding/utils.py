import numpy as np
import torch

import pickle as pkl


def r_squared(pred,targets):
        target_mean = targets.mean(dim=0)
        pred_sqared_error = torch.linalg.norm(pred-targets,dim=1).square().sum()
        variance = torch.linalg.norm(targets-target_mean,dim=1).square().sum()
        return 1-pred_sqared_error/variance
    


DATASET_MAP = {
    '8_10000_25': '$D_1$',
    '16_10000_15': '$D_2$',
}


def load_kernel_matrix(dataset,distance=False,iterated=False):
    if distance:
        with open(f'data/distance_matrix_{dataset}.pkl','rb') as f:
            kernel = pkl.load(f)    
    else:
        with open(f'data/kernel_matrix_{dataset}.pkl','rb') as f:
            kernel = pkl.load(f)
        if iterated:
            kernel = 1/kernel.shape[0] * kernel @ kernel
    return kernel

def dataset_string(dataset,distance,iterated):
    return f'{dataset}{"_distance" if distance else ""}{"_iterated" if iterated else ""}'