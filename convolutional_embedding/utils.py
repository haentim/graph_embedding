import numpy as np
import torch
from scipy.special import binom

import pickle as pkl


def laplacian_from_adjacency(adjacency):
    return np.diag(adjacency.sum(axis=-1).squeeze()) - adjacency


def normalize_laplacian(lap):
    degrees = torch.diag(lap)
    deg_cpu = degrees.to('cpu')
    deg_cpu.apply_(lambda x: np.sqrt(1/x) if x != 0 else 0.)
    inv_root_deg = torch.diag(deg_cpu.to(degrees.device))
    return .5 * inv_root_deg @ lap @ inv_root_deg
    

def bernstein_coefs(n):
    return torch.tensor([binom(n-1,j) for j in range(n)],dtype=torch.float32)


def laplacian_powers(lap,n):
    powers = [torch.eye(lap.shape[0], device=lap.device)]
    for _ in range(n-1):
        powers.append(lap @ powers[-1])
    return torch.stack([torch.diag(power) for power in powers],dim=1)


def bernstein_polynomials(lap,n):
    
    powers = [torch.eye(lap.shape[0], device=lap.device)]
    for _ in range(n-1):
        powers.append(lap @ powers[-1])
        
    inv_powers = [torch.eye(lap.shape[0], device=lap.device)]
    for _ in range(n-1):
        inv_powers.append((torch.eye(*lap.shape,device=lap.device) - lap) @ powers[-1])
    
    evals = []
    for j in range(n):
        evals.append(binom(n-1,j) * torch.diag(powers[j] @ inv_powers[-j]))
    
    return torch.stack(evals,dim=1)
    

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
        with open(f'../data/distance_matrix_{dataset}.pkl','rb') as f:
            kernel = pkl.load(f)    
    else:
        with open(f'../data/kernel_matrix_{dataset}.pkl','rb') as f:
            kernel = pkl.load(f)
        if iterated:
            kernel = 1/kernel.shape[0] * kernel @ kernel
    return kernel

def dataset_string(dataset,distance,iterated):
    return f'{dataset}{"_distance" if distance else ""}{"_iterated" if iterated else ""}'
