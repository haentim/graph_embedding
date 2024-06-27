import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
from copy import deepcopy
from utils import *
import os

        
class GenericModel(nn.Module):
    
    def __init__(self,size=8, hidden_size=2**10, embedding_size=256, device='cpu'):
        super().__init__()
        self.size = size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        self.device = device
        
        self.main = nn.Sequential(
            nn.Linear(size**2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,embedding_size)
        ).to(device)
        
    def forward(self,x):
        return self.main(x.view(*x.shape[:-2],-1))

def fit_supervised(model,data,targets,kernel,n_positive,batch_size=64,epochs=10000,test_size=.15, random_permutations=True,return_test=False):
    
    optim = torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=1e-9)
    loss_fn = nn.MSELoss()
    
    n_test = int(test_size*len(data))
    p = np.random.permutation(len(data))
    data, targets = data[p], targets[p]
    kernel = kernel[p][:,p]
    fit_data, fit_targets = data[:-n_test], targets[:-n_test]
    test_data, test_targets = data[-n_test:], targets[-n_test:]
    test_kernel = kernel[-n_test:][:,-n_test:]
    
    
    n_batches = (len(fit_data)-1)//batch_size + 1
    history = []
    
    fundamental_symmetry = torch.cat([
        torch.cat([torch.eye(n_positive,device=model.device), torch.zeros((n_positive,targets.shape[1]-n_positive),device=model.device)],dim=1),
        torch.cat([torch.zeros((targets.shape[1]-n_positive,n_positive),device=model.device), -torch.eye(targets.shape[1]-n_positive,device=model.device)],dim=1),
    ],dim=0)
    
    clear_diag = lambda x: x-torch.diag(torch.diag(x))
    
    test_kernel_ = test_targets @ fundamental_symmetry @ test_targets.T
    test_r_squared_ = (1-clear_diag(torch.square(test_kernel-test_kernel_)).sum()/clear_diag(torch.square(test_kernel-torch.mean(test_kernel))).sum()).item()
    print('kernel approximation:', test_r_squared_)
    
    
    best_state = deepcopy(model.state_dict())
    best_loss = np.inf
    
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            epoch_loss = 0.
            p = np.random.permutation(len(fit_data))
            fit_data, fit_targets = fit_data[p], fit_targets[p]
            if random_permutations:
                perm = np.random.permutation(np.arange(data[0].shape[0]))
                fit_data = fit_data[:,perm,:][:,:,perm]
            for batch in range(n_batches):
                x_batch = fit_data[batch::n_batches]
                y_batch = fit_targets[batch::n_batches]
                pred = model(x_batch)
                loss = loss_fn(pred,y_batch)
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
            with torch.no_grad():
                test_pred = model(test_data)
                test_loss = loss_fn(test_pred,test_targets).item()
                test_kernel_pred = test_pred @ fundamental_symmetry @ test_pred.T
                test_r_squared = (1-clear_diag(torch.square(test_kernel_pred-test_kernel)).sum()/clear_diag(torch.square(test_kernel-torch.mean(test_kernel))).sum()).item()            
                
            pbar.set_postfix({'loss':epoch_loss/n_batches, 'test_loss':test_loss, 'test_R^2': test_r_squared})
            history.append((epoch, epoch_loss/n_batches,test_loss,test_r_squared))
            if test_loss < best_loss:
                best_state = deepcopy(model.state_dict())
                best_loss = test_loss
    model.load_state_dict(best_state)
    if return_test:
        return history, test_data, test_kernel
    return history
    
def fit_unsupervised(model,data,kernel_matrix,batch_size=128,epochs=10000,test_size=.15, random_permutations=True,return_test=False):
    n_positive = model.embedding_size // 2
    optim = torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=1e-9)
    loss_fn = nn.MSELoss()
    
    n_test = int(test_size*len(data))
    p = np.random.permutation(len(data))
    data = data[p]
    kernel_matrix = kernel_matrix[p][:,p]
    fit_data= data[:-n_test]
    test_data = data[-n_test:]
    
    fit_kernel = kernel_matrix[:-n_test][:,:-n_test]
    test_kernel = kernel_matrix[-n_test:][:,-n_test:]
    
    fundamental_symmetry = torch.cat([
        torch.cat([torch.eye(n_positive,device=model.device), torch.zeros((n_positive,model.embedding_size-n_positive),device=model.device)],dim=1),
        torch.cat([torch.zeros((model.embedding_size-n_positive,n_positive),device=model.device), -torch.eye(model.embedding_size-n_positive,device=model.device)],dim=1),
    ],dim=0)
    
    clear_diag = lambda x: x-torch.diag(torch.diag(x))
    
    n_batches = (len(fit_data)-1)//batch_size + 1
    history = []
    
    best_state = deepcopy(model.state_dict())
    best_loss = np.inf
    last_improved = 0
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            epoch_loss = 0.
            p = np.random.permutation(len(fit_data))
            fit_data, fit_kernel = fit_data[p], fit_kernel[p][:,p]
            if random_permutations:
                perm1 = np.random.permutation(np.arange(data[0].shape[0]))
                fit_data_perm1 = fit_data[:,perm1,:][:,:,perm1]
                perm2 = np.random.permutation(np.arange(data[0].shape[0]))
                fit_data_perm2 = fit_data[:,perm2,:][:,:,perm2]
            for _ in range(n_batches):
                i,j = np.random.randint(0,n_batches,2)
                x_batch = fit_data_perm1[i::n_batches]
                y_batch = fit_data_perm2[j::n_batches]
                kernel_xy = fit_kernel[i::n_batches][:,j::n_batches]
                
                x_pred = model(x_batch)
                y_pred = model(y_batch)
                
                kernel_xy_pred = x_pred @ fundamental_symmetry @ y_pred.T
                
                loss = loss_fn(kernel_xy_pred.reshape(-1), kernel_xy.reshape(-1))
                epoch_loss += loss.item()
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
            with torch.no_grad():
                test_pred = model(test_data)
                test_kernel_pred = test_pred @ fundamental_symmetry @ test_pred.T
                test_loss =  loss_fn(test_kernel_pred.reshape(-1), test_kernel.reshape(-1)).item()
                
                test_r_squared = (1-clear_diag(torch.square(test_kernel-test_kernel_pred)).sum()/clear_diag(torch.square(test_kernel-torch.mean(test_kernel))).sum()).item()
            
            pbar.set_postfix({'loss':epoch_loss/n_batches, 'test_loss':test_loss, 'test_R^2': test_r_squared})
            history.append((epoch, epoch_loss/n_batches,test_loss,test_r_squared))
            
            if test_loss < best_loss:
                best_state = deepcopy(model.state_dict())
                best_loss = test_loss
    model.load_state_dict(best_state)
    if return_test:
        return history, test_data, test_kernel
    return history
