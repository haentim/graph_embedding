import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from copy import deepcopy
from utils import *


class ConvolutionalModel(nn.Module):
    
    def __init__(self, graph_size=8, filter_degree=None, n_filters=None, vertex_embedding_size=None, hidden_size=2**10, embedding_size=256, device='cpu'):
        super().__init__()
        
        self.device=device
        
        self.graph_size = graph_size
        self.filter_degree = filter_degree  if filter_degree is not None else 2*graph_size
        self.n_filters = n_filters if n_filters is not None else graph_size
        
        self.filter_weights = nn.Parameter(1/np.sqrt(self.filter_degree)*torch.normal(0,1,(self.filter_degree,self.n_filters),device=device))
        
        self.bernstein_weights = None
        
        self.embedding_size = embedding_size
        self.vertex_embedding_size = vertex_embedding_size if vertex_embedding_size is not None else embedding_size
        
        self.hidden_size = hidden_size
        
        self.vertex_model = nn.Sequential(
            nn.Linear(self.n_filters,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.vertex_embedding_size),
        ).to(device)
        
        self.final_model = nn.Sequential(
            nn.Linear(self.vertex_embedding_size,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.embedding_size)
        ).to(device)
        
        
    def prepare_data(self,laplacians):
        lap_power_diags = torch.stack([laplacian_powers(laplacian,self.filter_degree) for laplacian in laplacians])
        log_lap_power_diags = torch.log(1+lap_power_diags)
        return log_lap_power_diags
        
        
    def forward(self, laplacians, laplacian_powers=None):
        if laplacian_powers is None:
            laplacian_powers = self.prepare_data(laplacians)
        filters = laplacian_powers @ self.filter_weights
        dual_pairings = self.vertex_model(filters).mean(dim=-2)
        graph_embeddings = self.final_model(dual_pairings)
        return graph_embeddings


def fit_supervised(model,data,targets,kernel, n_positive,batch_size=64,epochs=10000,test_size=.15,return_test=False):
    optim = torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=1e-9)
    loss_fn = nn.MSELoss()
    
    n_test = int(test_size*len(data[0]))
    p = np.random.permutation(len(data[0]))
    data, targets = [d[p] for d in data], targets[p]
    kernel = kernel[p][:,p]
    
    fit_data, fit_targets = [d[:-n_test] for d in data], targets[:-n_test]
    test_data, test_targets = [d[-n_test:] for d in data], targets[-n_test:]
    test_kernel = kernel[-n_test:][:,-n_test:]
    
    
    n_batches = (len(fit_data[0])-1)//batch_size + 1
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
            p = np.random.permutation(len(fit_data[0]))
            fit_data, fit_targets = [d[p] for d in fit_data], fit_targets[p]
            for batch in range(n_batches):
                x_batch = [d[batch::n_batches] for d in fit_data]
                y_batch = fit_targets[batch::n_batches]
                pred = model(*x_batch)
                loss = loss_fn(pred,y_batch)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                epoch_loss += loss.item()
            
            with torch.no_grad():
                test_pred = model(*test_data)
                test_loss = loss_fn(test_pred,test_targets).item()
                test_kernel_pred = test_pred @ fundamental_symmetry @ test_pred.T
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

def fit_unsupervised(model,data,kernel_matrix,batch_size=64,epochs=10000,test_size=.15,return_test=False):
    n_positive = model.embedding_size//2
    optim = torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=1e-9) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 1e-2**(1/epochs))
    loss_fn = nn.MSELoss()
    
    n_test = int(test_size*len(data[0]))
    p = np.random.permutation(len(data[0]))
    data = [d[p] for d in data]
    kernel_matrix = kernel_matrix[p][:,p]
    fit_data= [d[:-n_test] for d in data]
    test_data = [d[-n_test:] for d in data]
    
    fit_kernel = kernel_matrix[:-n_test][:,:-n_test]
    test_kernel = kernel_matrix[-n_test:][:,-n_test:]
    
    
    fundamental_symmetry = torch.cat([
        torch.cat([torch.eye(n_positive,device=model.device), torch.zeros((n_positive,model.embedding_size-n_positive),device=model.device)],dim=1),
        torch.cat([torch.zeros((model.embedding_size-n_positive,n_positive),device=model.device), -torch.eye(model.embedding_size-n_positive,device=model.device)],dim=1),
    ],dim=0)
    
    clear_diag = lambda x: x-torch.diag(torch.diag(x))
    
    n_batches = (len(fit_data[0])-1)//batch_size + 1
    history = []
    
    best_state = deepcopy(model.state_dict())
    best_loss = np.inf
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            epoch_loss = 0.
            p = np.random.permutation(len(fit_data[0]))
            fit_data, fit_kernel = [d[p] for d in fit_data], fit_kernel[p][:,p]
            for _ in range(n_batches):
                i,j = np.random.randint(0,n_batches,2)
                x_batch = [d[i::n_batches] for d in fit_data]
                y_batch = [d[j::n_batches] for d in fit_data]
                
                kernel_xy = fit_kernel[i::n_batches][:,j::n_batches]
                
                x_pred = model(*x_batch)
                y_pred = model(*y_batch)
                
                kernel_xy_pred = x_pred @ fundamental_symmetry @ y_pred.T
                
                loss = loss_fn(kernel_xy_pred.reshape(-1), kernel_xy.reshape(-1))
                epoch_loss += loss.item()
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
            with torch.no_grad():
                test_pred = model(*test_data)
                test_kernel_pred = test_pred @ fundamental_symmetry @ test_pred.T
                test_loss =  loss_fn(test_kernel_pred.reshape(-1), test_kernel.reshape(-1)).item()
                
                test_r_squared = (1-clear_diag(torch.square(test_kernel-test_kernel_pred)).sum()/clear_diag(torch.square(test_kernel-torch.mean(test_kernel))).sum()).item()
            scheduler.step()
            
            pbar.set_postfix({'loss':epoch_loss/n_batches, 'test_loss':test_loss, 'test_R^2': test_r_squared})
            history.append((epoch, epoch_loss/n_batches,test_loss,test_r_squared))
            
            if test_loss < best_loss:
                best_state = deepcopy(model.state_dict())
                best_loss = test_loss
             
    model.load_state_dict(best_state)
    
    if return_test:
        return history, test_data, test_kernel
    return history   
