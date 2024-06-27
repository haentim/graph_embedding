import numpy as np
import scipy.sparse
import torch
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
import os
from ot.gromov import gromov_wasserstein
from joblib import Parallel, delayed

import cProfile

TRANSPOSITIONS = {}
TWO_TRANSPOSITIONS = {}

def get_transpositions(size):
    """
    | get all transpositions in the symmetric group S_size
    """
    if size in TRANSPOSITIONS: # early return if transpositions for 'size' are known
        return 
    transpositions = []
    for i in range(size-1):
        for j in range(i+1,size):
            transpositions.append((i,j))
    TRANSPOSITIONS[size] = transpositions
    
    # get all pairs of transpositions for search in the 2-neighborhood
    two_transpositions = []
    for transposition1 in TRANSPOSITIONS[size]:
        for transposition2 in TRANSPOSITIONS[size]:
            two_transpositions.append(((transposition1,transposition2)))
    TWO_TRANSPOSITIONS[size] = two_transpositions
    
    
def apply_tp(alignment,tp):
    """
    | change the alignment by transposing the entries specified by tp
    """
    new_alignment=alignment.copy()
    new_alignment[tp[0]] = alignment[tp[1]]
    new_alignment[tp[1]] = alignment[tp[0]]
    return new_alignment
    

def kernel_aligned(adj1,adj2,alignment):
    """
    | compute the kernel function, given the optimal alignment for the two adjacency matrices
    """
    return (adj1*(adj2[alignment][:,alignment])).sum()

def kernel_aligned_tp(adj1,adj2,alignment,tp):
    """
    | compute the kernel function for the alignment obtained from the initial alignemnt by transposing two vertices
    """
    new_alignment = alignment.copy()
    new_alignment[tp[0]], new_alignment[tp[1]] = new_alignment[tp[1]], new_alignment[tp[0]]
    
    return (adj1*adj2[new_alignment][:,new_alignment]).sum()

def kernel_aligned_double_tp(adj1,adj2,alignment,tps):
    """
    | compute the kernel function for the alignment obtained from the initial alignemnt by two transposistions of vertices
    """
    tp1, tp2 = tps
    
    new_alignment = alignment.copy()
    new_alignment[tp1[0]], new_alignment[tp1[1]] = new_alignment[tp1[1]], new_alignment[tp1[0]]
    new_alignment[tp2[0]], new_alignment[tp2[1]] = new_alignment[tp2[1]], new_alignment[tp2[0]]
    return (adj1*adj2[new_alignment][:,new_alignment]).sum()

def laplacians_from_adjs(adjacencies):
    """
    | returns the laplacians of the graphs with given adjacencies
    """
    
    size = adjacencies[0].shape[0]
    print('calculating laplacians')
    laplacians = [
        scipy.sparse.spdiags(adjacency.sum(axis=1).reshape(-1),0,size,size) - adjacency
    for adjacency in tqdm(adjacencies)]
    return laplacians
            
def low_rank_sparse_spectral_decomp(matrices,rank=4):
    """
    | calculates the 'rank' largest eigenvalues and corresponding eigenvectors of the
    | matrices (typically laplacians).
    """
    print('calculating spectral decomposition')
    spectral_decomp = [
        scipy.sparse.linalg.eigs(matrix,rank)
    for matrix in tqdm(matrices)]
    return spectral_decomp

def spectral_vertex_embedding(adjacencies, dim=4):
    """
    | calculates the spectral vertex embedding (the coordinates of each vertex are the 
    | entries of the corresponding row/col of the eigen-decomposition)
    """
    laplacians = laplacians_from_adjs(adjacencies)
    decomps = low_rank_sparse_spectral_decomp(laplacians,dim)
    print('calculating embeddings')
    embeddings = [
        decomp[1] * np.sqrt(decomp[0])
    for decomp in tqdm(decomps)]
    return embeddings
                
def embedding_distance_matrix(embeddings):
    """
    | calculates the square distance matrix of the embedded vertices
    """
    print('calculating vertex distances')
    distance_matrices = [
        np.array([[np.linalg.norm(v1e-v2e) for v1e in embedding] for v2e in embedding])
    for embedding in tqdm(embeddings)]
    return distance_matrices

def _alignment_local_search(adj1,adj2,starting_alignment):
    """
    | given the ot alignment, search for descents of the kernel value using transposition 
    | of vertex alignments
    """
    current_alignment = starting_alignment
    current_kernel_val = kernel_aligned(adj1,adj2,current_alignment)
    size = adj1.shape[0]
    steps = 0
    alignment_changed = True
    while alignment_changed:
        alignment_changed = False
        
        # search in the 1-step neihborhood
        for tp in TRANSPOSITIONS[size]:
            # tp_alignment = apply_tp(current_alignment,tp)
            # tp_kernel = kernel_aligned(adj1,adj2,tp_alignment)
            tp_kernel = kernel_aligned_tp(adj1,adj2,current_alignment,tp)
            if tp_kernel > current_kernel_val:
                tp_alignment = apply_tp(current_alignment,tp)
                current_alignment = tp_alignment
                current_kernel_val = tp_kernel
                alignment_changed = True
                steps += 1
                break
        
        # search in the 2-step neighborhood
        if not alignment_changed:
            for tp1,tp2 in TWO_TRANSPOSITIONS[size]:
                # tp_alignment = apply_tp(apply_tp(current_alignment,tp1),tp2)
                # tp_kernel = kernel_aligned(adj1,adj2,tp_alignment)
                tp_kernel = kernel_aligned_double_tp(adj1,adj2,current_alignment,(tp1,tp2))
                if tp_kernel > current_kernel_val:
                    tp_alignment = apply_tp(apply_tp(current_alignment,tp1),tp2)
                    current_alignment = tp_alignment
                    current_kernel_val = tp_kernel
                    alignment_changed = True
                    steps += 2
                    break # go back to search in the 1-neighborhood, if an improvement is found in the 2-neighborhood
                
    return current_alignment, current_kernel_val, steps 


def _sinkhorn(mat, max_iter=5, max_dev=1e-3):
    """
    | Sinkhorn iteration to obtain a doubly stochastic matrix
    """
    u,v = torch.ones(mat.shape[0]), torch.ones(mat.shape[1])
    for it in range(max_iter):
        u = 1/(mat @ v)
        v = 1/(mat.T @ u)
        if torch.abs(mat @ v - 1.).max() < max_dev and torch.abs(mat.T @ u - 1.).max() < max_dev:
            break
    return torch.diag(u) @ mat @ torch.diag(v)


def _align_iterative(mat1,mat2, max_iter=10, eta=1, max_dev = .1):
    """ 
    | A Sinkhorn iteration based method for finding the starting point for the local search for an optimal alignment
    """
    mat1 = torch.tensor(mat1,dtype=torch.float32)
    mat2 = torch.tensor(mat2,dtype=torch.float32)
    m = torch.ones_like(mat1, requires_grad=True) / (mat1.shape[0] * mat1.shape[1]) 
    m = m + torch.normal(0.,1./(20*mat1.shape[0] * mat1.shape[1]),m.shape)
    for it in range(max_iter):
        k = ((m.T @ mat1 @ m) * mat2).sum()
        grad = torch.autograd.grad(k,m)[0]
        m = (m + eta * grad)
        m = _sinkhorn(torch.exp(10*m+torch.normal(0.,1e-4,m.shape)))
    return m.detach().numpy()
    
def _find_alignment(i,j,adj1,mat1,adj2,mat2):
    """
    | find the optimal alignment of the two graphs given by adj1 and adj2, with distances
    | of the embedded vertices given by mat1 and mat2, using gromov-wasserstein alignment
    | for an initial alignment, and descent along edges in the cayley-graph generated by
    | transpositions.
    """
    
    ## find an optimal stochastic alignment by minimizing the Gromov-Wasserstein distance
    ot_alignment = gromov_wasserstein(mat1,mat2,loss_fun='kl_loss', max_iter=100, tol_rel=1e-02, tol_abs=1e-04)
    # ot_alignment = _align_iterative(mat1,mat2)
    
    ## consider the most similar deterministic alignment and use it as a starting point for the local search
    ot_alignment_order = np.argmax(ot_alignment,axis=1)
    kernel_alignment,kernel_val,steps = _alignment_local_search(adj1,adj2,ot_alignment_order)
    return i,j,kernel_alignment,kernel_val,steps
                
def find_optimal_alignments(adjacencies1,distance_matrices1,adjacencies2=None,distance_matrices2=None, file=None, save_intervall=10000):
    """
    | compute pairwise optimal alignments for the graphs given by 'adjacencies' with vertex embedding
    | distances given by 'distance_matrices'.
    """
    print('finding optimal alignments')
    if adjacencies2 is None:
        adjacencies2 = adjacencies1
    if distance_matrices2 is None:
        distance_matrices2 = distance_matrices1
    get_transpositions(adjacencies1[0].shape[0]) # ensure that the required transpositions are available
    
    
    alignment_ixs = set()
    if file is not None and os.path.isfile(file):
        with open(file, 'r') as f:
            for line in f:
                i,j, vec, val = line.rstrip('\n').split(';')
                i,j = int(i), int(j)
                alignment_ixs.add((i,j))
    
       
    total_steps = 0
    
    i_,j_ = 0, 0
    size = len(adjacencies1)
    size2 = len(adjacencies2)
    limit = save_intervall
    print(size,size2,limit)
    
    for chunk in tqdm(range((size*size2 - ((size*(size+1)) // 2 - 1))// save_intervall + 1)):
        args = []
        ctr = 0
        while ctr < limit and i_<size:
            if (i_,j_) not in alignment_ixs:
                args.append((i_,j_))
                if i_>= size or j_ >= size2:
                    print(i_,j_)
                    raise IndexError
            ctr += 1
            j_ += 1
            if j_ >= size2:
                i_ += 1
                j_ = i_ + 1
                if j_ >= size2:
                    break
                
        alignments = Parallel(n_jobs=-1)(delayed(_find_alignment)(i,j,adjacencies1[i], distance_matrices1[i], adjacencies2[j],distance_matrices2[j]) for (i,j) in args)
        
        with open(file,'a') as f:
            for i,j,vec,val,steps in alignments:
                total_steps += steps
                f.write(f'{i};{j};{vec.tolist()};{val}\n')

def kernel_matrix_from_alignments(adjacencies,file,diag_limit=None):
    
    kernel_matrix = np.zeros((len(adjacencies),len(adjacencies)))
    print('filling diagonal')
    
    for i,adj in enumerate(adjacencies[:diag_limit]):
        kernel_matrix[i,i] = kernel_aligned(adj,adj,np.arange(adj.shape[0]))
    print('filling matrix from alignment data')
    print(adjacencies[3141])
    print(kernel_matrix[3141,3141])
    
    with open(file, 'r') as f:
        for line in tqdm(f):
            i,j, vec, val = line.rstrip('\n').split(';')
            i,j = int(i), int(j)
            # vec = ast.literal_eval(vec)
            val = float(val)
            kernel_matrix[i,j] = val
            kernel_matrix[j,i] = val
    return kernel_matrix
                
if __name__ == '__main__':
    dataset = '16_10000_15'
    with open(f'../data/graphs_{dataset}.pkl','rb') as f:
        graphs = [np.asarray(graph.todense()) for graph in pkl.load(f)]
    
    embeddings = spectral_vertex_embedding(graphs)
    distance_matrices = embedding_distance_matrix(embeddings)
    alignments = find_optimal_alignments(graphs,distance_matrices, file=f'data/{dataset}_alignments.csv')
    kernel_matrix = kernel_matrix_from_alignments(graphs,file=f'data/{dataset}_alignments.csv')
    
    print('kernel F-norm:', np.linalg.norm(kernel_matrix))
    
    with open(f'../data/kernel_matrix_{dataset}.pkl','wb') as f:
        pkl.dump(kernel_matrix,f)

    truncation = None
    
    evals,evecs = scipy.sparse.linalg.eigsh(kernel_matrix,k=(truncation if truncation is not None else 256))
    signature = np.where(evals<0,1,0)
    
    mercer_embeddings = evecs * np.sqrt(np.abs(evals))
    mercer_pos = mercer_embeddings[:,np.where(1-signature)]
    mercer_neg = mercer_embeddings[:,np.where(signature)]
    print(mercer_pos.shape)
    print(mercer_neg.shape)
    
    with open(f'../data/mercer_{dataset}{f"_{truncation}" if truncation is not None else ""}.pkl','wb') as f:
        pkl.dump((mercer_pos,mercer_neg),f)
        