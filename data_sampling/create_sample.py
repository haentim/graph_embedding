import numpy as np
import scipy.sparse
import pickle as pkl
from tqdm import tqdm
import os

def create_sample_dataset(
    size, # number of vertices
    n_samples, # number of graphs
    p_edge, # probability for an edge to exist
    weight_range=[0.,1.], # range from which the edge weights are drawn
    weight_distrbution=np.random.uniform, # distribution from which the edge weights are drawn
    directed=False, # whether the sampled graphs are directed or undirected
):
    """
    | Creates a dataset of 'n_samples' graphs with (maximal) 'size' vertices, where the
    | edges are drawn according to an Erdös-Rényi model and weights are assigned according
    | to 'weight_distribution' on the specified 'weight_range'. If 'directed' is set to
    | 'False', the resulting adjacency is forced to be symmetric.
    """
    weight_sampler = lambda x: weight_distrbution(*weight_range,x)
    print('sampling edges')
    edges = [scipy.sparse.random(size,size,p_edge,data_rvs=weight_sampler) for _ in tqdm(range(n_samples))]
    if not directed: # keep the upper triangular part of each matrix, assign the transpose to the lower triangular part.
        print('enforcing symmetry')
        edges = [scipy.sparse.triu(edges_) + scipy.sparse.triu(edges_,1).T for edges_ in tqdm(edges)]
    return edges
    
if __name__ == '__main__':
    graph_size = 32
    sample_size = 5000
    edge_prob = .15
    ds_name = f'{graph_size}_{sample_size}_{int(100*edge_prob+.5)}'
    if os.path.isfile(f'data/graphs_{ds_name}.pkl'):
        print('file exists, rename or remove to generate new sample.')
    else:
        ds = create_sample_dataset(graph_size,sample_size,edge_prob)
        with open(f'data/graphs_{ds_name}.pkl','wb') as f:
            pkl.dump(ds,f)
    