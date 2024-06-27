import numpy as np
import pickle as pkl
import scipy.sparse


def distance_from_kernel_matrix(kernel_matrix):
    
    mat = np.repeat(np.diag(kernel_matrix)[np.newaxis,:],kernel_matrix.shape[0],axis=0) \
                        + np.repeat(np.diag(kernel_matrix)[:,np.newaxis],kernel_matrix.shape[1], axis=1) \
                        - 2 * kernel_matrix
    
    assert mat.min() > -1.e-9
    mat = np.where(mat>0,mat,0.)
    
    distance_matrix = np.sqrt(mat)
    return distance_matrix


if __name__ == '__main__':
    dataset = '16_10000_15'
    
    with open(f'../data/kernel_matrix_{dataset}.pkl','rb') as f:
        kernel_matrix = pkl.load(f)
    
    distance_matrix = distance_from_kernel_matrix(kernel_matrix)
    
    with open(f'../data/distance_matrix_{dataset}.pkl','wb') as f:
        pkl.dump(distance_matrix,f)
    
    truncation = 64
    
    evals,evecs = scipy.sparse.linalg.eigsh(distance_matrix,k=(truncation if truncation is not None else 256))
    signature = np.where(evals<0,1,0)
    mercer_embeddings = evecs * np.sqrt(np.abs(evals))
    mercer_pos = mercer_embeddings[:,np.where(1-signature)]
    mercer_neg = mercer_embeddings[:,np.where(signature)]
    print(mercer_pos.shape)
    print(mercer_neg.shape)
    with open(f'../data/mercer_{dataset}_distance{f"_{truncation}" if truncation is not None else ""}.pkl','wb') as f:
        pkl.dump((mercer_pos,mercer_neg),f)
        