import numpy as np
import scipy
import pickle as pkl
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def sample_landmarks_random(kernel,target_size, return_ixs=False):
    landmark_ixs = np.random.choice(np.arange(kernel.shape[0]), target_size, replace=False)
    new_kernel = kernel[landmark_ixs][:,landmark_ixs]
    if return_ixs:
        return new_kernel, landmark_ixs
    return new_kernel


def sample_landmarks_det(kernel, target_size, return_ixs=False):
    print('computing eigendecomposition')
    eigenvals, eigenvecs = scipy.linalg.eigh(kernel)
    print('computing feature vectors')
    init_feature_vecs = eigenvecs @ np.diag(np.sqrt(np.abs(eigenvals)))
    feature_vecs = init_feature_vecs.copy()
    landmark_ixs = []
    p_inv = np.vectorize(lambda x: 1/x if np.abs(x)>1e-9 else 0.)
    print('determining landmarks')
    for _ in tqdm(range(target_size)):
        normal_feature_vecs = feature_vecs*p_inv(np.expand_dims(np.linalg.norm(feature_vecs,axis=1),1))
        corr = normal_feature_vecs @ feature_vecs.T
        landmark_ix = np.argmax(np.square(corr).sum(axis=1))
        if landmark_ix in landmark_ixs:
            print('double landmark')
            raise ValueError
        landmark_ixs.append(landmark_ix)
        normal_landmark_feature = feature_vecs[landmark_ix]/np.linalg.norm(feature_vecs[landmark_ix])
        feature_vecs = feature_vecs - np.expand_dims(corr[landmark_ix],1) * np.repeat(normal_landmark_feature[np.newaxis,:],feature_vecs.shape[0], axis=0)
        
    new_kernel = kernel[landmark_ixs][:,landmark_ixs]
    if return_ixs:
        return new_kernel, landmark_ixs
    return new_kernel


def sample_landmarks_kmeans(kernel,target_size, return_ixs=False):
    print('computing eigendecompsition')
    eigenvals, eigenvecs = scipy.linalg.eigh(kernel)
    print('computing feature vectors')
    feature_vecs = eigenvecs @ np.diag(np.sqrt(np.abs(eigenvals)))
    print('finding kmeans centers')
    kmeans = KMeans(n_clusters=target_size)
    kmeans.fit(feature_vecs)
    centers = kmeans.cluster_centers_
    print('finding samples (center nearest neighbours)')
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(feature_vecs)
    new_landmarks = neigh.kneighbors(centers,return_distance=False).squeeze(-1)
    new_kernel = kernel[new_landmarks][:,new_landmarks]
    if return_ixs:
        return new_kernel, new_landmarks
    return new_kernel


def sample_landmarks_dist(kernel, target_size, return_ixs=False):
    print('computing distance matrix')
    dists = np.sqrt(np.tile(np.expand_dims(np.diag(kernel),0), (kernel.shape[0],1)) + np.tile(np.expand_dims(np.diag(kernel),1), (1,kernel.shape[0])) - 2*kernel)
    landmark_ixs = [0]
    for _ in tqdm(range(target_size-1)):
        min_dists = np.min(dists[:,landmark_ixs],axis=1)
        new_ix = np.argmax(min_dists)
        landmark_ixs.append(new_ix)
    new_kernel = kernel[landmark_ixs][:,landmark_ixs]
    if return_ixs:
        return new_kernel, landmark_ixs
    return new_kernel
