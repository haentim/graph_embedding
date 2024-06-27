# Implementation of methods for approximating optimal alignment kernels on graph spaces

This repository contains implementations of graph embedding methods for approximating kernels and metrics on optimal alignment graph spaces. 

## Data sampling
- run 'data_sampling/create_sample.py' to create a sample dataset of undirected weighted graphs.
- run 'data_sampling/compute_kernel_matrix.py' to compute the optimal alignment kernel for each pair from the sample dataset.
- run 'data_sampling/compute_distance_matrix.py' to compute the pairwise optimal alignment metric from the kernel matrix.

## Kernel matrix completion
- execute 'data_sampling/analyse_landmark_sampling.ipynb' to analyse the Nyström method for kernel completion for the sample dataset.
- execute 'data_sampling/compare_sampling_methods.ipynb' to compare the different methods for sampling landmark points for the Nyström method.

## Learning of graph embedding functions
- execute 'generic_embedding/generic_embedding.ipynb' to fit the generic graph embedding model for the sample datasets.
- execute 'convolutional_embedding/convolutional_embedding.ipynb' to fit the convolutional graph embedding model for the sample datasets.