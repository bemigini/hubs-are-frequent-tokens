"""

Functions for analysing hubness


"""

import os

import logging

from enum import Enum
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

import pandas as pd

from scipy.stats import skew

from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

from src.file_handling import naming
from src.plots_tables import plotting


class Similarity(Enum):
    """ Implemented (dis)similarity measures """
    EUCLIDEAN = 'euclidean'
    NORM_EUCLIDEAN = 'norm_euclidean'
    SOFTMAX_DOT = 'softmax_dot'
    COSINE = 'cosine'


def N_k_x(
        x_index: int, 
        nearest_k_for_every_point: NDArray) -> float:
    """ Get the k-occurrence of point with index [x_index] """
    # If x is in the k nearest neighbours for a point it only occurs once
    # So we can sum all the occurences of x in the array of nearest neighbours. 
    among_k_nearest = (nearest_k_for_every_point == x_index).sum()
    
    return among_k_nearest


def get_k_skew(N_ks: NDArray) -> float:
    """ Get the skewness of the k-occurrence distribution """
    return skew(N_ks)


def get_all_N_k_and_k_skew_from_neighbours(
        num_neighbour_vectors:int, neighbours) -> NDArray:
    """ Get all the k-occurrences from neighbourhoods """
    N_k_result = np.zeros(num_neighbour_vectors).astype(int)
    for j in tqdm(range(num_neighbour_vectors)):
        N_k_result[j] = N_k_x(
            x_index = j, 
            nearest_k_for_every_point = neighbours)
    k_skew = get_k_skew(N_k_result)

    return N_k_result, k_skew

def get_N_k_idx_and_N_k_from_neighbours(
    num_neighbour_vectors:int, neighbours, 
    hub_N_k: int, num_top_hubs:int):
    """
        If hub_N_k > 0, then N_k size are used to choose which hubs to retrieve,
        else the num_top_hubs with the largest N_k are retrieved
    """
    N_k_result, _ = get_all_N_k_and_k_skew_from_neighbours(
        num_neighbour_vectors, neighbours)

    large_first_N_k_idx = np.argsort(N_k_result)[::-1]    
    
    if hub_N_k > 0:
        large_first_N_k = N_k_result[large_first_N_k_idx]
        larger_than_filter = large_first_N_k >= hub_N_k
        return large_first_N_k_idx[larger_than_filter], large_first_N_k[larger_than_filter]

    top_N_k_idx = large_first_N_k_idx[:num_top_hubs]

    return top_N_k_idx, N_k_result[top_N_k_idx]
    

def N_k_for_fitted(
        k: int, 
        vectors: NDArray, 
        fitted_n_neighbour: NearestNeighbors) -> NDArray:
    """ Get the k-occurence of [vectors] using a fitted NearestNeighbors model """
    
    nearest_k = fitted_n_neighbour.kneighbors(
        vectors, 
        n_neighbors = k, 
        return_distance = False)
    
    num_vectors = vectors.shape[0]
    
    N_k_result = np.zeros(num_vectors).astype(int)
    for i in tqdm(range(num_vectors)):
        N_k_result[i] = N_k_x(
            x_index = i, 
            nearest_k_for_every_point = nearest_k)
    
    return N_k_result


def N_k(
        distance_function: Similarity,
        k: int, 
        vectors: NDArray) -> Tuple[NDArray, float]:
    """ Get the k-occurence and k-skew of [vectors]"""
    
    nn_vectors = np.copy(vectors)
    if distance_function == Similarity.NORM_EUCLIDEAN:
        nn_vectors = preprocessing.normalize(nn_vectors, norm = 'l2', axis = 1)
        knn_metric = 'euclidean'
    else:
        knn_metric = distance_function.value
    
    nearest_neighbours = NearestNeighbors(n_neighbors = k, metric = knn_metric)
    nearest_neighbours.fit(nn_vectors)
    
    N_k_result = N_k_for_fitted(k, nn_vectors, nearest_neighbours)

    k_skew = get_k_skew(N_k_result)
    
    return N_k_result, k_skew


def get_N_k_and_plot_hist(
        distance_function: Similarity,
        k: int, 
        vectors: NDArray,
        title: str,
        add_k_skew: bool,
        log_scale: bool = False,
        save_to: str = '') -> NDArray:
    """ Get the k-occurence of [vectors] and plot the k-occurence histogram """
    
    N_k_result, k_skew =  N_k(distance_function, k, vectors)    
    
    plotting.plot_k_occurrence_hist_from_N_k(
        N_k_result, k_skew = k_skew, add_k_skew = add_k_skew, 
        log_scale = log_scale, plot_title = title,
        save_to = save_to)
    
    return N_k_result, k_skew


def get_N_k_and_plot(
        distance_function: Similarity,
        k: int, 
        vectors: NDArray,
        title: str,
        add_k_skew: bool,
        log_scale: bool = False,
        save_to: str = '') -> NDArray:
    """ Get the k-occurence of [vectors] and plot the k-occurence as a line plot """
    
    N_k_result, k_skew =  N_k(distance_function, k, vectors)    
    
    plotting.plot_k_occurrence_line_from_N_k(
        N_k_result, k_skew = k_skew, add_k_skew = add_k_skew, 
        log_scale = log_scale, plot_title = title,
        save_to = save_to)
    
    return N_k_result, k_skew


def L_2_dist_uniform(p, uniform_size):
    """
    Calculate the L2 distance between a discrete distribution, p, and a uniform distribution
    """
    delta = np.sqrt(np.sum((p - uniform_size)**2, axis=1))
    return delta 


def calculate_L2_dist_to_uni(
        representation_prefix: str, model_name: str, dataset_name: str,
        result_folder: str, distributions) -> None:
    """ Calculate the L2 distance to the uniform distribution """
    L_2_distances_to_uni_file = naming.get_dist_to_uni_file_name(
        representation_prefix, model_name, dataset_name)
    L_2_distances_to_uni_path = os.path.join(result_folder, L_2_distances_to_uni_file)

    if os.path.exists(L_2_distances_to_uni_path):
        logging.info(
            'L2 distance to uniform already calculated: %s', 
            f'{representation_prefix} {model_name} {dataset_name}')
    else:
        uniform_size = 1/distributions.shape[1]
        L_2_distances_to_uni = L_2_dist_uniform(distributions, uniform_size)
        pd.DataFrame(
        {'context_idx': range(distributions.shape[0]), 
        'L2_uni_dist':L_2_distances_to_uni}).to_csv(
            L_2_distances_to_uni_path, index=False)
        del L_2_distances_to_uni


def calculate_concentration_of_distances_condition(distances: NDArray, p: float = 1) -> float:
    """  
        Calculate the condition from the concentration of distances result
    """
    if p <= 0:
        raise ValueError('p must be positive')

    dist_var = np.var(distances**p)
    dist_mean = np.mean(distances**p)

    condition = dist_var/dist_mean**2

    return condition
