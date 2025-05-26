"""

Getting hubs and k-occurrences from model representations


"""

import logging

import os

import numpy as np
from numpy.typing import NDArray

import torch
from tqdm import tqdm

from src.file_handling import naming, save_load_hdf5
from src.hubness import analysis


def euc_queries_and_points_torch(queries, other_points, remove_first_neighbour: bool):
    """ Get euclidean distance of the points to queries"""
    # Make sure we get distances to all other points 
    num_queries = queries.shape[0]
    num_other_points = other_points.shape[0]
    euc_dists = torch.zeros((num_queries, num_other_points))
    
    b_size = torch.tensor(128)
    q_batch_nums = int(np.ceil(num_queries/b_size))
    other_batch_nums = int(np.ceil(num_other_points/b_size))

    # Time?
    for qidx in tqdm(torch.arange(q_batch_nums)):
        current_query_batch = (queries[qidx*b_size : (qidx + 1)*b_size]).unsqueeze(1)
        for oidx in torch.arange(other_batch_nums):
            current_other_batch = (other_points[oidx*b_size : (oidx + 1)*b_size]).unsqueeze(0)
            # pylint: disable=not-callable
            current_distances = torch.linalg.norm(
                current_query_batch - current_other_batch, dim = 2)
            
            euc_dists[qidx*b_size : (qidx + 1)*b_size, oidx*b_size : (oidx + 1)*b_size] = current_distances
    
    if remove_first_neighbour:
        if num_queries != num_other_points:
            raise NotImplementedError(
                'Only implemented for number of queries equal to number of other points')
        # Remove diagonal
        euc_dists = euc_dists.flatten()[1:].view(
            num_queries-1, num_queries+1)[:,:-1].reshape(num_queries, num_queries-1)

    return euc_dists


def do_soft_max_dots(representations: torch.Tensor):
    """ Calculate softmax dots  """
    # Do dot product in batches 
    softmax_dots = torch.zeros((representations.shape[0], representations.shape[0]))
    batch_size = torch.tensor(128)
    batch_nums = int(torch.ceil(representations.shape[0]/batch_size))
    torch_softmax = torch.nn.Softmax(dim = 1)

    for idx in tqdm(torch.arange(batch_nums)):
        current_batch = representations[idx*batch_size : (idx + 1)*batch_size]
        current_dots = torch.matmul(current_batch, representations.T).squeeze()
        softmax_dots[idx*batch_size : (idx + 1)*batch_size] = torch_softmax(current_dots)
    
    return softmax_dots


def calculate_distance_torch(
        similarity: analysis.Similarity, representations: NDArray, remove_first_neighbour: bool,        
        device: str) -> NDArray:
    """  
    calculate distances between representations
    """
    representations = torch.from_numpy(representations)
    representations = representations.to(device)

    if similarity == analysis.Similarity.EUCLIDEAN:
        distances = euc_queries_and_points_torch(
            representations, representations, remove_first_neighbour)
    elif similarity == analysis.Similarity.NORM_EUCLIDEAN:
        # pylint: disable=not-callable
        norm_unemb_matrix = torch.linalg.norm(representations, dim=1, keepdim= True)

        distances = euc_queries_and_points_torch(
            representations/norm_unemb_matrix, 
            representations/norm_unemb_matrix, remove_first_neighbour)
    elif similarity == analysis.Similarity.SOFTMAX_DOT:
        softmax_dots = do_soft_max_dots(representations)
        
        distances = 1 - softmax_dots
    else:
        raise NotImplementedError(f'similarity not implemented: {similarity}')
    
    distances = distances.detach().cpu().numpy()
    
    return distances


def get_hub_idxs_and_N_k(
    representation_prefix: str,
    model_name: str, dataset_name: str, similarity: analysis.Similarity,
    num_neighbours: int, result_folder: str, hub_N_k: int,
    num_top_hubs: int,
    representations: NDArray,
    device: str):
    """
        Get the hub idxs and N_ks.
        If hub_N_k > 0, then N_k size is used to choose which hubs to retrieve,
        else the num_top_hubs with the largest N_k are retrieved
    """
    neighb_file_name = naming.get_neighbourhood_file_name(
            representations_prefix=representation_prefix, 
            model_name=model_name, dataset_name=dataset_name, 
            num_neighbours=num_neighbours,
            similarity= similarity.value)
    neighb_path = os.path.join(result_folder, neighb_file_name)
    h5_data_name = naming.get_hdf5_dataset_name()

    if os.path.exists(neighb_path):
        logging.info('Loading neighbourhoods')
        neighb = save_load_hdf5.load_from_hdf5(neighb_path, h5_data_name)
    else:
        logging.info('Calculating distances')
        distances = calculate_distance_torch(
            similarity, representations, remove_first_neighbour=False, device=device)
        
        logging.info('Getting neighbourhoods')
        neighb = np.zeros((distances.shape[0], num_neighbours), dtype=int)
        for i, dist_row in tqdm(enumerate(distances)):
            neighb[i] = np.argsort(dist_row)[:num_neighbours]
        
        # Save neighbourhoods for later 
        save_load_hdf5.save_to_hdf5(neighb, neighb_path, h5_data_name)

    num_neighbour_vectors = representations.shape[0]
    N_k_idx, N_k = analysis.get_N_k_idx_and_N_k_from_neighbours(
        num_neighbour_vectors, neighb, hub_N_k, num_top_hubs)
    
    return N_k_idx, N_k
