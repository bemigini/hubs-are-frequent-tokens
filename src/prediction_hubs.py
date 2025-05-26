"""

Getting next token neighbourhoods from a model
from context compared with unembedding tokens 


"""

import os

import numpy as np

from tqdm import tqdm

from src import predictions

from src.file_handling import naming, save_load_hdf5
from src.hubness import analysis
from src.loading import util


def get_probability_sim_hub_idxs_and_N_k(
    num_neighbours: int, result_folder: str, hub_N_k: int,
    num_top_hubs: int,
    model_name: str, dataset_name: str):
    """
        Get the prediction hubs (tokens) when using model similarity.
        If hub_N_k > 0, then N_k size are used to choose which hubs to retrieve,
        else the num_top_hubs with the largest N_k are retrieved
    """
    file_suffix = 'emb'
    similarity = analysis.Similarity.SOFTMAX_DOT
    # Get model sim neighbourhoods 
    next_token_neighb_file_name = naming.get_next_token_neighbourhood_name(
        model_name, dataset_name, file_suffix, num_neighbours, similarity.value)
    next_token_neighb_path = os.path.join(result_folder, next_token_neighb_file_name)
    h5_data_name = naming.get_hdf5_dataset_name()

    if os.path.exists(next_token_neighb_path):
        print('Loading neighbourhoods')
        next_token_neighb = save_load_hdf5.load_from_hdf5(next_token_neighb_path, h5_data_name)
    else:
        # load probabilities 
        print('Loading probabilities') 
        next_probs_arr = predictions.load_next_token_probabilities(
            model_name, dataset_name, result_folder)

        next_token_neighb = np.zeros((next_probs_arr.shape[0], num_neighbours), dtype=int)
        for i, prob_row in tqdm(enumerate(next_probs_arr)):
            next_token_neighb[i] = np.argsort(prob_row)[::-1][:num_neighbours]
        
        # Save neighbourhoods for later 
        save_load_hdf5.save_to_hdf5(next_token_neighb, next_token_neighb_path, h5_data_name)
    
    num_neighbour_vectors = util.get_vocab_length(model_name)

    return analysis.get_N_k_idx_and_N_k_from_neighbours(
        num_neighbour_vectors, next_token_neighb, hub_N_k, num_top_hubs
    )
