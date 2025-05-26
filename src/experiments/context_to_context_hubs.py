"""


Experiment to show the context-to-context hubs we get with various 
distance/similarity measures. 
In this case it does not make sense to talk about crrelation with token frequencies.



"""

import logging

from datetime import datetime

import os

import pandas as pd

from src.file_handling import naming
from src.hubness import analysis, hubs_from_neighb
from src.loading import embedding_load



def save_context_context_hub_N_k_info_for_all(
        result_folder: str,
        num_neighbours: int,
        device: str) -> None:
    """ Get and save context to context hub k-occurrence 
    """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start context_to_context_hubs: %s", dt_string)
    
    embedding_folder = 'embeddings'
    num_neighbours = 10
    hub_N_k = 100
    num_top_hubs = 0

    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']
    datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN,
        analysis.Similarity.SOFTMAX_DOT]

    for current_model in model_names:
        logging.info('%s', current_model)
        for current_sim in similarities:
            for current_data in datasets:
                save_context_context_hub_info(
                    current_model, current_data, current_sim, result_folder,
                    num_neighbours, embedding_folder,
                    hub_N_k, num_top_hubs,
                    device)
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end context_to_context_hubs: %s", dt_string)


def save_context_context_hub_N_k_info_for_pythia_checkpoints(
        result_folder: str,
        num_neighbours: int,
        device: str) -> None:
    """ Get and save context to context hub k-occurrence 
    """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start context_to_context_hubs pythia checkpoints: %s", dt_string)
    
    embedding_folder = 'embeddings'
    num_neighbours = 10
    hub_N_k = 100
    num_top_hubs = 0

    model_names = ['pythia_step512', 'pythia_step4000', 'pythia_step16000', 'pythia_step64000']
    datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN,
        analysis.Similarity.SOFTMAX_DOT]

    for current_model in model_names:
        logging.info('%s', current_model)
        for current_sim in similarities:
            for current_data in datasets:
                save_context_context_hub_info(
                    current_model, current_data, current_sim, result_folder,
                    num_neighbours, embedding_folder,
                    hub_N_k, num_top_hubs,
                    device)
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end context_to_context_hubs pythia checkpoints: %s", dt_string)


def save_context_context_hub_info(
        current_model: str,
        dataset_name: str,
        similarity: analysis.Similarity, 
        result_folder: str,
        num_neighbours: int,
        embedding_folder: str,
        hub_N_k: int,
        num_top_hubs: int,
        device: str
        ) -> None:
    """ Get context to context hub k-occurrence for given model """
    representation_prefix = 'cc'
    get_frequencies_from = ''
    file_name = naming.get_hub_info_file_name(
        representation_prefix, current_model, dataset_name, 
        get_frequencies_from, similarity.value, num_neighbours)
    path = os.path.join(result_folder, file_name)

    if os.path.exists(path):
        logging.info('Hub info already saved at %s', path)  
    else:
        # We expect embeddings to already be saved to h5, 
        # so we give empty all_embeddings_folder
        contexts = embedding_load.load_last_layer_embeddings_from_h5(
            embedding_folder, current_model, dataset_name, all_embeddings_folder='')
        
        hub_idx, N_ks = hubs_from_neighb.get_hub_idxs_and_N_k(
            representation_prefix,
            current_model, dataset_name, similarity, num_neighbours, result_folder, 
            hub_N_k, num_top_hubs, contexts, 
            device=device)
        
        df_dict = {'hub_idx': hub_idx, 'N_k': N_ks} 
        df = pd.DataFrame(df_dict)

        df.to_csv(path)
