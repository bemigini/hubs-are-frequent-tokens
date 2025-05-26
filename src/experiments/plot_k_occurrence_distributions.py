"""

Plot the k-occurence distributions of the model representations

"""

import logging

from datetime import datetime

import os

from numpy.typing import NDArray

import torch
from tqdm import tqdm 

from src.plots_tables import plotting

from src.file_handling import naming, save_load_hdf5, save_load_json
from src.hubness import analysis

from src.loading import embedding_load, model_load, util


def get_dot_product_neighbours(
        representations: NDArray, num_neighbours: int, device: str):
    """ Get the num_neighbours neighbourhoods of the representations 
        using dot product as similarity. Higher -> more similar.  
    """
    representations = torch.from_numpy(representations)
    representations = representations.to(device)
    
    num_reps = representations.shape[0] 
    neighbourhoods = torch.zeros((num_reps, num_neighbours))
    batch_size = torch.tensor(128)
    batch_nums = int(torch.ceil(representations.shape[0]/batch_size))
    logging.info('Neighbourhood shape: %s', neighbourhoods.shape)
    logging.info('Num batches: %s', batch_nums)

    for idx in tqdm(range(batch_nums)):
        current_batch = representations[idx*batch_size : (idx + 1)*batch_size]
        current_dots = torch.matmul(current_batch, representations.T).squeeze()
        largest_k_dots = torch.flip(torch.argsort(current_dots, axis=1), [1])[:, :num_neighbours]
        neighbourhoods[idx*batch_size : (idx + 1)*batch_size] = largest_k_dots
    
    neighbourhoods = neighbourhoods.detach().cpu().numpy()
    return neighbourhoods
    

def make_line_plots_of_k_occurrence(
        result_folder: str, vocab_main_folder: str, device: str) -> None:
    """ 
    Make line plots of k-occurrence for representations 
    for various models and on the different datasets 
    """ 
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start plot_k_occurrence_distributions: %s", dt_string)
    
    k_skew_file = naming.get_k_skew_all_file_name()
    k_skew_path = os.path.join(result_folder, k_skew_file)
    k_skew_dict = {
        'model_name': [],
        'dataset_name': [],
        'similarity': [],
        'representation_prefix': [],
        'k-skew': []
    }

    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']
    
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.SOFTMAX_DOT, 
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN]

    num_neighbours = 10

    # Probability distances context with tokens
    k_skew_dict = k_occ_ct(
        result_folder, model_names, hub_datasets, num_neighbours, k_skew_dict)


    # Various distances token with token
    k_skew_dict = k_occ_tt(
        vocab_main_folder, model_names, similarities, num_neighbours, k_skew_dict, device)


    # Various distances context with context
    k_skew_dict = k_occ_cc(
        model_names, similarities, hub_datasets, num_neighbours, k_skew_dict, device)
    
    save_load_json.save_as_json(k_skew_dict, k_skew_path)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end plot_k_occurrence_distributions: %s", dt_string)


def make_line_plots_of_k_occurrence_pythia_checkpoints(
        result_folder: str, vocab_main_folder: str, device: str) -> None:
    """ 
    Make line plots of k-occurrence for representations 
    for various models and on the different datasets 
    """ 
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start plot_k_occurrence_distributions pythia checkpoints: %s", dt_string)
    
    k_skew_file = naming.get_k_skew_pythia_steps_file_name()
    k_skew_path = os.path.join(result_folder, k_skew_file)
    k_skew_dict = {
        'model_name': [],
        'dataset_name': [],
        'similarity': [],
        'representation_prefix': [],
        'k-skew': []
    }

    model_names = ['pythia_step512', 'pythia_step4000', 'pythia_step16000', 'pythia_step64000']
    
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.SOFTMAX_DOT, 
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN]

    num_neighbours = 10

    # Probability distances context with tokens
    k_skew_dict = k_occ_ct(
        result_folder, model_names, hub_datasets, num_neighbours, k_skew_dict)


    # Various distances token with token
    k_skew_dict = k_occ_tt(
        vocab_main_folder, model_names, similarities, num_neighbours, k_skew_dict, device)


    # Various distances context with context
    k_skew_dict = k_occ_cc(
        model_names, similarities, hub_datasets, num_neighbours, k_skew_dict, device)
    
    save_load_json.save_as_json(k_skew_dict, k_skew_path)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end plot_k_occurrence_distributions pythia checkpoints: %s", dt_string)


def k_occ_ct(
        result_folder,
        model_names, hub_datasets, num_neighbours,
        k_skew_dict) -> dict:
    """ plot k-occurrence for context with token comparisons """
    current_similarity = analysis.Similarity.SOFTMAX_DOT
    file_suffix = 'emb'
    representation_prefix = 'ct'

    for current_model in model_names:
        logging.info('%s', current_model)        
        num_tokens = util.get_vocab_length(current_model)

        for dataset_name in hub_datasets:            
            # Get k-occurrences from neighbourhoods
            pred_neighbourhood_file_name = naming.get_next_token_neighbourhood_name(
                current_model, dataset_name, file_suffix, 
                num_neighbours= num_neighbours, similarity=current_similarity.value)
            
            pred_neighb_path = os.path.join(
                result_folder, pred_neighbourhood_file_name)
            h5_data_name = naming.get_hdf5_dataset_name()

            if os.path.exists(pred_neighb_path):
                logging.info('Loading neighbourhoods')
                pred_neighb = save_load_hdf5.load_from_hdf5(pred_neighb_path, h5_data_name)
            else:
                raise ValueError(f'Prediction neighbourhood file not found: {pred_neighb_path}')
            
            N_k, k_skew = analysis.get_all_N_k_and_k_skew_from_neighbours(
                num_tokens, pred_neighb)
            
            # save k-skew
            k_skew_dict['model_name'].append(current_model)
            k_skew_dict['dataset_name'].append(dataset_name)
            k_skew_dict['similarity'].append(analysis.Similarity.SOFTMAX_DOT.value)
            k_skew_dict['representation_prefix'].append(representation_prefix)
            k_skew_dict['k-skew'].append(k_skew)

            #plot_title = f'K-occurrence distribution \n predictions of {current_model} on {dataset_name}'
            plot_title = ''
            plot_file_name = f'N_k_line_plot_predictions_{current_model}_{dataset_name}.png'

            plotting.plot_k_occurrence_line_from_N_k(
                N_k, k_skew, 
                add_k_skew=True, log_scale=True, 
                plot_title=plot_title, save_to=plot_file_name)
    
    return k_skew_dict


def k_occ_tt(
        vocab_main_folder,
        model_names, similarities, num_neighbours,
        k_skew_dict,
        device: str) -> dict:
    """ plot k-occurrence for token with token comparisons """
    representation_prefix = 'tt'
    config_folder = 'configs'

    model_name_to_vocab_folder_file = naming.get_model_name_to_vocab_folder_file()
    model_name_to_vocab_folder_path = os.path.join(config_folder, model_name_to_vocab_folder_file)
    model_name_to_vocab_folder = save_load_json.load_json(model_name_to_vocab_folder_path)

    for current_model in model_names: 
        logging.info('%s', current_model)      
        vocab_folder = os.path.join(
            vocab_main_folder, model_name_to_vocab_folder[current_model])
        
        unembedding_matrix = model_load.load_unembedding_matrix(
            current_model, vocab_folder)
        if isinstance(unembedding_matrix, torch.Tensor):
            unembedding_matrix = unembedding_matrix.cpu().detach().numpy()
        
        for current_similarity in similarities:
            if (current_similarity == analysis.Similarity.EUCLIDEAN 
                or current_similarity == analysis.Similarity.NORM_EUCLIDEAN):
                N_k, k_skew = analysis.N_k(current_similarity, num_neighbours, unembedding_matrix)
            elif current_similarity == analysis.Similarity.SOFTMAX_DOT:
                # Compute dot products
                dot_neighbourhoods = get_dot_product_neighbours(
                    unembedding_matrix, num_neighbours, device)
                
                N_k, k_skew = analysis.get_all_N_k_and_k_skew_from_neighbours(
                    unembedding_matrix.shape[0], dot_neighbourhoods
                )
            else: 
                raise ValueError('Unknown similarity')       

            # save k-skew
            k_skew_dict['model_name'].append(current_model)
            k_skew_dict['dataset_name'].append('')
            k_skew_dict['similarity'].append(current_similarity.value)
            k_skew_dict['representation_prefix'].append(representation_prefix)
            k_skew_dict['k-skew'].append(k_skew)
            
            #plot_title = f'K-occurrence distribution \n tokens from {current_model} with {current_similarity.value}'
            plot_title = ''
            plot_file_name = f'N_k_line_plot_tt_{current_model}_{current_similarity.value}.png'

            plotting.plot_k_occurrence_line_from_N_k(
                N_k, k_skew, 
                add_k_skew=True, log_scale=True, 
                plot_title=plot_title, save_to=plot_file_name)
    
    return k_skew_dict


def k_occ_cc(
        model_names, similarities, hub_datasets, num_neighbours,
        k_skew_dict,
        device: str) -> dict:
    """ plot k-occurrence for context with context comparisons """
    representation_prefix = 'cc'
    embedding_folder = 'embeddings'

    for current_model in model_names:
        logging.info('%s', current_model)
        for dataset_name in hub_datasets:
            # We expect embeddings to already be saved to h5, 
            # so we give empty all_embeddings_folder
            contexts = embedding_load.load_last_layer_embeddings_from_h5(
                embedding_folder, current_model, dataset_name, all_embeddings_folder='')
            if isinstance(contexts, torch.Tensor):
                contexts = contexts.cpu().detach().numpy()
            
            for current_similarity in similarities:
                if (current_similarity == analysis.Similarity.EUCLIDEAN 
                    or current_similarity == analysis.Similarity.NORM_EUCLIDEAN):
                    N_k, k_skew = analysis.N_k(
                        current_similarity, num_neighbours, contexts)
                elif current_similarity == analysis.Similarity.SOFTMAX_DOT:
                    # Compute dot products
                    dot_neighbourhoods = get_dot_product_neighbours(
                        contexts, num_neighbours, device)
                    
                    N_k, k_skew = analysis.get_all_N_k_and_k_skew_from_neighbours(
                        contexts.shape[0], dot_neighbourhoods
                    )
                else: 
                    raise ValueError('Unknown similarity')     

                # save k-skew
                k_skew_dict['model_name'].append(current_model)
                k_skew_dict['dataset_name'].append(dataset_name)
                k_skew_dict['similarity'].append(current_similarity.value)
                k_skew_dict['representation_prefix'].append(representation_prefix)
                k_skew_dict['k-skew'].append(k_skew)
                
                #plot_title = f'K-occurrence distribution \n contexts using {current_model} on {dataset_name} with {current_similarity.value}'
                plot_title = ''
                plot_file_name = f'N_k_line_plot_cc_{current_model}_{dataset_name}_{current_similarity.value}.png'

                plotting.plot_k_occurrence_line_from_N_k(
                    N_k, k_skew, 
                    add_k_skew=True, log_scale=True, 
                    plot_title=plot_title, save_to=plot_file_name)
    
    return k_skew_dict
