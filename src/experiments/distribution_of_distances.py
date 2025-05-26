"""

Experiments showing the distributions of distances we get 
when using the various distance measures on the representations 
from the models. 


"""

import logging

from datetime import datetime

import os

from typing import List

import numpy as np

import torch

from src.plots_tables import plotting
from src import predictions

from src.file_handling import naming, save_load_hdf5, save_load_json
from src.hubness import analysis, hubs_from_neighb

from src.loading import embedding_load, model_load


def make_histograms_of_distances(
        result_folder: str, vocab_main_folder: str, device: str, skip_calcs: List[str]) -> None:
    """ 
    Make histograms of distances for various models and on the different datasets 
    """ 
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start distribution_of_distances: %s", dt_string)

    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']
    
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.SOFTMAX_DOT, 
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN]

    # Probability distances context with tokens
    if 'ct' in skip_calcs:
        logging.info('Skipping probability distances')
    else:
        make_histograms_of_probability_distances(
            result_folder=result_folder, 
            model_names=model_names, hub_datasets=hub_datasets)

    # Various distances token with token
    if 'tt' in skip_calcs:
        logging.info('Skipping token to token distances')
    else:
        make_histograms_of_token_token_distances(
            result_folder=result_folder, model_names=model_names, 
            similarities=similarities, 
            vocab_main_folder=vocab_main_folder,
            device=device
        )

    # Various distances context with context
    if 'cc' in skip_calcs:
        logging.info('Skipping context to context distances')
    else:
        make_histograms_of_context_context_distances(
            result_folder=result_folder, model_names=model_names, 
            similarities=similarities, 
            hub_datasets=hub_datasets,
            device=device
        )

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end distribution_of_distances: %s", dt_string)


def make_histograms_of_distances_pythia_checkpoints(
        result_folder: str, vocab_main_folder: str, device: str, skip_calcs: List[str]) -> None:
    """ 
    Make histograms of distances for various models and on the different datasets 
    """ 
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start distribution_of_distances pythia checkpoints: %s", dt_string)

    model_names = ['pythia_step512', 'pythia_step4000', 'pythia_step16000', 'pythia_step64000']
    
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.SOFTMAX_DOT, 
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN]

    # Probability distances context with tokens
    if 'ct' in skip_calcs:
        logging.info('Skipping probability distances')
    else:
        make_histograms_of_probability_distances(
            result_folder=result_folder, 
            model_names=model_names, hub_datasets=hub_datasets)

    # Various distances token with token
    if 'tt' in skip_calcs:
        logging.info('Skipping token to token distances')
    else:
        make_histograms_of_token_token_distances(
            result_folder=result_folder, model_names=model_names, 
            similarities=similarities, 
            vocab_main_folder=vocab_main_folder,
            device=device
        )

    # Various distances context with context
    if 'cc' in skip_calcs:
        logging.info('Skipping context to context distances')
    else:
        make_histograms_of_context_context_distances(
            result_folder=result_folder, model_names=model_names, 
            similarities=similarities, 
            hub_datasets=hub_datasets,
            device=device
        )

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end distribution_of_distances pythia checkpoints: %s", dt_string)


def make_histograms_of_probability_distances(
        result_folder: str,
        model_names: List[str],
        hub_datasets: List[str]
):
    """ Make histograms for the probability distances """
    representation_prefix = 'ct'
    for model_name in model_names:
        logging.info('%s', model_name)
        for dataset_name in hub_datasets:
            next_probs_arr = predictions.load_next_token_probabilities(
                        model_name, dataset_name, result_folder)

            analysis.calculate_L2_dist_to_uni(
                representation_prefix, model_name, dataset_name, result_folder, next_probs_arr)

            prob_distances = 1 - next_probs_arr
            # In case of numerical inaccuracy 
            # any distance smaller than 0 is set to 0
            prob_distances = np.maximum(prob_distances, 0)
            max_dist = 1

            #plot_title = f'Probability distances in {model_name} on {dataset_name}'
            plot_title = ''
            plot_file_name = f'hist_ct_prob_dist_{model_name}_{dataset_name}.png'

            plotting.make_distance_hist_plot(
                prob_distances, max_dist, plot_title, plot_file_name)

            del next_probs_arr


def make_histograms_of_token_token_distances(
        result_folder: str,
        model_names: List[str],
        similarities: List[analysis.Similarity],
        vocab_main_folder: str,
        device: str
    ):
    """ Make histograms for the token to token distances """
    config_folder = 'configs'
    mean_representations_tokens = {}
    token_key = 'token'
    representation_prefix = 'tt'
    h5_data_name = naming.get_hdf5_dataset_name()

    model_name_to_vocab_folder_file = naming.get_model_name_to_vocab_folder_file()
    model_name_to_vocab_folder_path = os.path.join(
        config_folder, model_name_to_vocab_folder_file)
    model_name_to_vocab_folder = save_load_json.load_json(model_name_to_vocab_folder_path)

    # We remove distance of token to itself
    remove_first_neighbour = True

    for current_model in model_names:
        logging.info('%s', current_model)      
        vocab_folder = os.path.join(
            vocab_main_folder, model_name_to_vocab_folder[current_model])
        
        unembedding_matrix = model_load.load_unembedding_matrix(current_model, vocab_folder)
        if isinstance(unembedding_matrix, torch.Tensor):
            unembedding_matrix = unembedding_matrix.cpu().detach().numpy()
        
        if not 'step' in current_model:
            mean_token = np.mean(unembedding_matrix, axis = 0)
            mean_dist_to_mean = np.mean(np.linalg.norm(unembedding_matrix - mean_token, axis=1))
            mean_representations_tokens[current_model] = {
                'mean_rep': mean_token, 'mean_dist': float(mean_dist_to_mean)}
        
        for current_similarity in similarities:
            dist_file_name = naming.get_distances_file_name(
                representation_prefix, current_model, '', current_similarity.value)
            dist_path = os.path.join(result_folder, dist_file_name)
            
            # Load distances if they exist
            # Else calculate and save
            if os.path.exists(dist_path):
                logging.info('Loading distances %s', dist_path)
                distances = save_load_hdf5.load_from_hdf5(dist_path, h5_data_name)
            else:
                logging.info('Calculating distances')
                distances = hubs_from_neighb.calculate_distance_torch(
                    current_similarity, unembedding_matrix, remove_first_neighbour,                    
                    device = device)
                save_load_hdf5.save_to_hdf5(distances, dist_path, h5_data_name)
            
            if current_similarity == analysis.Similarity.SOFTMAX_DOT:
                analysis.calculate_L2_dist_to_uni(
                    representation_prefix, current_model, '', result_folder, 1-distances)

            distances = distances[~np.isnan(distances)]
            max_dist = np.max(distances)

            #plot_title = f'token-token {current_similarity.value} distances in {current_model}'
            plot_title = ''
            plot_file_name = f'hist_tt_{current_similarity.value}_dist_{current_model}.png'

            plotting.make_distance_hist_plot(
                distances, max_dist, plot_title, plot_file_name)

            if current_similarity == analysis.Similarity.SOFTMAX_DOT:
                # Also make zoomed plot
                #plot_title = f'token-token {current_similarity.value} distances in {current_model}, zoom'
                plot_title = ''
                plot_file_name = f'hist_tt_{current_similarity.value}_dist_{current_model}_zoom.png'

                min_dist = np.min(distances)
                plotting.make_distance_hist_plot(
                    distances, max_dist, plot_title, plot_file_name,
                    min_dist=min_dist)
            
            del distances
    
    if len(mean_representations_tokens) > 0:
        mean_representations_file = naming.get_mean_representations_file(token_key)
        mean_representations_path = os.path.join(result_folder, mean_representations_file)
        save_load_json.save_as_json(mean_representations_tokens, mean_representations_path)

    del unembedding_matrix
    del mean_representations_tokens


def make_histograms_of_context_context_distances(
        result_folder: str,
        model_names: List[str],
        similarities: List[analysis.Similarity],
        hub_datasets: List[str],
        device: str
    ):
    """ Make histograms for the context to context distances """
    embedding_folder = 'embeddings'
    mean_representations_contexts = {}
    context_key = 'context'
    representation_prefix = 'cc'
    h5_data_name = naming.get_hdf5_dataset_name()

    # We remove distance of context to itself
    remove_first_neighbour = True

    for current_model in model_names:        
        logging.info('%s', current_model)
        mean_representations_contexts[current_model] = {}
        for dataset_name in hub_datasets:
            # We expect embeddings to already be saved to h5, 
            # so we give empty all_embeddings_folder
            contexts = embedding_load.load_last_layer_embeddings_from_h5(
                embedding_folder, current_model, dataset_name, all_embeddings_folder='')
            
            if isinstance(contexts, torch.Tensor):
                contexts = contexts.cpu().detach().numpy()
            
            if not 'step' in current_model:
                current_mean_context = np.mean(contexts, axis=0)
                mean_dist_to_mean = np.mean(np.linalg.norm(contexts - current_mean_context, axis=1))            
                mean_representations_contexts[current_model][dataset_name] = {
                    'mean_rep': current_mean_context, 'mean_dist': float(mean_dist_to_mean)}
            
            for current_similarity in similarities:
                dist_file_name = naming.get_distances_file_name(
                representation_prefix, current_model, dataset_name, current_similarity.value)
                dist_path = os.path.join(result_folder, dist_file_name)
                
                # Load distances if they exist
                # Else calculate and save
                if os.path.exists(dist_path):
                    logging.info('Loading distances %s', dist_path)
                    distances = save_load_hdf5.load_from_hdf5(dist_path, h5_data_name)
                else:
                    logging.info('Calculating distances')
                    distances = hubs_from_neighb.calculate_distance_torch(
                        current_similarity, contexts, remove_first_neighbour, device)
                    save_load_hdf5.save_to_hdf5(distances, dist_path, h5_data_name)
                                    
                if current_similarity == analysis.Similarity.SOFTMAX_DOT:
                    analysis.calculate_L2_dist_to_uni(
                        representation_prefix, current_model, dataset_name, 
                        result_folder, 1-distances)
                
                distances = distances[~np.isnan(distances)]
                max_dist = np.max(distances)

                #plot_title = f'context-context {current_similarity.value} distances \n in {current_model} on {dataset_name}'
                plot_title = ''
                plot_file_name = f'hist_cc_{current_similarity.value}_dist_{current_model}_{dataset_name}.png'

                plotting.make_distance_hist_plot(
                    distances, max_dist, plot_title, plot_file_name)
                
                if current_similarity == analysis.Similarity.SOFTMAX_DOT:
                    # Also make zoomed plot
                    #plot_title = f'context-context {current_similarity.value} \n distances in {current_model} on {dataset_name}, zoom'
                    plot_title = ''
                    plot_file_name = f'hist_cc_{current_similarity.value}_dist_{current_model}_{dataset_name}_zoom.png'

                    min_dist = np.min(distances)
                    plotting.make_distance_hist_plot(
                        distances, max_dist, plot_title, plot_file_name,
                        min_dist=min_dist)
                
                del distances
    
    if len(mean_representations_contexts) > 0:
        mean_representations_file = naming.get_mean_representations_file(context_key)
        mean_representations_path = os.path.join(result_folder, mean_representations_file)
        save_load_json.save_as_json(mean_representations_contexts, mean_representations_path)
