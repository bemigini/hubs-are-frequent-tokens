"""


Experiment to show correlation between prediction hubs and token frequency



"""

import logging

from datetime import datetime

import os

import pandas as pd

from src import frequency

from src import prediction_hubs as pred_hub
from src.plots_tables import plotting

from src.file_handling import naming
from src.hubness import analysis


def plot_hub_N_k_vs_token_frequency_and_save_hub_info(
    similarity: analysis.Similarity,
    num_neighbours: int, result_folder: str, 
    hub_N_k: int, num_top_hubs: int, 
    model_name: str, dataset_name: str,
    normalized: bool = False,
    get_frequencies_from: str = 'train_dataset'
    ) -> None:
    """
        Plot the k-occurrence of the prediction hubs vs the token frequency.
        Hubs are calculated on the neighbourhoods of the predictions of 
        [model_name] on [dataset_name].
    """

    file_name = naming.get_hub_info_file_name(
        'ct', model_name, dataset_name, get_frequencies_from, similarity.value, num_neighbours)
    path = os.path.join(result_folder, file_name)

    if os.path.exists(path):
        logging.info('Loading hub info from %s', path)
        df = pd.read_csv(path)
        top_N_k_idx = df['hub_idx'].values
        N_ks = df['N_k'].values
        hub_token_frequencies = df['token_freq'].values
    else:
        if similarity == analysis.Similarity.SOFTMAX_DOT:
            top_N_k_idx, N_ks = pred_hub.get_probability_sim_hub_idxs_and_N_k(
                num_neighbours, result_folder, hub_N_k, num_top_hubs, model_name, dataset_name)
        else:
            raise NotImplementedError(f'similarity not implemented: {similarity}')

        # Get token frequencies
        if get_frequencies_from == 'train_dataset':
            hub_token_frequencies = frequency.get_frequencies_from_train_frequency_dataset(
                model_name, normalized, top_N_k_idx)
        else:
            hub_token_frequencies = frequency.get_frequencies_from_dataset(
                model_name, get_frequencies_from, normalized, top_N_k_idx)
        
        df_dict = {'hub_idx': top_N_k_idx, 'N_k': N_ks, 'token_freq': hub_token_frequencies} 
        df = pd.DataFrame(df_dict)        
        df.to_csv(path)

    
    freq_suffix = f'_freq_{get_frequencies_from}'
    similarity_str = similarity.value
    
    # plot hub N_k vs frequency 
    if normalized:
        norm_suff = '_normalized'
    else:
        norm_suff = ''

    if hub_N_k > 0:
        hub_suff = f'N_k_g{hub_N_k}'
    else:
        hub_suff = f'{num_top_hubs}'

    dataset_suff = f'_{dataset_name}'    
    
    plot_name = f'hub_N_k{dataset_suff}_vs_token_frequency{norm_suff}_{model_name}_{similarity_str}_{hub_suff}{freq_suffix}.png'

    #plot_title = f'Prediction hubs vs token frequency \n {model_name} on {dataset_name}, freq {get_frequencies_from}'
    plot_title = ''

    corr_placement = 'lower_right'

    plotting.make_log_vs_log_scatterplot_w_spearman(
        N_ks, hub_token_frequencies, plot_title, plot_name, corr_placement)


def plot_hub_N_k_vs_token_frequencies(
        result_folder: str,
        num_neighbours: int) -> None:
    """ Get hub N_k vs token frequency plots for considered models and datasets """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Plotting start hub_N_k_vs_token_frequencies: %s", dt_string)

    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']
    hub_N_k = 100
    num_top_hubs = 0
    # We use size of hubs instead of num_top_hubs to decide which hubs to retrieve
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    similarity = analysis.Similarity.SOFTMAX_DOT

    for current_model in model_names:
        logging.info('%s', current_model)       
        for current_hub_data in hub_datasets:
            for frequency_data in hub_datasets:
                plot_hub_N_k_vs_token_frequency_and_save_hub_info(
                    similarity, num_neighbours, result_folder, hub_N_k, num_top_hubs,
                    current_model, current_hub_data,
                    normalized = True,
                    get_frequencies_from = frequency_data
                )

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Plotting end hub_N_k_vs_token_frequencies: %s", dt_string)


def plot_hub_N_k_vs_token_train_frequencies(
        result_folder: str,
        num_neighbours: int) -> None:
    """ Get hub N_k vs token train frequency plots for models where we know train data """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Plotting start hub_N_k_vs_token_train_frequencies: %s", dt_string)

    model_names = ['pythia', 'olmo']
    hub_N_k = 100
    num_top_hubs = 0
    # We use size of hubs instead of num_top_hubs to decide which hubs to retrieve
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    get_frequencies_from_train = 'train_dataset'

    similarity = analysis.Similarity.SOFTMAX_DOT

    for current_model in model_names:
        logging.info('%s', current_model)        
        for current_hub_data in hub_datasets:
            plot_hub_N_k_vs_token_frequency_and_save_hub_info(
                similarity, num_neighbours, result_folder, hub_N_k, num_top_hubs,
                current_model, current_hub_data,
                normalized = True,
                get_frequencies_from = get_frequencies_from_train
            )
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Plotting end hub_N_k_vs_token_train_frequencies: %s", dt_string)


def plot_hub_N_k_vs_token_frequencies_checkpoints(
        result_folder: str,
        num_neighbours: int) -> None:
    """ Get hub N_k vs token frequency plots for considered checkpoints """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Plotting start hub_N_k_vs_token_frequencies_checkpoints: %s", dt_string)

    model_names = [
        'pythia_step512', 'pythia_step4000', 'pythia_step16000', 'pythia_step64000']
    hub_N_k = 100
    num_top_hubs = 0
    # We use size of hubs instead of num_top_hubs to decide which hubs to retrieve
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    similarity = analysis.Similarity.SOFTMAX_DOT

    for current_model in model_names:
        logging.info('%s', current_model)   
        for current_hub_data in hub_datasets:
            for frequency_data in hub_datasets:
                plot_hub_N_k_vs_token_frequency_and_save_hub_info(
                    similarity, num_neighbours, result_folder, hub_N_k, num_top_hubs,
                    current_model, current_hub_data,
                    normalized = True,
                    get_frequencies_from = frequency_data
                )

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Plotting end hub_N_k_vs_token_frequencies_checkpoints: %s", dt_string)
