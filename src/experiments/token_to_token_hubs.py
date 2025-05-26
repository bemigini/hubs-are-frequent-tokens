"""


Experiment to show the token to token hubs we get with various 
distance/similarity measures, and show that the N_k of the hubs 
do not correspond to token frequencies.



"""

import logging

from datetime import datetime

import os

import pandas as pd 

from src import frequency

from src.file_handling import naming, save_load_json
from src.hubness import analysis, hubs_from_neighb
from src.loading import model_load

from src.plots_tables import plotting


def plot_token_to_token_hub_N_k_vs_token_frequencies_for_all(
        result_folder: str,
        num_neighbours: int,
        vocab_main_folder: str,
        device: str) -> None:
    """ Get token to token hub k-occurrence vs token frequency plots for considered models        
    """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start token_to_token_hubs: %s", dt_string)
    
    config_folder = 'configs'
    num_neighbours = 10
    hub_N_k = 100
    num_top_hubs = 0
    normalized = True

    model_names = [
        'pythia', 
        'pythia_step512', 'pythia_step4000', 'pythia_step16000', 'pythia_step64000',
        'olmo', 'opt', 'mistral', 'llama']
    datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN,
        analysis.Similarity.SOFTMAX_DOT]

    model_name_to_vocab_folder_file = naming.get_model_name_to_vocab_folder_file()
    model_name_to_vocab_folder_path = os.path.join(config_folder, model_name_to_vocab_folder_file)
    model_name_to_vocab_folder = save_load_json.load_json(model_name_to_vocab_folder_path)

    for current_model in model_names:
        logging.info('%s', current_model)
        for current_sim in similarities:
            for current_data in datasets:
                plot_token_to_token_hub_N_k_vs_token_frequencies_for_model_and_save_hub_info(
                    current_model, current_sim, result_folder,
                    num_neighbours, vocab_main_folder, model_name_to_vocab_folder,
                    hub_N_k, num_top_hubs, current_data, normalized,
                    device=device)
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end token_to_token_hubs: %s", dt_string)


def plot_token_to_token_hub_N_k_vs_token_train_frequencies(
        result_folder: str,
        num_neighbours: int,
        vocab_main_folder: str,
        device: str) -> None:
    """ Get token to token hub k-occurrence vs token frequency plots for considered models        
    """
    
    config_folder = 'configs'
    num_neighbours = 10
    hub_N_k = 100
    num_top_hubs = 0
    normalized = True

    model_names = ['pythia', 'olmo']
    similarities = [
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN,
        analysis.Similarity.SOFTMAX_DOT]
    get_frequencies_from = 'train_dataset'

    model_name_to_vocab_folder_file = naming.get_model_name_to_vocab_folder_file()
    model_name_to_vocab_folder_path = os.path.join(config_folder, model_name_to_vocab_folder_file)
    model_name_to_vocab_folder = save_load_json.load_json(model_name_to_vocab_folder_path)

    for current_model in model_names:
        logging.info('%s', current_model)
        for current_sim in similarities:
            plot_token_to_token_hub_N_k_vs_token_frequencies_for_model_and_save_hub_info(
                current_model, current_sim, result_folder,
                num_neighbours, vocab_main_folder, model_name_to_vocab_folder,
                hub_N_k, num_top_hubs, get_frequencies_from, normalized,
                device=device)


def plot_token_to_token_hub_N_k_vs_token_frequencies_for_model_and_save_hub_info(
        current_model: str,
        similarity: analysis.Similarity, 
        result_folder: str,
        num_neighbours: int,
        vocab_main_folder: str,
        model_name_to_vocab_folder: dict,
        hub_N_k: int,
        num_top_hubs: int,
        get_frequencies_from: str,
        normalized: bool,
        device: str) -> None:
    """ Get token to token hub k-occurrence vs token frequency plots for given model """
    dataset_name = ''
    representation_prefix = 'tt'

    file_name = naming.get_hub_info_file_name(
        representation_prefix, current_model, dataset_name, 
        get_frequencies_from, similarity.value, num_neighbours)
    path = os.path.join(result_folder, file_name)

    if os.path.exists(path):
        logging.info('Loading hub info from %s', path)
        df = pd.read_csv(path)
        N_k_idx = df['hub_idx'].values
        N_ks = df['N_k'].values
        hub_token_frequencies = df['token_freq'].values
    else:
        vocab_folder = os.path.join(
            vocab_main_folder, model_name_to_vocab_folder[current_model])
        
        unembedding_matrix = model_load.load_unembedding_matrix(current_model, vocab_folder)        
        
        N_k_idx, N_ks = hubs_from_neighb.get_hub_idxs_and_N_k(
            representation_prefix,
            current_model, dataset_name=dataset_name, 
            similarity=similarity, num_neighbours=num_neighbours, 
            result_folder=result_folder, 
            hub_N_k=hub_N_k, num_top_hubs=num_top_hubs, 
            representations=unembedding_matrix,
            device=device)
        
        # Plot hub N_k vs token frequency
        # Get token frequencies
        if get_frequencies_from == 'train_dataset':
            hub_token_frequencies = frequency.get_frequencies_from_train_frequency_dataset(
                current_model, normalized, N_k_idx)
        else:
            hub_token_frequencies = frequency.get_frequencies_from_dataset(
                current_model, get_frequencies_from, normalized, N_k_idx)
        
        
        df_dict = {'hub_idx': N_k_idx, 'N_k': N_ks, 'token_freq': hub_token_frequencies} 
        df = pd.DataFrame(df_dict)
        
        df.to_csv(path)

    freq_suffix = f'_freq_{get_frequencies_from}'
    
    # plot hub N_k vs frequency 
    if normalized:
        norm_suff = '_normalized'
    else:
        norm_suff = ''

    if hub_N_k > 0:
        hub_suff = f'N_k_g{hub_N_k}'
    else:
        hub_suff = f'{num_top_hubs}'
    
    plot_name = f'token_to_token_hub_N_k_vs_token_frequency{norm_suff}_{current_model}_{similarity.value}_{hub_suff}{freq_suffix}.png'

    #plot_title = f'token to token hubs vs token frequency \n {similarity.value} {current_model}, freq {get_frequencies_from}'
    plot_title = ''

    corr_placement = 'upper_right'

    plotting.make_log_vs_log_scatterplot_w_spearman(
        N_ks, hub_token_frequencies, plot_title, plot_name, corr_placement)
