"""

Plot showing the correlation of prediction hubs with frequent tokens 
for three checkpoints of Pythia


"""

import logging

import os

import pandas as pd

from src.plots_tables import plotting

from src.file_handling import naming
from src.hubness import analysis


def plot_three_step_correlation(
    num_neighbours: int, result_folder: str) -> None:
    """
        Plot the k-occurrence of the prediction hubs vs the token frequency
        for three checkpoints of Pythia. 
        
    """
    model_names = ['pythia_step512', 'pythia_step4000', 'pythia_step16000']
    plot_model_names = ['512', '4000', '16000']
    dataset_name = 'bookcorpus'
    get_frequencies_from = dataset_name
    similarity = analysis.Similarity.SOFTMAX_DOT
    normalized = True
    hub_N_k = 100

    model_N_ks = []
    model_token_frequencies = []

    for model_name in model_names:
        file_name = naming.get_hub_info_file_name(
            'ct', model_name, dataset_name, get_frequencies_from, similarity.value, num_neighbours)
        path = os.path.join(result_folder, file_name)

        if os.path.exists(path):
            logging.info('Loading hub info from %s', path)
            df = pd.read_csv(path)
            N_ks = df['N_k'].values
            hub_token_frequencies = df['token_freq'].values
        else:
            raise ValueError(f'hub info not found: {path}')
        
        model_N_ks.append(N_ks)
        model_token_frequencies.append(hub_token_frequencies)

    
    freq_suffix = f'_freq_{get_frequencies_from}'
    similarity_str = similarity.value
    
    # plot hub N_k vs frequency 
    if normalized:
        norm_suff = '_normalized'
    else:
        norm_suff = ''

    hub_suff = f'N_k_g{hub_N_k}'

    dataset_suff = f'_{dataset_name}'    
    
    plot_name = f'three_steps_pythia_hub_N_k{dataset_suff}_vs_token_frequency{norm_suff}_{similarity_str}_{hub_suff}{freq_suffix}.png'

    corr_placement = 'lower_right'

    plotting.make_three_steps_log_vs_log_scatterplot_w_spearman(
        model_N_ks, model_token_frequencies, plot_model_names, plot_name, corr_placement)
