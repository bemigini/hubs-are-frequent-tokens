"""

Code for the four in one plot:

1. Freq from pile preds on bookcorpus, 2. Freq from bookcorpus preds on bookcorpus
3. Freq from pile preds on pile, 4. Freq from bookcorpus preds on pile


"""


import logging

import os

import numpy as np

import pandas as pd

from scipy import stats

from src.plots_tables import plotting

from src.file_handling import naming
from src.hubness import analysis


def make_and_save_four_in_one_plot() -> None:
    """ Make and save the four in one plot """
    result_folder = 'results'

    model_name = 'pythia'
    num_neighbours = 10
    hub_datasets = ['pile', 'bookcorpus']
    similarity = analysis.Similarity.SOFTMAX_DOT

    eps = 1e-9

    all_N_ks = []
    all_hub_token_frequencies = []
    corrs = []
    
    # Get hub data
    for current_hub_data in hub_datasets:
        for frequency_from in hub_datasets:
            file_name = naming.get_hub_info_file_name(
                'ct', model_name, current_hub_data, frequency_from, similarity.value, num_neighbours)
            path = os.path.join(result_folder, file_name)

            if os.path.exists(path):
                logging.info('Loading hub info from %s', path)
                df = pd.read_csv(path)
                N_ks = df['N_k'].values
                # Avoid frequency zero
                hub_token_frequencies = df['token_freq'].values + eps
            else:
                raise ValueError(f'Could not find hub info file: {path}')
            
            var_token_frequencies = np.var(hub_token_frequencies)
            if var_token_frequencies == 0:
                corr_str = '0.0'
            else:
                spearman_rank_corr = stats.spearmanr(N_ks, hub_token_frequencies)
                spearman_rank_corr_stat = spearman_rank_corr.statistic
                corr_str = f'{spearman_rank_corr_stat:.2f}'

            all_N_ks.append(N_ks)
            all_hub_token_frequencies.append(hub_token_frequencies)
            corrs.append(corr_str)

    plot_file_name = 'four_in_one_plot_pythia_pile_bookcorpus.png'

    # Make plot 
    plotting.make_four_in_one_plot(plot_file_name, all_N_ks, all_hub_token_frequencies, corrs)
