"""

Get examples for hubs which are zero frequency tokens 
for various models and various datasets


"""


import logging

from datetime import datetime

import os

import numpy as np
import pandas as pd

from src import frequency

from src.file_handling import naming, save_load_json
from src.hubness import analysis
from src.loading import model_load


def get_examples_of_frequency_zero_pred_hubs(
    result_folder: str, vocab_main_folder: str):
    """ Get the examples of hubs used in the article """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start get_examples_of_top_hubs: %s", dt_string)

    num_neighbours = 10
    normalized = False

    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']
    
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    similarity = analysis.Similarity.SOFTMAX_DOT
    sim_str = similarity.value
    
    representation_prefix = 'ct'

    for model_name in model_names:
        logging.info('%s', model_name)
        id_to_token = model_load.get_id_to_decoded_token_dict(
            model_name, vocab_main_folder)                        

        for get_frequencies_from in hub_datasets:
            for dataset_name in hub_datasets:                
                file_name = naming.get_hub_info_file_name(
                    representation_prefix, model_name, dataset_name, 
                    get_frequencies_from, sim_str, num_neighbours)
                path = os.path.join(result_folder, file_name)

                if os.path.exists(path):
                    print(f'Loading hub info from {path}')
                    df = pd.read_csv(path)
                    hub_idxs = df['hub_idx'].values            
                else:
                    raise ValueError(f'Could not find hub info file: {path}')
                
                token_frequencies = frequency.get_frequencies_from_dataset(
                    model_name, get_frequencies_from, normalized, hub_idxs)
    
                token_frequencies = np.array(token_frequencies)

                zero_frequency_filter = token_frequencies == 0
                zero_fequency_idxs = hub_idxs[zero_frequency_filter]

                zero_frequency_strings = [id_to_token[k] for k in zero_fequency_idxs]                
                
                # Save the examples
                examples_file_name = naming.get_freq_zero_pred_hub_file_name(
                    model_name, dataset_name, get_frequencies_from, num_neighbours)
                examples_path = os.path.join(result_folder, examples_file_name)
                save_load_json.save_as_json(zero_frequency_strings, examples_path)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end get_examples_of_top_hubs: %s", dt_string)


def get_example_of_similar_tokens_different_counts(
        vocab_main_folder: str) -> None:
    """ 
        Print an example of very similar tokens with very different counts
    """

    model_name = 'llama'
    get_frequencies_from = 'pile'
    normalized = False
    id_to_token = model_load.get_id_to_decoded_token_dict(
            model_name, vocab_main_folder)     

    token_idxs = [] 
    for k in id_to_token:
        if (('.' == id_to_token[k]) 
            or ('. ' in id_to_token[k]) 
            or ('.\n' == id_to_token[k])):
            token_idxs.append(k)

    token_frequencies = frequency.get_frequencies_from_dataset(
                    model_name, get_frequencies_from, normalized, token_idxs)

    for i, idx in enumerate(token_idxs):
        print(f'{idx}, {id_to_token[idx].encode("utf-8")}, count: {token_frequencies[i]}')
