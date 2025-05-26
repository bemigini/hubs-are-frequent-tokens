"""

Get examples of hubs when using the various similarities and comparing 
context with token, token with token and context with contex


"""


import logging

from datetime import datetime

import os

import numpy as np

import pandas as pd

from src.file_handling import naming, save_load_json
from src.hubness import analysis

from src.loading import model_load, text_load



def get_examples_of_top_hubs(
    result_folder: str, vocab_main_folder: str, copora_folder: str, num_top_hubs: int):
    """ Get the examples of hubs used in the article """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start get_examples_of_top_hubs: %s", dt_string)

    num_neighbours = 10

    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']
    
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.SOFTMAX_DOT, 
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN]
    
    representation_prefixes = ['ct', 'tt', 'cc']

    for model_name in model_names:
        logging.info('%s', model_name)
        for representation_prefix in representation_prefixes:
            for dataset_name in hub_datasets:
                # Tokens are fixed to a model and do not change with he dataset
                # So we only get token to token hub examples for one dataset
                if (representation_prefix == 'tt') and (dataset_name != 'pile'):
                    continue
                if representation_prefix in ('ct', 'tt'):
                    # Get the token id to token dict 
                    id_to_ex_dict = model_load.get_id_to_decoded_token_dict(
                        model_name, vocab_main_folder)
                else:
                    contexts = text_load.get_texts(copora_folder, dataset_name)
                    contexts = [[c.decode('UTF-8'), nw.decode('UTF-8')] for c, nw in contexts]
                    id_to_ex_dict = dict(enumerate(contexts))
                for similarity in similarities:
                    if representation_prefix == 'ct':
                        if similarity == analysis.Similarity.SOFTMAX_DOT:
                            sim_str = similarity.value
                        else:
                            continue
                    else:
                        sim_str = similarity.value
                    
                    if representation_prefix == 'cc':
                        get_frequencies_from = ''
                    else:
                        # The frequencies are not relevant, so we just take the file 
                        # with frequencies from the same dataset
                        get_frequencies_from = dataset_name

                    if representation_prefix == 'tt':
                        hub_info_dataset = ''
                    else:
                        hub_info_dataset = dataset_name

                    file_name = naming.get_hub_info_file_name(
                        representation_prefix, model_name, hub_info_dataset, 
                        get_frequencies_from, sim_str, num_neighbours)
                    path = os.path.join(result_folder, file_name)

                    if os.path.exists(path):
                        print(f'Loading hub info from {path}')
                        df = pd.read_csv(path)
                        hub_idxs = df['hub_idx'].values
                        N_ks = df['N_k'].values                    
                    else:
                        raise ValueError(f'Could not find hub info file: {path}')
                    
                    # Get the examples of hubs
                    top_hub_idxs = hub_idxs[np.argsort(N_ks)[::-1][:num_top_hubs]]

                    current_examples = []
                    for hub_idx in top_hub_idxs:
                        current_examples.append(id_to_ex_dict[hub_idx])
                    
                    # Save the examples
                    examples_file_name = naming.get_examples_file_name(
                        representation_prefix, model_name, hub_info_dataset, 
                        similarity.value, num_neighbours, num_top_hubs)
                    examples_path = os.path.join(result_folder, examples_file_name)
                    save_load_json.save_as_json(current_examples, examples_path)
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end get_examples_of_top_hubs: %s", dt_string)
