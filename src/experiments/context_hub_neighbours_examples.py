"""

Get examples of the neighbours of context hubs 


"""




import logging

from datetime import datetime

import os

import numpy as np

import pandas as pd

from tqdm import tqdm

from src.file_handling import naming, save_load_hdf5, save_load_json
from src.hubness import analysis

from src.loading import text_load


def get_examples_of_when_context_hubs_are_neighbours(
    result_folder: str, copora_folder: str):
    """ Get the examples where context hubs are neighbours used in the article """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start get_examples_of_when_context_hubs_are_neighbours: %s", dt_string)

    num_top_hubs = 10
    num_examples = 10

    num_neighbours = 10

    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']
    
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.SOFTMAX_DOT, 
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN]
    
    representation_prefix = 'cc'

    for model_name in model_names:
        logging.info('%s', model_name)        
        for dataset_name in hub_datasets:            
            contexts = text_load.get_texts(copora_folder, dataset_name)
            contexts = [[c.decode('UTF-8'), nw.decode('UTF-8')] for c, nw in contexts]
            id_to_ex_dict = dict(enumerate(contexts))

            for similarity in similarities:
                sim_str = similarity.value
                
                get_frequencies_from = ''
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
                
                # Get the top hubs
                arg_sorted_N_ks = np.argsort(N_ks)[::-1]
                top_hub_idxs = hub_idxs[arg_sorted_N_ks[:num_top_hubs]]

                # Get neighbourhoods
                neighb_file_name = naming.get_neighbourhood_file_name(
                        representations_prefix=representation_prefix, 
                        model_name=model_name, dataset_name=dataset_name, 
                        num_neighbours=num_neighbours,
                        similarity= sim_str)
                neighb_path = os.path.join(result_folder, neighb_file_name)
                h5_data_name = naming.get_hdf5_dataset_name()

                if os.path.exists(neighb_path):
                    logging.info('Loading neighbourhoods')
                    neighb = save_load_hdf5.load_from_hdf5(neighb_path, h5_data_name)
                else:
                    raise FileNotFoundError(f'Neighbourhood file not found: {neighb_path}')

                current_examples = {}
                for idx, row in tqdm(enumerate(neighb)):
                    for j, hub_idx in enumerate(top_hub_idxs):
                        hub_idx_int = int(hub_idx)                    
                        if hub_idx in row:
                            if hub_idx_int not in current_examples:
                                current_examples[hub_idx_int] = {}
                                current_examples[hub_idx_int]['hub'] = id_to_ex_dict[hub_idx_int]
                                current_examples[hub_idx_int]['hub_N_k'] = N_ks[arg_sorted_N_ks[j]]
                                current_examples[hub_idx_int]['neighbours'] = []
                            if len(current_examples[hub_idx_int]['neighbours']) < num_examples:
                                current_examples[hub_idx_int]['neighbours'].append(id_to_ex_dict[idx])

                
                # Save the examples
                examples_file_name = naming.get_hub_is_neighbour_examples_file_name(
                    representation_prefix, model_name, hub_info_dataset, 
                    similarity.value, num_neighbours, num_top_hubs)
                examples_path = os.path.join(result_folder, examples_file_name)
                save_load_json.save_as_json(current_examples, examples_path)
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end get_examples_of_when_context_hubs_are_neighbours: %s", dt_string)
