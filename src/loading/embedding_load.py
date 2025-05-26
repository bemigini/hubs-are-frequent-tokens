"""

Loading used text embeddings 


"""


import os
from typing import List


import pickle 

import numpy as np
from numpy.typing import NDArray

from src.file_handling import naming, save_load_hdf5
from src.loading import util


def get_embedding_file_name(model_name: str, dataset: str) -> str:
    """ Get filename of the embeddings for the model and dataset """
    if 'step' in model_name:
        step_part = model_name.split('_')[1]
        model_part = model_name.split('_')[0]
        step_suffix = f'_{step_part}'
        model_name = model_part
    else:
        step_suffix = ''
    return f'hidden_{dataset}_sane_{model_name}_reps{step_suffix}.pickle'


def get_pile_embedding_filename():
    """ Get filename of the pile embeddings """
    return 'hidden_pile_sane_pythia_reps.pickle'


def get_wiki_embedding_filename():
    """ Get filename of the wikitext embeddings """
    return 'hidden_wikitext_sane_pythia_reps.pickle'


def get_embeddings(embedding_folder:str, dataset: str, layers: List[int], model_name: str = 'pythia'):
    """ 
        Get the embeddings from the model and dataset for the given layers 
    """

    use_filename = get_embedding_file_name(model_name, dataset)
    filepath = os.path.join(embedding_folder, use_filename)
    embeddings = []

    with open(filepath,'rb') as f:
        d=pickle.load(f)

        for l in layers:
            embeddings.append(d[l])

    embeddings = [np.array(emb) for emb in embeddings]

    return embeddings


def get_last_layer_embeddings(
    embedding_folder:str, model_name: str, dataset: str):
    """ Get the last layer embeddings from the model and dataset
    """
    last_layer_idx = util.get_last_layer_idx(model_name)
    
    use_filename = get_embedding_file_name(model_name, dataset)
    filepath = os.path.join(embedding_folder, use_filename)    

    with open(filepath,'rb') as f:
        d=pickle.load(f)
        embeddings = d[last_layer_idx]            

    embeddings = np.array(embeddings)

    return embeddings


def save_last_layer_embedings_to_h5(
        all_embeddings_folder: str,
        save_folder: str,
        models: List[str], datasets: List[str]) -> None:
    """
        Save the embeddings for the models and the datasets to the save folder as h5
    """
    file_suffix = naming.get_embedding_file_suffix()
    h5_data_name = naming.get_hdf5_dataset_name()
    for current_model in models:
        last_layer_idx = util.get_last_layer_idx(current_model)
        for current_dataset in datasets:
            file_name = naming.get_embeddings_hdf5_name(
                current_model, current_dataset, last_layer_idx, file_suffix)
            path = os.path.join(save_folder, file_name)

            if os.path.exists(path):
                print(f'Embeddings already saved: {path}')
                continue

            embeddings = get_last_layer_embeddings(
                all_embeddings_folder, current_model, current_dataset)

            save_load_hdf5.save_to_hdf5(embeddings, path, h5_data_name)


def load_last_layer_embeddings_from_h5(
        load_folder: str,
        model_name: str, dataset_name: str,
        all_embeddings_folder: str) -> NDArray:
    """
        Load the embeddings for the model and the dataset 
    """
    h5_data_name = naming.get_hdf5_dataset_name()
    file_suffix = naming.get_embedding_file_suffix()
    last_layer_idx = util.get_last_layer_idx(model_name)
    file_name = naming.get_embeddings_hdf5_name(
                model_name, dataset_name, last_layer_idx, file_suffix)
    path = os.path.join(load_folder, file_name)

    if os.path.exists(path):
        embeddings = save_load_hdf5.load_from_hdf5(path, h5_data_name)
    else:
        embeddings = get_last_layer_embeddings(
                all_embeddings_folder, model_name, dataset_name)

        save_load_hdf5.save_to_hdf5(embeddings, path, h5_data_name)


    return embeddings
