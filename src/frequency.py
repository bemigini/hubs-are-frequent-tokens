"""

    Calculate the token frequency of datasets using models


"""


import os

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm 

from src.file_handling import naming, save_load_hdf5, save_load_json
from src.loading import model_load



def get_frequencies_from_train_frequency_dataset(
        model_name: str, normalized: bool, hub_idx: NDArray) -> NDArray:
    """ Get token frequencies for the dataset the model was trained on, if we know it.  
    """
    if model_name in ('olmo', 'pythia'):
        if model_name == 'olmo':
            token_idx_to_frequency = get_olmo_token_idx_to_train_frequency()            
        if model_name == 'pythia':
            token_idx_to_frequency = get_pythia_token_idx_to_train_frequency()
        hub_frequencies = [token_idx_to_frequency[str(idx)] for idx in hub_idx]
        if normalized:
            normalization = sum(token_idx_to_frequency.values())
            hub_frequencies = [freq/normalization for freq in hub_frequencies]
    else:
        raise NotImplementedError(f'Train dataset not known for model:{model_name}')
    
    hub_frequencies = np.array(hub_frequencies)
    return hub_frequencies


def get_frequencies_from_dataset(
        model_name: str, dataset_name: str,
        normalized: bool, token_idxs: NDArray) -> NDArray:
    """ Get token frequencies for the given dataset. """

    token_idx_to_frequency = load_token_frequencies_dataset(
            model_name, dataset_name)
    
    token_frequencies = []
    for idx in token_idxs:
        try:
            count = token_idx_to_frequency[str(idx)] 
        except KeyError:
            count = 0
        token_frequencies.append(count)
        
    if normalized:
        normalization = sum(token_idx_to_frequency.values())
        token_frequencies = [freq/normalization for freq in token_frequencies]
    
    return token_frequencies


def get_olmo_token_idx_to_train_frequency() -> Dict[str, int]:
    """ Get dictionary with token indexes to frequency """
    olmo_freq_name = naming.get_olmo_train_frequencies_file_name()
    olmo_freq_path = os.path.join('meta_data', olmo_freq_name)
    token_idx_to_frequency = save_load_json.load_json(olmo_freq_path)

    return token_idx_to_frequency


def get_pythia_token_idx_to_train_frequency() -> Dict[str, int]:
    """ Get dictionary with token indexes to frequency """
    freq_name = naming.get_pythia_train_frequencies_file_name()
    freq_path = os.path.join('meta_data', freq_name)
    token_idx_to_frequency = save_load_json.load_json(freq_path)

    return token_idx_to_frequency


def get_token_frequencies_from_passages(
        token_passages: List[List[int]], model_name: str):
    """ 
        Get the token frequencies for the model on the given passages
    """
    vocab_length = model_load.util.get_vocab_length(model_name)
    vocab_idxs = np.expand_dims(np.arange(vocab_length), axis=1)
    frequencies = np.concatenate((vocab_idxs, np.zeros_like(vocab_idxs)), axis=1)

    for current_passage in tqdm(token_passages):
        unique_tokens, tokens_count = np.unique(current_passage, return_counts=True)
        np.put(frequencies[:, 1], 
               unique_tokens, 
               tokens_count + frequencies[unique_tokens][:, 1], mode='raise')
    
    return frequencies


def get_tokenized_passages_dataset(
        model_name: str, dataset_name: str, 
        corpora_folder_path: str) -> List[List[int]]:
    """ 
        Get tokenized passages of the given dataset 
    """
    tokenizer = model_load.get_tokenizer(model_name)

    dataset_file = naming.get_dataset_file_name(dataset_name)
    dataset_path = os.path.join(corpora_folder_path, dataset_file)
    
    tokenized_passages = []
    with open(dataset_path, encoding='utf-8') as f:
        for line in f:
            current_passage = line.replace('\t', ' ').rstrip('\n')
            token_ids = tokenizer.encode(current_passage)
            tokenized_passages.append(token_ids)
    
    return tokenized_passages


def get_token_frequencies_from_dataset(
        model_name: str, dataset_name: str, corpora_folder_path: str) -> NDArray:
    """ 
        Get the token frequencies for the model on the given dataset
    """
    token_passages = get_tokenized_passages_dataset(
        model_name, dataset_name, corpora_folder_path)

    frequencies = get_token_frequencies_from_passages(token_passages, model_name)
    
    return frequencies


def save_token_frequencies_dataset(
        model_name: str, 
        dataset_name: str, corpora_folder_path: str,
        save_folder: str) -> None:
    """ 
        Get and save the token frequencies for the model on the frequency dataset.
    """
    file_name = naming.get_frequencies_file_name(model_name, dataset_name)
    file_path = os.path.join(save_folder, file_name)

    if os.path.exists(file_path):
        print(f'frequencies file already exists: {file_path}')
        return

    print('Getting frequencies')
    frequencies = get_token_frequencies_from_dataset(
        model_name, dataset_name, corpora_folder_path)
    
    hdf5_dataset_name = naming.get_hdf5_dataset_name()
    save_load_hdf5.save_to_hdf5(frequencies, file_path, hdf5_dataset_name)


def load_token_frequencies_dataset(
        model_name: str, dataset_name: str) -> Dict[str, int]:
    """ 
        Load the token frequencies for the model on the given dataset.
    """
    if 'step' in model_name:
        model_name = model_name.split('_')[0]

    file_path = naming.get_model_frequencies_file_path(model_name, dataset_name)

    frequencies = save_load_json.load_json(file_path)
    return frequencies


# If batch size is needed, 1024 passages per batch would be fitting 
def get_tokenized_passages_full_wikitext(
        model_name: str, full_wikitext_path: str) -> List[List[int]]:
    """ 
        Get tokenized passages of the frequency dataset 
    """
    tokenizer = model_load.get_tokenizer(model_name)

    current_passage = ""

    tokenized_passages = []
    with open(full_wikitext_path, encoding='utf-8') as f:
        for line in f:        
            if line.startswith("<s pile"): # it's in Pile, and it should be ignored
                continue
            if line.startswith("<s>") or line.startswith("</s>"):
                if len(current_passage) > 0: # we've seen something
                    token_ids = tokenizer.encode(current_passage)
                    tokenized_passages.append(token_ids)
                    current_passage = ""
            else:
                fields = line.strip('\n').split('\t')
                current_passage += fields[0] + " "

    # flush buffer if corpus doesn't end with delimiter
    if len(current_passage) > 0: # we've seen something
        token_ids = tokenizer.encode(current_passage)
        tokenized_passages.append(token_ids)
    
    return tokenized_passages


def get_token_frequencies_full_wikitext(
        model_name: str, full_wikitext_path: str):
    """ 
        Get the token frequencies for the model on the frequency dataset
    """
    token_passages = get_tokenized_passages_full_wikitext(
        model_name, full_wikitext_path)

    frequencies = get_token_frequencies_from_passages(token_passages, model_name)
    
    return frequencies


def save_token_frequencies_full_wikitext(
        model_name: str, wikitext_dataset_path: str,
        save_folder: str) -> None:
    """ 
        Get and save the token frequencies for the model on the frequency dataset.
    """
    file_name = naming.get_frequencies_file_name(model_name, 'full_wikitext')
    file_path = os.path.join(save_folder, file_name)

    if os.path.exists(file_path):
        print(f'frequencies file already exists: {file_path}')
        return

    print('Getting frequencies')
    frequencies = get_token_frequencies_full_wikitext(model_name, wikitext_dataset_path)
    
    hdf5_dataset_name = naming.get_hdf5_dataset_name()
    save_load_hdf5.save_to_hdf5(frequencies, file_path, hdf5_dataset_name)


def load_token_frequencies_full_wikitext(
        model_name: str, save_folder: str) -> NDArray:
    """ 
        Load the token frequencies for the model on the full_wikitext dataset.
    """
    file_name = naming.get_frequencies_file_name(model_name, 'full_wikitext')
    file_path = os.path.join(save_folder, file_name)
    hdf5_dataset_name = naming.get_hdf5_dataset_name()    

    frequencies = save_load_hdf5.load_from_hdf5(file_path, hdf5_dataset_name)

    return frequencies
