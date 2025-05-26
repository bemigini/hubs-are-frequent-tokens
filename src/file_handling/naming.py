"""  

For keeping track of file names


"""

import os 

from typing import List

from src.file_handling import save_load_json

from src.hubness import analysis


def get_config_folder_name() -> str:
    """ Get the name of the config folder """
    return 'configs'


def get_hdf5_dataset_name() -> str:
    """ Get hdf5 dataset name """
    return 'data'


def get_model_frequencies_file_path(model_name: str, dataset: str) -> str:
    """ Get frequencies file path for model and data """
    return f'./meta_data/{dataset}_{model_name}_counts.json'


def get_olmo_train_frequencies_file_name() -> str:
    """ Get olmo frequencies file name """
    return 'token_freq_olmo_sample_dolma_train.json'


def get_pythia_train_frequencies_file_name() -> str:
    """ Get pythia frequencies file name """
    return 'token_freq_pythia_pile_train.json'


def get_hub_info_file_name(
    representation_prefix: str,
    model_name: str, dataset_name: str, frequencies_from: str, 
    similarity: str, num_neighbours: int) -> str:
    """ Get the hub info file name """
    if dataset_name == '':
        ds_suff = ''
    else:
        ds_suff = f'_{dataset_name}'
    
    if frequencies_from == '':
        freq_suff = ''
    else:
        freq_suff = f'_freq_from_{frequencies_from}'

    file_name = f'{representation_prefix}_hub_info_{model_name}{ds_suff}{freq_suff}_{similarity}_{num_neighbours}.csv'
    return file_name


def get_mean_representations_file(token_or_context: str) -> str:
    """ Get the mean representations file name """
    return f'mean_representations_all_models_{token_or_context}.json'


def get_true_next_word_file_name(dataset: str) -> str:
    """ Get the true next word file name """
    return f'true_next_words_{dataset}.json'


def get_hub_info_table_file_name(
    num_neighbours: int, representation_prefix: str, all_or_checkpoints: str) -> str:
    """ Get the hub info table file name """
    return f'hub_info_table_{all_or_checkpoints}_{representation_prefix}_{num_neighbours}.csv'


def get_examples_file_name(
    representation_prefix: str, model_name: str, dataset_name: str, 
    similarity: str, num_neighbours: int, top_num_hubs: int) -> str:
    """ Get the examples file name """
    if dataset_name == '':
        ds_suff = ''
    else:
        ds_suff = f'_{dataset_name}'

    return f'hub_examples_top{top_num_hubs}_{representation_prefix}_{model_name}{ds_suff}_{similarity}_{num_neighbours}.json'


def get_hub_is_neighbour_examples_file_name(
    representation_prefix: str, model_name: str, dataset_name: str, 
    similarity: str, num_neighbours: int, top_num_hubs: int) -> str:
    """ Get the hub is neighbour examples file name """
    if dataset_name == '':
        ds_suff = ''
    else:
        ds_suff = f'_{dataset_name}'

    return f'hub_is_neighbour_examples_top{top_num_hubs}_{representation_prefix}_{model_name}{ds_suff}_{similarity}_{num_neighbours}.json'


def get_hub_neighbour_examples_file_name(
    representation_prefix: str, model_name: str, dataset_name: str, 
    similarity: str, num_neighbours: int, top_num_hubs: int) -> str:
    """ Get the hub neighbour examples file name """
    if dataset_name == '':
        ds_suff = ''
    else:
        ds_suff = f'_{dataset_name}'

    return f'hub_neighbour_examples_top{top_num_hubs}_{representation_prefix}_{model_name}{ds_suff}_{similarity}_{num_neighbours}.json'


def get_freq_zero_pred_hub_file_name(
    model_name: str, dataset_name: str, 
    frequencies_from: str, num_neighbours: int) -> str:
    """ Get the zero frequency prediction hubs file name """
    
    return f'freq_zero_pred_hubs_{model_name}_{dataset_name}_freq_{frequencies_from}_{num_neighbours}.json'


def get_dist_to_uni_file_name(
        representation_prefix: str, 
        model_name: str, dataset_name: str) -> str:
    """ Get the distance to uniform file name """
    if representation_prefix == 'tt':
        dataset_suff = ''
    else:
        dataset_suff = f'_{dataset_name}'
    file = f'L_2_distances_to_uni_{representation_prefix}_{model_name}{dataset_suff}.csv'
    return file


def get_distances_file_name(
    representation_prefix: str, 
    model_name: str, dataset_name: str, 
    similarity: str) -> str:
    """ Get the calculated distances file name """
    if representation_prefix == 'tt':
        dataset_suff = ''
    else:
        dataset_suff = f'_{dataset_name}'
    
    return f'distances_{representation_prefix}_{model_name}{dataset_suff}_{similarity}.h5'


def get_concentration_of_distances_condition_all_file_name() -> str:
    """ Get the concentration of distances file name """
        
    return 'concentration_condition_all_models_all_datasets.json'


def get_k_skew_all_file_name() -> str:
    """ Get the k-skew file name """
        
    return 'k_skew_all_models_all_datasets.json'


def get_k_skew_pythia_steps_file_name() -> str:
    """ Get the k-skew file name for pythia checkpoints """
        
    return 'k_skew_pythia_checkpoints_all_datasets.json'


def get_dataset_file_name(dataset_name: str) -> str:
    """
        Get file name of the dataset
    """
    file = f'{dataset_name}_sane_ds.txt'
    return file 


def get_model_name_to_path_file() -> str:
    """
        Get the file name of the model name to path file 
    """
    return 'model_name_to_path.json'


def get_model_name_to_vocab_folder_file() -> str:
    """
        Get the file name of the model name to vocab folder file 
    """
    return 'model_name_to_vocab_folder.json'


def get_model_path(model_name: str) -> str:
    """
        Get the path to the model 
    """    
    config_folder = get_config_folder_name()
    model_name_to_path_file = get_model_name_to_path_file()
    model_name_to_path_path = os.path.join(config_folder, model_name_to_path_file)

    if model_name == 'olmo':
        model_path = "allenai/OLMo-7B-hf"
    else:
        model_path = save_load_json.load_json(model_name_to_path_path)[model_name]

    return model_path


def get_embedding_file_suffix():
    """ Get the file suffix for the embeddings """
    return 'emb'


def get_pred_is_hub_is_true_file_name(
        model_name: str, dataset_name: str) -> str:
    """ Get the file name for the prediction is hub and is true file """

    return f'pred_is_hub_is_true_{model_name}_{dataset_name}.h5'


def get_pred_prob_file_name(
        model_name: str, dataset_name: str) -> str:
    """ Get the file name for the prediction probabilities file """

    return f'pred_prob_{model_name}_{dataset_name}.h5'


def get_embedding_file_neighbour_suffix(k: int):
    """ Get the file suffix for the embeddings neighbourhoods """
    embedding_file_suff = get_embedding_file_suffix()
    neigh_suff = embedding_file_suff + f'{k}neigh'
    return neigh_suff


def get_embedding_file_neighbour_idx_suffix(k: int):
    """ Get the file suffix for the embeddings neighbourhoods """
    embedding_file_suff = get_embedding_file_suffix()
    neigh_suff = embedding_file_suff + f'{k}neigh_idx'
    return neigh_suff


def get_vocab_embedding_file_suffix():
    """ Get the file suffix for the vocab embeddings """
    return 'voc_emb'


def get_text_file_suffix():
    """ Get the file suffix for the texts """
    return 'text'


def get_next_token_predictions_name(
        model_name: str, dataset_name: str, file_suffix: str,
        batch_idx: int) -> str:
    """ Get the name of a next token predictions file """
    if batch_idx < 0:
        batch_suff = ''
    else:
        batch_suff = f'_{batch_idx}'
    file_name = f'next_token_predictions_{model_name}_{dataset_name}_{file_suffix}{batch_suff}'
    return file_name


def get_next_token_probabilities_name(
        model_name: str, dataset_name: str, file_suffix: str) -> str:
    """ Get the name of the next token probabilities file """
    file_name =f'next_token_probabilities_{model_name}_{dataset_name}_{file_suffix}.h5'
    return file_name


def get_next_token_neighbourhood_name(
        model_name: str, dataset_name: str, file_suffix: str,
        num_neighbours: int, similarity: str, context_len: int = -1):
    """ Get the name of the next token neighbourhoods file """
    
    if similarity == analysis.Similarity.SOFTMAX_DOT.value:
        sim_suffix = ''
    elif similarity == 'norm_euc':
        sim_suffix = '_' + similarity
    else:
        raise NotImplementedError(f'Similarity not implemented: {similarity}')
    if context_len > -1:
        context_suff = f'_context{context_len}'
    else:
        context_suff = ''

    file = f'next_token{sim_suffix}_{num_neighbours}neighb_{model_name}_{dataset_name}{context_suff}_{file_suffix}.h5'
    return file


def get_neighbourhood_file_name(
        representations_prefix: str, model_name: str, 
        dataset_name: str, num_neighbours: int,
        similarity: str):
    """ Get the name of the neighbourhoods file """
    if dataset_name == '':
        ds_suff = ''
    else:
        ds_suff = f'_{dataset_name}'
    file = f'{representations_prefix}_{similarity}_{num_neighbours}neighb_{model_name}{ds_suff}.h5'
    return file


def get_frequencies_file_name(model_name: str, dataset_name: str):
    """ Get the frequency file name"""
    return f'frequencies_{model_name}_{dataset_name}.h5'


def get_embeddings_hdf5_name(
        model_name: str, dataset_name: str,
        layer_number: int, file_suffix: str) -> str:
    """ Get the name of an embeddings file """    
    file_name = f'embeddings_{model_name}_{dataset_name}_l{layer_number}_{file_suffix}.h5'
    return file_name


def get_top_hubs_file_name(
        num_hubs:int, model_name: str, dataset_name: str,
        layer_numbers: List[int], file_suffix: str) -> str:
    """ Get name for file containing num_hubs top hubs for embeddings """
    layer_numbers_str = '_'.join([str(l) for l in layer_numbers])
    return f'top{num_hubs}_hubs_idx_{model_name}_{dataset_name}_l{layer_numbers_str}_{file_suffix}.h5'
