"""

Making the plot info table



"""

import os 

from typing import List

import numpy as np
import pandas as pd 


from src.file_handling import naming, save_load_json
from src.hubness import analysis

from src.plots_tables import make_table


def make_hub_info_dicts_all_models(
    result_folder: str) -> None:
    """
        Make all hub info dicts
    """
    k_skew_file = naming.get_k_skew_all_file_name()
    k_skew_path = os.path.join(result_folder, k_skew_file)

    k_skew_dict = save_load_json.load_json(k_skew_path)
    k_skew_df = pd.DataFrame(k_skew_dict)

    num_neighbours = 10
    
    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']
    datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN,
        analysis.Similarity.SOFTMAX_DOT]
    
    make_hub_info_dicts(
        result_folder,
        model_names, datasets, similarities,
        num_neighbours, 'all', k_skew_df
    )


def make_hub_info_dicts_pythia_checkpoints(
    result_folder: str) -> None:
    """
        Make hub info dicts for pythia checkpoints
    """
    k_skew_file = naming.get_k_skew_pythia_steps_file_name()
    k_skew_path = os.path.join(result_folder, k_skew_file)

    k_skew_dict = save_load_json.load_json(k_skew_path)
    k_skew_df = pd.DataFrame(k_skew_dict)

    num_neighbours = 10
    
    model_names = ['pythia_step512', 'pythia_step4000', 'pythia_step16000', 'pythia_step64000']
    datasets = ['pile', 'wikitext', 'bookcorpus']
    similarities = [
        analysis.Similarity.EUCLIDEAN, 
        analysis.Similarity.NORM_EUCLIDEAN,
        analysis.Similarity.SOFTMAX_DOT]
    
    make_hub_info_dicts(
        result_folder, 
        model_names, datasets, similarities,
        num_neighbours, 'pythia_checkpoints', k_skew_df
    )


def make_hub_info_dicts(
    result_folder: str,
    model_names: List[str], datasets: List[str], similarities: List[analysis.Similarity],
    num_neighbours: int, all_or_checkpoints: str, 
    k_skew_df) -> None:
    """
        Make hub info dicts
    """    
    
    representation_prefix = 'tt'
    
    tt_hub_info_df_dict = {
        'model_name': [],
        'similarity': [],
        'dataset': [],
        'freq_from': [],
        'number_of_hubs': [],
        'k-skew': [],
        'median_N_k': [],
        'mean_N_k': [],
        'max_N_k': [],
        'var_N_k': [],
        'median_token_freq': [],
        'mean_token_freq': [],
        'max_token_freq': [],
        'var_token_freq': [],
        'num_g_zero_freqs': [],
        'spearman_corr': []
    }

    for current_model in model_names:        
        for current_sim in similarities:            
            for dataset_name in datasets:
                tt_hub_info_df_dict['model_name'].append(current_model)
                tt_hub_info_df_dict['similarity'].append(current_sim.value)
                tt_hub_info_df_dict['dataset'].append('')
                tt_hub_info_df_dict['freq_from'].append(dataset_name)

                current_k_skew = k_skew_df[(
                    (k_skew_df['model_name'] == current_model) 
                    & (k_skew_df['dataset_name'] == '') 
                    & (k_skew_df['similarity'] == current_sim.value)
                    & (k_skew_df['representation_prefix'] == representation_prefix))]['k-skew'].values[0]
                tt_hub_info_df_dict['k-skew'].append(current_k_skew)

                tt_hub_info_df_dict = get_hub_info_dict_for_model(
                    current_sim.value, num_neighbours, result_folder, 
                    current_model, '', 'tt', dataset_name, 
                    dict_to_append_to = tt_hub_info_df_dict)
    
    tt_hub_dict_file = naming.get_hub_info_table_file_name(num_neighbours, 'tt', all_or_checkpoints)
    tt_hub_dict_path = os.path.join(result_folder, tt_hub_dict_file)
    tt_df = pd.DataFrame(tt_hub_info_df_dict)
    tt_df.to_csv(tt_hub_dict_path, index=False)


    representation_prefix = 'ct'
    
    ct_hub_info_df_dict = {
        'model_name': [],
        'similarity': [],
        'dataset': [],
        'freq_from': [],
        'number_of_hubs': [],
        'k-skew': [],
        'median_N_k': [],
        'mean_N_k': [],
        'max_N_k': [],
        'var_N_k': [],
        'median_token_freq': [],
        'mean_token_freq': [],
        'max_token_freq': [],
        'var_token_freq': [],
        'num_g_zero_freqs': [],
        'spearman_corr': []
    }

    for current_model in model_names:        
        for dataset_name in datasets:            
            for get_frequencies_from in datasets + ['train_dataset']:
                if (get_frequencies_from == 'train_dataset' 
                    and (not (current_model in ['pythia', 'olmo']))):
                    continue
                ct_hub_info_df_dict['model_name'].append(current_model)
                ct_hub_info_df_dict['similarity'].append(analysis.Similarity.SOFTMAX_DOT.value)
                ct_hub_info_df_dict['dataset'].append(dataset_name)
                ct_hub_info_df_dict['freq_from'].append(get_frequencies_from)  

                current_k_skew = k_skew_df[(
                    (k_skew_df['model_name'] == current_model) 
                    & (k_skew_df['dataset_name'] == dataset_name) 
                    & (k_skew_df['similarity'] == analysis.Similarity.SOFTMAX_DOT.value)
                    & (k_skew_df['representation_prefix'] == representation_prefix))]['k-skew'].values[0]
                ct_hub_info_df_dict['k-skew'].append(current_k_skew)

                ct_hub_info_df_dict = get_hub_info_dict_for_model(
                    analysis.Similarity.SOFTMAX_DOT.value, num_neighbours, result_folder, 
                    current_model, dataset_name, 'ct', get_frequencies_from,
                    dict_to_append_to= ct_hub_info_df_dict)
    
    ct_hub_dict_file = naming.get_hub_info_table_file_name(num_neighbours, 'ct', all_or_checkpoints)
    ct_hub_dict_path = os.path.join(result_folder, ct_hub_dict_file)
    ct_df = pd.DataFrame(ct_hub_info_df_dict)
    ct_df.to_csv(ct_hub_dict_path, index=False)
   
    
    representation_prefix = 'cc'
    
    cc_hub_info_df_dict = {
        'model_name': [],
        'similarity': [],
        'dataset': [],
        'number_of_hubs': [],
        'k-skew': [],
        'median_N_k': [],
        'mean_N_k': [],
        'max_N_k': [],
        'var_N_k': []
    }

    for current_model in model_names:        
        for similarity in similarities:            
            for dataset_name in datasets:
                cc_hub_info_df_dict['model_name'].append(current_model)
                cc_hub_info_df_dict['similarity'].append(similarity.value)
                cc_hub_info_df_dict['dataset'].append(dataset_name)

                current_k_skew = k_skew_df[(
                    (k_skew_df['model_name'] == current_model) 
                    & (k_skew_df['dataset_name'] == dataset_name) 
                    & (k_skew_df['similarity'] == similarity.value)
                    & (k_skew_df['representation_prefix'] == representation_prefix))]['k-skew'].values[0]
                cc_hub_info_df_dict['k-skew'].append(current_k_skew)

                cc_hub_info_df_dict = get_hub_info_dict_for_model(
                    similarity.value, num_neighbours, result_folder, 
                    current_model, dataset_name, 'cc', '',
                    dict_to_append_to = cc_hub_info_df_dict)
    
    cc_hub_dict_file = naming.get_hub_info_table_file_name(num_neighbours, 'cc', all_or_checkpoints)
    cc_hub_dict_path = os.path.join(result_folder, cc_hub_dict_file)
    cc_df = pd.DataFrame(cc_hub_info_df_dict)
    cc_df.to_csv(cc_hub_dict_path, index=False)


def get_hub_info_dict_for_model(
    similarity: str,
    num_neighbours: int, result_folder: str, 
    model_name: str, dataset_name: str,
    representation_prefix: str,
    get_frequencies_from: str,
    dict_to_append_to: dict
    ) -> None:
    """
        Get hub info dict
    """ 

    file_name = naming.get_hub_info_file_name(
        representation_prefix, model_name, dataset_name, 
        get_frequencies_from, similarity, num_neighbours)
    path = os.path.join(result_folder, file_name)

    if os.path.exists(path):
        print(f'Loading hub info from {path}')
        df = pd.read_csv(path)
        top_N_k_idx = df['hub_idx'].values
        N_ks = df['N_k'].values

        if 'token_freq' in df.columns:
            hub_token_frequencies = df['token_freq'].values

            if hub_token_frequencies.shape[0] > 0:
                median_token_freq = np.median(hub_token_frequencies)
                mean_token_freq = hub_token_frequencies.mean()
                max_token_freq = hub_token_frequencies.max()
                var_token_freq = hub_token_frequencies.var()
                num_g_zero_freqs = (hub_token_frequencies > 0).sum()
            else:
                median_token_freq = np.nan
                mean_token_freq = np.nan
                max_token_freq = np.nan
                var_token_freq = np.nan
                num_g_zero_freqs = np.nan

            spearman_corr = df[['N_k','token_freq']].corr(method='spearman')
            
            dict_to_append_to['median_token_freq'].append(median_token_freq)
            dict_to_append_to['mean_token_freq'].append(mean_token_freq)
            dict_to_append_to['max_token_freq'].append(max_token_freq)
            dict_to_append_to['var_token_freq'].append(var_token_freq)
            dict_to_append_to['num_g_zero_freqs'].append(num_g_zero_freqs)
            dict_to_append_to['spearman_corr'].append(spearman_corr.iloc[0,1])
        
    else:
        raise ValueError(f'Could not find hub info file: {path}')
    
    number_of_hubs = len(top_N_k_idx)
    if number_of_hubs > 0:
        median_N_k = np.median(N_ks)
        mean_N_k = N_ks.mean()
        max_N_k = N_ks.max()
        var_N_k = N_ks.var()
    else:
        median_N_k = np.nan
        mean_N_k = np.nan
        max_N_k = np.nan
        var_N_k = np.nan
    
    dict_to_append_to['number_of_hubs'].append(number_of_hubs)
    dict_to_append_to['median_N_k'].append(median_N_k)
    dict_to_append_to['mean_N_k'].append(mean_N_k)
    dict_to_append_to['max_N_k'].append(max_N_k)
    dict_to_append_to['var_N_k'].append(var_N_k)    

    return dict_to_append_to


def get_hub_info_table_latex(
    result_folder: str, representation_prefix: str, num_neighbours: int,
    all_or_checkpoints: str) -> str:
    """
        Make the hub info table
    """

    hub_dict_file = naming.get_hub_info_table_file_name(
        num_neighbours, representation_prefix, all_or_checkpoints)
    hub_dict_path = os.path.join(result_folder, hub_dict_file)
    hub_info_df = pd.read_csv(hub_dict_path)

    return make_table.get_hub_ocurr_and_freq_tables_as_latex_strings_from_df(
        hub_info_df, representation_prefix)
    