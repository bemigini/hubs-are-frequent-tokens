"""

Making latex tables


"""


from typing import Tuple

import pandas as pd 


def get_context_dataset_dict():
    """ Get dictionary of context names """
    d = {
        'pile': 'Pile10k',
        'bookcorpus': 'Bookcorpus',
        'wikitext': 'WikiText-103'
        }
    return d


def get_frequency_dataset_dict():
    """ Get dictionary of frequency dataset names """
    d = {
        'pile': 'Pile10k',
        'bookcorpus': 'Bookcorpus',
        'wikitext': 'WikiText-103',
        'train_dataset': 'train_dataset'
        }
    return d


def get_model_dict():
    """ Get dictionary of model names """
    d = {
        'pythia': 'Pythia',
        'olmo': 'OLMo',
        'opt': 'Opt',
        'mistral': 'Mistral',
        'llama': 'Llama',
        'pythia_step512': '512',
        'pythia_step4000': '4000',
        'pythia_step16000': '16000',
        'pythia_step64000': '64000'
        }
    return d


def get_hub_ocurr_and_freq_tables_as_latex_strings_from_df(
        df: pd.DataFrame, representation_prefix: str) -> Tuple[str, str]:
    """ Get the dataframe as a latex string """
    # Make hub occurrence table 
    m_dict = get_model_dict()
    context_dict = get_context_dataset_dict()
    freq_dict = get_frequency_dataset_dict()

    hub_occ_df = df[['model_name', 'similarity', 'dataset', 'number_of_hubs', 'k-skew',
       'median_N_k', 'mean_N_k', 'max_N_k', 'var_N_k']].drop_duplicates()
    if representation_prefix == 'ct':
        hub_occ_df = hub_occ_df.drop(columns = 'similarity')    

    hub_occ_df['model_name'] = hub_occ_df['model_name'].apply(lambda s: m_dict[s])
    
    if representation_prefix == 'tt':
        hub_occ_df = hub_occ_df.drop(columns = 'dataset')
    else:
        hub_occ_df['dataset'] = hub_occ_df['dataset'].apply(lambda s: context_dict[s])

    hub_occurrence_latex = hub_occ_df.to_latex(index=False, float_format="{:.2f}".format)
    hub_occurrence_latex = hub_occurrence_latex.replace('_', ' ')
    hub_occurrence_latex = hub_occurrence_latex.replace(r'\toprule', r'\hline')
    hub_occurrence_latex = hub_occurrence_latex.replace(r'\midrule', r'\hline')
    hub_occurrence_latex = hub_occurrence_latex.replace(r'\bottomrule', r'\hline')
    hub_occurrence_latex = hub_occurrence_latex.replace('euclidean', 'euc')
    hub_occurrence_latex = hub_occurrence_latex.replace('NaN', '-')


    if representation_prefix in ['ct', 'tt']:
        # Make frequency correlation table
        freq_df = df[
            ['model_name', 'similarity', 'dataset', 'freq_from', 'spearman_corr']].drop_duplicates()

        if representation_prefix == 'ct':
            freq_df = freq_df.drop(columns = 'similarity')
        
        freq_df['model_name'] = freq_df['model_name'].apply(lambda s: m_dict[s])        
        freq_df['freq_from'] = freq_df['freq_from'].apply(lambda s: freq_dict[s])

        if representation_prefix == 'tt':
            freq_df = freq_df.drop(columns = 'dataset')
        else:
            freq_df['dataset'] = freq_df['dataset'].apply(lambda s: context_dict[s])

        frequency_latex = freq_df.to_latex(index=False, float_format="{:.2f}".format)
        frequency_latex = frequency_latex.replace('_', ' ')
        frequency_latex = frequency_latex.replace(r'\toprule', r'\hline')
        frequency_latex = frequency_latex.replace(r'\midrule', r'\hline')
        frequency_latex = frequency_latex.replace(r'\bottomrule', r'\hline')
        frequency_latex = frequency_latex.replace('euclidean', 'euc')
        frequency_latex = frequency_latex.replace('NaN', '-')
        
    else:
        frequency_latex = ''

    return hub_occurrence_latex, frequency_latex


def get_L2_dist_table_as_latex_strings_from_df(
        l2_df: pd.DataFrame) -> str:
    """ Get the dataframe as a latex string """
    m_dict = get_model_dict()
    context_dict = get_context_dataset_dict()

    l2_df['model_name'] = l2_df['model_name'].apply(lambda s: m_dict[s])
    l2_df['context'] = l2_df['context'].apply(lambda s: context_dict[s] if s != '' else s)
       

    l2_latex = l2_df.to_latex(index=False, float_format="{:.2f}".format)
    l2_latex = l2_latex.replace('_', ' ')
    l2_latex = l2_latex.replace(r'\toprule', r'\hline')
    l2_latex = l2_latex.replace(r'\midrule', r'\hline')
    l2_latex = l2_latex.replace(r'\bottomrule', r'\hline')
    l2_latex = l2_latex.replace('euclidean', 'euc')
    l2_latex = l2_latex.replace('NaN', '-')


    return l2_latex


def get_hub_acc_table_as_latex_strings_from_df(
        hub_acc_df: pd.DataFrame) -> str:
    """ Get the dataframe as a latex string """
    m_dict = get_model_dict()
    context_dict = get_context_dataset_dict()

    hub_acc_df['model_name'] = hub_acc_df['model_name'].apply(lambda s: m_dict[s])
    hub_acc_df['context'] = hub_acc_df['context'].apply(lambda s: context_dict[s] if s != '' else s)
       

    latex = hub_acc_df.to_latex(index=False, float_format="{:.2f}".format)
    latex = latex.replace('_', ' ')
    latex = latex.replace(r'\toprule', r'\hline')
    latex = latex.replace(r'\midrule', r'\hline')
    latex = latex.replace(r'\bottomrule', r'\hline')
    latex = latex.replace('euclidean', 'euc')
    latex = latex.replace('NaN', '-')


    return latex
