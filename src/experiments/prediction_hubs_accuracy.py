"""

Experiment to show the accuracy of the model predictions which are also hubs


"""

import logging

from datetime import datetime

import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from src import predictions

from src.file_handling import naming, save_load_hdf5
from src.hubness import analysis
from src.loading import model_load, text_load


def save_pred_is_hub_is_true_and_prob_files(
        result_folder: str, vocab_main_folder: str) -> None: 
    """ Make and save the 
        pred_is_hub_is_true (whether prediction is a hub and whether it is true) 
        and pred_prob (prediction probability) files 
    """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment start prediction_hubs_accuracy: %s", dt_string)

    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']
        
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']
    num_neighbours = 10
    pred_similarity = analysis.Similarity.SOFTMAX_DOT

    h_5_data_name = naming.get_hdf5_dataset_name()


    for model_name in model_names:
        logging.info('%s', model_name)
        for dataset_name in hub_datasets:
            hub_info_file_name = naming.get_hub_info_file_name(
            'ct', model_name, dataset_name, dataset_name, pred_similarity.value, num_neighbours)
            hub_info_path = os.path.join(result_folder, hub_info_file_name)
            hub_info = pd.read_csv(hub_info_path, index_col=0)
            hub_idxs = hub_info['hub_idx'].values

            next_probs_arr = predictions.load_next_token_probabilities(
                model_name, dataset_name, result_folder)
            
            predicted_tokens = np.argmax(next_probs_arr, axis=1)

            # Get the actual next tokens
            actual_next_tokens = text_load.get_texts_next_word('', dataset_name)
            
            id_to_token = model_load.get_id_to_decoded_token_dict(
                model_name, vocab_main_folder)
            
            pred_is_hub_is_true_array = np.zeros((next_probs_arr.shape[0], 3), dtype=int)
            pred_prob = np.zeros(next_probs_arr.shape[0])

            for i, current_pred_id in tqdm(enumerate(predicted_tokens)):
                current_prob = next_probs_arr[i, current_pred_id]
                current_pred_str = id_to_token[current_pred_id]
                
                true_str = actual_next_tokens[i]
                
                is_true_pred = (
                    (current_pred_str == true_str) 
                    or (current_pred_str == ' ' + true_str))

                # Get whether prediction is a hub
                pred_is_hub = current_pred_id in hub_idxs

                pred_is_hub_is_true_array[i] = np.array(
                    [current_pred_id, pred_is_hub, is_true_pred])
                pred_prob[i] = current_prob
            
            pred_is_hub_is_true_file_name = naming.get_pred_is_hub_is_true_file_name(
                model_name, dataset_name)
            pred_is_hub_is_true_path = os.path.join(result_folder, pred_is_hub_is_true_file_name)
            
            save_load_hdf5.save_to_hdf5(
                pred_is_hub_is_true_array, pred_is_hub_is_true_path, h_5_data_name)

            pred_prob_file_name = naming.get_pred_prob_file_name(
                model_name, dataset_name)
            pred_prob_path = os.path.join(result_folder, pred_prob_file_name)
            
            save_load_hdf5.save_to_hdf5(pred_prob, pred_prob_path, h_5_data_name)
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Experiment end prediction_hubs_accuracy: %s", dt_string)


def print_general_accuracy_vs_hub_accuracy(result_folder: str) -> None:
    """ Print general accuracy of models vs the sccuracy of hub predictions """
    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']        
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']

    h_5_data_name = naming.get_hdf5_dataset_name()


    for model_name in model_names:
        for dataset_name in hub_datasets:
            pred_is_hub_is_true_file_name = naming.get_pred_is_hub_is_true_file_name(
                model_name, dataset_name)
            pred_is_hub_is_true_path = os.path.join(result_folder, pred_is_hub_is_true_file_name)
            
            pred_is_hub_is_true_array = save_load_hdf5.load_from_hdf5(
                pred_is_hub_is_true_path, h_5_data_name)
            
            general_correct = (pred_is_hub_is_true_array[:, 2]).sum()
            general_accuracy = general_correct/pred_is_hub_is_true_array.shape[0]
            hub_filter = pred_is_hub_is_true_array[:, 1] == 1
            hub_correct = (pred_is_hub_is_true_array[:, 2][hub_filter]).sum()
            hub_accuracy = hub_correct/hub_filter.sum()

            not_hub_filter = (pred_is_hub_is_true_array[:, 1] == 0)
            not_hub_correct = (pred_is_hub_is_true_array[:, 2][not_hub_filter]).sum()
            not_hub_accuracy = not_hub_correct/not_hub_filter.sum()

            logging.info('For %s:', f'{model_name} on {dataset_name}')
            logging.info('General accuracy: %s', general_accuracy)
            logging.info('Hub accuracy: %s', hub_accuracy)
            logging.info('Non-hub accuracy: %s', not_hub_accuracy)
            logging.info('----------------------------------------------')
