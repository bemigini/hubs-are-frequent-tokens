"""

Getting next token probabilities and predictions from models


"""

import logging

from datetime import datetime

import os
import re
from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
import torch
from tqdm import tqdm 

from src.file_handling import naming
from src.file_handling import save_load_json, save_load_hdf5
from src.hubness import analysis
from src.loading import embedding_load, model_load


def save_next_token_probabilities_for_models(
        result_folder: str, vocab_main_folder: str, 
        device: str, all_embeddings_folder: str) -> None:
    """
    Save next token probabilities for models used in experiments    
    """
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Calculating probabilities start time: %s", dt_string)

    model_names = [
        'pythia', 'pythia_step512', 'pythia_step4000', 'pythia_step16000', 'pythia_step64000', 
        'olmo', 'opt', 'mistral', 'llama']
    datasets = ['pile', 'wikitext', 'bookcorpus']
    batch_size = 128
    embedding_folder = 'embeddings'    
    config_folder = 'configs'
    
    model_name_to_vocab_folder_file = naming.get_model_name_to_vocab_folder_file()
    model_name_to_vocab_folder_path = os.path.join(config_folder, model_name_to_vocab_folder_file)
    model_name_to_vocab_folder = save_load_json.load_json(model_name_to_vocab_folder_path)    

    for current_model in model_names:        
        logging.info("Calculating probabilities: %s", current_model)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        logging.info("Time: %s", dt_string)
        vocab_folder = os.path.join(
            vocab_main_folder, model_name_to_vocab_folder[current_model])
        for current_dataset in datasets:
            save_next_token_probabilities(
                current_model, current_dataset, batch_size, result_folder, 
                unembedding_folder = vocab_folder, embedding_folder = embedding_folder,
                device=device, all_embeddings_folder = all_embeddings_folder)
        
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info("Calculating probabilities end time: %s", dt_string)


def get_next_token_probabilities_for_text(
        texts: List[str], model_name: str, 
        batch_size: int, context_len: int,
        use_cuda: bool) -> NDArray:
    """ Given a model name and a list of texts, 
        get probabilities of next tokens for the texts. 
        Texts are truncated to only use the last [context_len].
    """
    num_batches = int(np.ceil(len(texts)/batch_size))
    max_length = 200

    tokenizer = model_load.get_tokenizer(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = model_load.get_model(model_name)
    model.eval()
    
    next_token_probs = []

    for i in tqdm(range(num_batches)):
        text_batch = texts[i*batch_size:(i+1)*batch_size]

        inputs = tokenizer(text_batch, 
                            padding=True, truncation = True,
                            max_length = max_length, 
                            padding_side = 'left',
                            return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'][:, -context_len:]

        if use_cuda:
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs[0]
        
            prediction_logits = last_hidden_states.cpu().detach().numpy()       

        prediction_preds = softmax(prediction_logits[:, -1, :], axis = 1)
        
        next_token_probs.extend(prediction_preds)
    
    next_token_probs = np.array(next_token_probs)

    return next_token_probs


def save_top_n_next_token_neighbourhoods_for_text(
        texts: List[str], model_name: str, dataset_name: str, 
        batch_size: int, context_len: int, top_n: int,
        result_folder: str, use_cuda: bool) -> None:
    """ Given a model name and a list of texts, 
        get and save [top_n] neighbourhoods of next tokens for the texts. 
        Texts are truncated to [context_len].
    """
    file_suffix = 'text'
    similarity=analysis.Similarity.SOFTMAX_DOT
    file_name = naming.get_next_token_neighbourhood_name(
        model_name, dataset_name, file_suffix, top_n, 
        similarity=similarity.value, context_len = context_len)
    file_path = os.path.join(result_folder, file_name)

    if os.path.exists(file_path):
        print(f'Next token neighbourhood file already exists: {file_path}')
        return
    
    next_token_probs = get_next_token_probabilities_for_text(
        texts, model_name, batch_size, context_len, use_cuda)    
    
    args_sorted = np.argsort(next_token_probs, axis = 1)[:, ::-1]

    top_n_neighbourhoods = args_sorted[:, :top_n]
    
    h5_dataset_name = naming.get_hdf5_dataset_name()
    save_load_hdf5.save_to_hdf5(top_n_neighbourhoods, file_path, h5_dataset_name)


def get_predicted_next_tokens_for_texts(
        texts: List[str], tokenizer, model, batch_size:int) -> List[str]:
    """ Given a model, a tokenizer and a number of texts,
        get the predicted next tokens for the texts. 
    """
    num_batches = int(np.ceil(len(texts)/batch_size))

    all_predicted_tokens = []
    for i in tqdm(range(num_batches)):
        predicted_tokens = get_predicted_next_tokens_for_text_batch(
            texts, tokenizer, model, batch_size, i)
        all_predicted_tokens.extend(predicted_tokens)
    
    return predicted_tokens


def get_predicted_next_tokens_for_text_batch(
        texts, tokenizer, model, batch_size, i):
    """ Predict the next tokens for a text batch  """
    try:
        inputs = tokenizer(texts[i*batch_size:(i+1)*batch_size], 
                            padding=True, truncation = True,
                            max_length = 100, 
                            padding_side = 'left', 
                            return_tensors="pt")
        tokens = model.generate(**inputs, max_new_tokens = 1)
        text_batch_output = tokenizer.batch_decode(
            torch.reshape(tokens[:,-1], (-1, 1)))
    # pylint: disable=broad-exception-caught
    # We want to print any error, but not throw the exception
    except Exception as e:
        print('Error while predicting for batch: {i}')
        print(e)
        text_batch_output = []
        
    return text_batch_output


def save_predicted_tokens_batch(
        predicted_tokens: List[str], 
        save_folder: str,
        model_name: str, dataset_name: str, 
        file_suffix: str, batch_idx: int) -> None:
    """ Save the predicted tokens to a file """
    json_file_name = naming.get_next_token_predictions_name(
        model_name, dataset_name, file_suffix, batch_idx) + '.json'
    json_file_path = os.path.join(save_folder, json_file_name)
    save_load_json.save_as_json(predicted_tokens, json_file_path)


def predicted_tokens_file_exists( 
        save_folder: str,
        model_name: str, dataset_name: str, 
        file_suffix: str, batch_idx: int) -> bool:
    """ Returns true if the predicted tokens file exists """
    json_file_name = naming.get_next_token_predictions_name(
        model_name, dataset_name, file_suffix, batch_idx) + '.json'
    json_file_path = os.path.join(save_folder, json_file_name)
    return os.path.exists(json_file_path) 


def save_completed_file( 
        save_folder: str,
        model_name: str, dataset_name: str, 
        file_suffix: str) -> None:
    """ Saves COMPLETED file to mark all predictions were saved """
    json_file_name = naming.get_next_token_predictions_name(
        model_name, dataset_name, file_suffix, -1) + '_COMPLETE' + '.json'
    json_file_path = os.path.join(save_folder, json_file_name)
    completed = 'COMPLETED'
    save_load_json.save_as_json(completed, json_file_path)


def all_predicted_tokens_files_exist( 
        save_folder: str,
        model_name: str, dataset_name: str, 
        file_suffix: str) -> bool:
    """ Returns true if the predicted tokens COMPLETED file exists """
    json_file_name = naming.get_next_token_predictions_name(
        model_name, dataset_name, file_suffix, -1) + '_COMPLETE' + '.json'
    json_file_path = os.path.join(save_folder, json_file_name)
    return os.path.exists(json_file_path)


def load_predicted_tokens(
        save_folder: str,
        model_name: str, dataset_name: str, 
        file_suffix: str, batch_idx: int) -> List[str]:
    """ Load the predicted tokens from a file """
    json_file_name = naming.get_next_token_predictions_name(
        model_name, dataset_name, file_suffix, batch_idx) + '.json'
    json_file_path = os.path.join(save_folder, json_file_name)

    predicted_tokens = save_load_json.load_json(json_file_path)
    return predicted_tokens


def load_all_predicted_tokens(
        save_folder: str,
        model_name: str, dataset_name: str, 
        file_suffix: str) -> List[str]:
    """ Load the predicted tokens from a file """
    file_base_name = naming.get_next_token_predictions_name(
        model_name, dataset_name, file_suffix, -1)
    file_regex = fr'^{file_base_name}_+\d.json'
    files_to_load = [file for file in os.listdir(save_folder)
                     if re.search(file_regex, file) is not None 
                     and '_COMPLETE' not in file]
    # Sort files
    batch_numbers = np.array(
        [int(s.replace('.json', '').split('_')[-1]) 
         for s in files_to_load])
    sorted_idxs= np.argsort(batch_numbers)
    
    all_predicted_tokens = []
    for current_file_idx in sorted_idxs:
        json_file_path = os.path.join(save_folder, files_to_load[current_file_idx])
        all_predicted_tokens.extend(save_load_json.load_json(json_file_path))

    return all_predicted_tokens


def get_next_token_probabilities_for_embeddings(
        embeddings, batch_size, unembedding_matrix, device: str) -> NDArray:
    """ Given an unembedding matrix from a model, 
        and a number of embeddings, get probabilities of 
        next tokens for the embeddings. 
    """
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings).float()
    if isinstance(unembedding_matrix, np.ndarray):
        unembedding_matrix = torch.from_numpy(unembedding_matrix).float()
    
    embeddings.to(device)
    unembedding_matrix.to(device)

    batch_size = torch.tensor(batch_size)
    num_batches = int(torch.ceil(embeddings.shape[0]/batch_size))
    next_token_probs = torch.zeros((embeddings.shape[0], unembedding_matrix.shape[0]))
    torch_softmax = torch.nn.Softmax(dim = 1)

    for i in tqdm(range(num_batches)):
        embedding_batch = embeddings[i*batch_size:(i+1)*batch_size][:, :, None]
        unembedding_batch = torch.reshape(unembedding_matrix, (1, *unembedding_matrix.shape))

        unembedded = torch.matmul(unembedding_batch, embedding_batch).squeeze()

        softmaxed = torch_softmax(unembedded) 
        
        next_token_probs[i*batch_size:(i+1)*batch_size] = softmaxed

    next_token_probs = next_token_probs.detach().cpu().numpy()
    return next_token_probs


def save_next_token_probabilities(
        model_name: str, dataset_name: str, 
        batch_size: int, result_folder: str,
        unembedding_folder: str, embedding_folder: str,
        device: str, all_embeddings_folder: str) -> None:
    """
        Save next token probabilities of model on dataset
    """
    file_suffix = 'emb'
    file_name = naming.get_next_token_probabilities_name(model_name, dataset_name, file_suffix)
    file_path = os.path.join(result_folder, file_name)

    if os.path.exists(file_path):
        logging.info('Probabilities file already exists: %s', file_path)
        return
    
    unembedding_matrix = model_load.load_unembedding_matrix(model_name, unembedding_folder)
    embeddings = embedding_load.load_last_layer_embeddings_from_h5(
        embedding_folder, model_name, dataset_name, all_embeddings_folder)
    
    next_token_probs = get_next_token_probabilities_for_embeddings(
        embeddings, batch_size, unembedding_matrix, device=device)
    
    h5_dataset_name = naming.get_hdf5_dataset_name()
    save_load_hdf5.save_to_hdf5(next_token_probs, file_path, h5_dataset_name)


def load_next_token_probabilities(
        model_name: str, dataset_name: str, result_folder: str) -> NDArray:
    """
        Load next token probabilities of model on dataset
    """
    file_suffix = 'emb'
    file_name = naming.get_next_token_probabilities_name(model_name, dataset_name, file_suffix)
    file_path = os.path.join(result_folder, file_name)
    h5_dataset_name = naming.get_hdf5_dataset_name()

    next_token_probs = save_load_hdf5.load_from_hdf5(file_path, h5_dataset_name)

    return next_token_probs


def get_predicted_next_tokens_for_embeddings(
        embeddings, batch_size, unembedding_matrix, tokenizer) -> List[str]:
    """ Given an unembedding matrix from a model, 
        a tokenizer and a number of embeddings,
        get the predicted next tokens for the embeddings. 
    """
    num_batches = int(np.ceil(len(embeddings)/batch_size))
    predicted_tokens = []

    for i in tqdm(range(num_batches)):
        embedding_batch = embeddings[i*batch_size:(i+1)*batch_size][:, :, None]
        unembedding_batch = np.reshape(unembedding_matrix, (1, *unembedding_matrix.shape))

        unembedded = np.matmul(unembedding_batch, embedding_batch).squeeze()

        softmaxed = softmax(unembedded, axis = 1) 
        
        arg_maxes = np.argmax(softmaxed, axis=1)
        arg_max_tokens = [tokenizer.decode(t) for t in arg_maxes]
        predicted_tokens.extend(arg_max_tokens)

    return predicted_tokens


def get_knn_neighbours_of_predicted_next_tokens_for_embeddings(
        embeddings, batch_size, unembedding_matrix, tokenizer, k:int) -> List[str]:
    """ Given an unembedding matrix from a model, 
        a tokenizer and a number of embeddings,
        get the k nearest neighbours for the predicted next tokens for the embeddings. 
    """
    num_batches = int(np.ceil(len(embeddings)/batch_size))
    predicted_tokens = []

    for i in tqdm(range(num_batches)):
        embedding_batch = embeddings[i*batch_size:(i+1)*batch_size][:, :, None]
        unembedding_batch = np.reshape(unembedding_matrix, (1, *unembedding_matrix.shape))

        unembedded = np.matmul(unembedding_batch, embedding_batch).squeeze()
        softmaxed = softmax(unembedded, axis = 1) 

        k_neighbours_idxs = np.argsort(softmaxed, axis= 1)[:, -k:]        
        
        k_neighbours_tokens = [
            [tokenizer.decode(idx) for idx in kn[::-1]] for kn in k_neighbours_idxs]
        predicted_tokens.extend(k_neighbours_tokens)

    return predicted_tokens


def get_knn_neighbour_idxs_of_predicted_next_tokens_for_embeddings(
        embeddings, batch_size, unembedding_matrix, k:int) -> List[str]:
    """ Given an unembedding matrix from a model, 
        and a number of embeddings,
        get the indexes of the k nearest neighbours for the predicted 
        next tokens for the embeddings. 
    """
    num_batches = int(np.ceil(len(embeddings)/batch_size))
    predicted_tokens = []

    for i in tqdm(range(num_batches)):
        embedding_batch = embeddings[i*batch_size:(i+1)*batch_size][:, :, None]
        unembedding_batch = np.reshape(unembedding_matrix, (1, *unembedding_matrix.shape))

        unembedded = np.matmul(unembedding_batch, embedding_batch).squeeze()
        softmaxed = softmax(unembedded, axis = 1) 

        k_neighbours_idxs = np.argsort(softmaxed, axis= 1)[:, -k:]        
        
        k_neighbours_idxs = [[idx for idx in kn[::-1]] for kn in k_neighbours_idxs]
        k_neighbours_idxs = np.array(k_neighbours_idxs)
        predicted_tokens.extend(k_neighbours_idxs)

    return predicted_tokens
