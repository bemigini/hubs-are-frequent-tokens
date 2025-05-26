"""

Loading the Pythia model and getting (un)embedding matrix


"""


import os
import pickle

from numpy.typing import NDArray
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM

from src.file_handling import naming, save_load_json
from src.loading import util




def load_unembedding_matrix(model_name: str, load_folder: str) -> NDArray:
    """ Load the unembedding matrix of the model """
    if model_name == 'opt':
        unembedding_matrix_file = 'emb.pickle'
    else:
        unembedding_matrix_file = 'unemb.pickle'
    unembedding_matrix_path = os.path.join(load_folder, unembedding_matrix_file)
    with open(unembedding_matrix_path, 'rb') as j:
        X_out = pickle.load(j)

    len_voc = util.get_vocab_length(model_name)
    unembedding_matrix = X_out[:len_voc]

    return unembedding_matrix


def get_token_to_id_dict(model_name: str, vocab_main_folder: str) -> dict:
    """ Get the token to id dictionary for the model"""
    config_folder = 'configs'
    model_name_to_vocab_folder_file = naming.get_model_name_to_vocab_folder_file()
    model_name_to_vocab_folder_path = os.path.join(config_folder, model_name_to_vocab_folder_file)
    model_name_to_vocab_folder = save_load_json.load_json(model_name_to_vocab_folder_path)   
    
    vocab_folder = os.path.join(
        vocab_main_folder, model_name_to_vocab_folder[model_name])
    vocab_file = 'voc.pickle'

    vocab_path = os.path.join(vocab_folder, vocab_file)
    with open(vocab_path, 'rb') as f:
        X_voc = pickle.load(f)
    
    return X_voc


def get_id_to_token_dict(model_name: str, vocab_main_folder: str) -> dict:
    """ Get the id to token dictionary for the model"""
    X_voc = get_token_to_id_dict(model_name, vocab_main_folder)

    id_to_token_dict = {X_voc[k]:k for k in X_voc}
    
    return id_to_token_dict


def get_id_to_decoded_token_dict(model_name: str, vocab_main_folder: str) -> dict:
    """ Get the id to decoded token dictionary for the model"""
    id_to_token_dict = get_id_to_token_dict(model_name, vocab_main_folder)

    tokenizer = get_tokenizer(model_name)

    X_voc_decoded = {k:tokenizer.decode([k]) for k in id_to_token_dict}
    
    return X_voc_decoded


def get_decoded_token_to_id_dict(model_name: str, vocab_main_folder: str) -> dict:
    """ Get the token to id dictionary for the model, where the token is decoded """
    raise NotImplementedError(
        "For some models several token ids are decoded to the same string. So this would 'collapse' some tokens.")


def get_tokenizer(model_name: str):
    """ Get tokenizer of the model """
    model_path = naming.get_model_path(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def get_model(model_name: str):
    """ Get the model """
    model_path = naming.get_model_path(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, load_in_8bit = True, torch_dtype = torch.float16)
    return model


def get_pythia_model_name():
    """ The name of the pythia model used """
    return "EleutherAI/pythia-6.9b-deduped"


def load_pythia_model():
    """ Load the pythia model used """
    model = GPTNeoXForCausalLM.from_pretrained(get_pythia_model_name())
    return model 


def load_pythia_tokenizer():
    """ Load the pythia tokenizer used """
    tokenizer = AutoTokenizer.from_pretrained(get_pythia_model_name())
    return tokenizer 


def get_embedding_and_unembedding_matrices(model:GPTNeoXForCausalLM):
    """ Get the embedding and unembedding matrices of the model """
    embedding_matrix = model.get_input_embeddings().weight.data.detach().numpy()
    unembedding_matrix = model.get_output_embeddings().weight.data.detach().numpy()

    return embedding_matrix, unembedding_matrix


def get_cropped_embedding_and_unembedding_matrices(model:GPTNeoXForCausalLM, model_name: str):
    """ Get the embedding and unembedding matrices of the model cropped to the vocab length """
    embedding_matrix = model.get_input_embeddings().weight.data.detach().numpy()
    unembedding_matrix = model.get_output_embeddings().weight.data.detach().numpy()

    vocab_len = util.get_vocab_length(model_name)
    embedding_matrix = embedding_matrix[:vocab_len]
    unembedding_matrix = unembedding_matrix[:vocab_len]

    return embedding_matrix, unembedding_matrix


def get_cropped_unembedding_matrix(model:GPTNeoXForCausalLM, model_name: str):
    """ Get the unembedding matrix of the model cropped to the vocab length """
    unembedding_matrix = model.get_output_embeddings().weight.data.detach().numpy()

    vocab_len = util.get_vocab_length(model_name)
    unembedding_matrix = unembedding_matrix[:vocab_len]

    return unembedding_matrix
