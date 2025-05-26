"""

Loading used texts 


"""


import os


import numpy as np

from src.file_handling import naming, save_load_json


def get_pile_text_filename():
    """ Get filename of the pile texts """
    return 'pile_sane_ds.txt'


def get_wiki_text_filename():
    """ Get filename of the wikitext texts """
    return 'wikitext_sane_ds.txt'


def get_texts(text_folder:str, dataset: str):
    """ Get the texts from the dataset as bytes, 
        split into context and next word 
    """
    use_text_filename = naming.get_dataset_file_name(dataset)

    text_filename = os.path.join(text_folder, use_text_filename)

    with open(text_filename,'rb') as f:
        text_lines = f.readlines()

    for i, line in enumerate(text_lines):
        text_lines[i] = line.rstrip(b'\n')

    text_lines_np = np.array(text_lines)

    split_text_lines_np = np.char.split(text_lines_np, [b'\t'])
    split_text_lines_np = np.array([np.array(l) for l in split_text_lines_np])

    return split_text_lines_np


def get_texts_next_word(text_folder:str, dataset: str) -> str:
    """ Get the next word from the texts """
    true_next_word_file_name = naming.get_true_next_word_file_name(dataset)
    meta_data_folder = 'meta_data'
    true_next_word_file_path = os.path.join(meta_data_folder, true_next_word_file_name)

    if os.path.exists(true_next_word_file_path):
        next_words = save_load_json.load_json(true_next_word_file_path)
        next_words = np.array(next_words)
    else:
        if text_folder == '':
            raise ValueError(
                f'Text folder not given when true_next_word_file_path did not exist: {true_next_word_file_path}')
        split_text_lines_np = get_texts(text_folder, dataset)
        next_words = split_text_lines_np[:,1]
        next_words = np.char.decode(next_words, encoding='utf-8')
        save_load_json.save_as_json(next_words, true_next_word_file_path)

    return next_words
