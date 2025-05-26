"""

Loading used vocabulary embeddings 


"""


import os
from typing import List


import pickle 



def get_layer_vocab_embedding_filename(layer_num: int):
    """ Get filename for the vocabulary embeddings for the given layer """
    return f'{layer_num}.pickle'


def get_vocab_embeddings(vocab_folder: str, layer_numbers: List[int]):
    """ Get the vocabulary embeddings for the given layers 
    """
    embeddings = []

    for l_n in layer_numbers:
        layer_file = f'{l_n}.pickle'
        layer_path = os.path.join(vocab_folder, layer_file)

        with open(layer_path, 'rb') as j:
            layer = pickle.load(j)
        
        embeddings.append(layer)
    
    return embeddings
