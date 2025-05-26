"""

Utility functions for loading objects


"""



def get_vocab_length(model_name: str) -> int:
    """ Get the vocabulary length of the model """
    if model_name in ('pythia', 'pythia_step512', 'pythia_step4000', 'pythia_step16000', 'pythia_step64000'):
        return 50277
    if model_name == 'olmo':
        return 50280
    if model_name == 'opt':
        return 50265
    if model_name == 'mistral':
        return 32000
    if model_name == 'llama':
        return 128256

    raise NotImplementedError(f'Model is not implemented: {model_name}')


def get_last_layer_idx(model_name: str) -> int:
    """
        Get the index of the last layer of the model
    """
    if model_name in ('pythia', 'pythia_step512', 'pythia_step4000', 'pythia_step16000', 'pythia_step64000', 'olmo', 'opt', 'mistral', 'llama'):
        last_layer_idx = 32
    else:
        raise NotImplementedError(f'Model not implemented: {model_name}')

    return last_layer_idx
