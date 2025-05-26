"""

For saving and loading hdf5 files


"""


import h5py

from numpy.typing import NDArray



def save_to_hdf5(data: NDArray, h5_file_path: str, h5_dataset_name: str) -> None:
    """ Save a matrix in hdf5 format """
    with h5py.File(h5_file_path, 'w') as h5f:
        h5f.create_dataset(h5_dataset_name, data=data, compression='gzip')
    

def load_from_hdf5(h5_file_path: str, dataset_name: str) -> NDArray:
    """ Load a hdf5 file """
    with h5py.File(h5_file_path, 'r') as h5f:
        data = h5f[dataset_name][:]
    
    return data
