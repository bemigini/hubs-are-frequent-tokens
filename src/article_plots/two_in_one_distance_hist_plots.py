"""

Make the two in one distance distribution plots


"""



import logging

import os

import numpy as np

from src.plots_tables import plotting

from src.file_handling import naming, save_load_hdf5
from src.hubness import analysis



def make_two_in_one_histograms_of_distances(
        result_folder: str) -> None:
    """ 
    Make two in one histograms of distances for example pairs
    """ 
    model_names = ['Pythia', 'Llama']
    
    datasets = ['Bookcorpus', 'Pile']
    similarity = analysis.Similarity.EUCLIDEAN
    h5_data_name = naming.get_hdf5_dataset_name()

    # token with token, pythia vs llama
    representation_prefix = 'tt'    
    
    model_distances = []
    model_max_distances = []
    for current_model in model_names:
        logging.info('%s', current_model) 
        
        dist_file_name = naming.get_distances_file_name(
            representation_prefix, current_model.lower(), '', similarity.value)
        dist_path = os.path.join(result_folder, dist_file_name)
            
        # Load distances if they exist
        # Else raise
        if os.path.exists(dist_path):
            distances = save_load_hdf5.load_from_hdf5(dist_path, h5_data_name)
        else:
            raise ValueError(f'Could not find distances: {dist_path}')
            
            
        distances = distances[~np.isnan(distances)]
        max_dist = np.max(distances)

        model_distances.append(distances)
        model_max_distances.append(max_dist)
    
    max_dist = np.max(model_max_distances)

    plot_titles = model_names
    plot_file_name = f'hist_tt_two_in_one_pythia_llama_{similarity.value}_dist.png'

    plotting.make_two_in_one_distance_hist_plot(
        model_distances, max_dist, plot_titles, plot_file_name, y_max = 5e9)
    

    # context with context, llama on bookcorpus vs pile
    representation_prefix = 'cc'
    current_model = 'Pythia'
    
    dataset_distances = []
    for current_dataset in datasets:
        logging.info('%s', current_dataset) 
        
        dist_file_name = naming.get_distances_file_name(
            representation_prefix, current_model.lower(), current_dataset.lower(), similarity.value)
        dist_path = os.path.join(result_folder, dist_file_name)
            
        # Load distances if they exist
        # Else raise
        if os.path.exists(dist_path):
            distances = save_load_hdf5.load_from_hdf5(dist_path, h5_data_name)
        else:
            raise ValueError(f'Could not find distances: {dist_path}')
            
            
        distances = distances[~np.isnan(distances)]

        dataset_distances.append(distances)

    
    max_dist = np.max(dataset_distances)    
    
    plot_titles = ['Bookcorpus', 'Pile10k']
    plot_file_name = f'hist_cc_two_in_one_{current_model}_pile_bookcorpus_{similarity.value}_dist.png'

    plotting.make_two_in_one_distance_hist_plot(
        dataset_distances, max_dist, plot_titles, plot_file_name, y_max=0.0)
