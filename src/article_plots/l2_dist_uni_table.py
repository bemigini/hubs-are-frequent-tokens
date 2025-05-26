"""

Make a table showing the L2 distances to the uniform distribution for the models


"""



import os 

import numpy as np
import pandas as pd 

from src.file_handling import naming
from src.plots_tables import make_table



def get_l2_dist_uni_latex_table(result_folder: str):
    """ 
         Get table with L2 distances to uniform distribution for softmax dot comparisons
    """

    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']
    
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']

    representation_types = ['ct', 'tt', 'cc']

    l2_dists_dict = {
        'model_name': [],
        'context': [],
        'comparison_type': [],
        'mean L2 distance to uniform': []
    }

    for model_name in model_names:
        for dataset_name in hub_datasets:
            for representation_prefix in representation_types:
                L_2_distances_to_uni_file = naming.get_dist_to_uni_file_name(
                    representation_prefix, model_name, dataset_name)
                L_2_distances_to_uni_path = os.path.join(result_folder, L_2_distances_to_uni_file)

                current_l2_df = pd.read_csv(L_2_distances_to_uni_path, index_col = 0)
                mean_l2_dist = np.mean(current_l2_df)

                l2_dists_dict['model_name'].append(model_name)
                l2_dists_dict['context'].append(dataset_name)
                l2_dists_dict['comparison_type'].append(representation_prefix)
                l2_dists_dict['mean L2 distance to uniform'].append(mean_l2_dist)

    l2_df = pd.DataFrame(l2_dists_dict)

    latex_str = make_table.get_L2_dist_table_as_latex_strings_from_df(l2_df)

    return latex_str
