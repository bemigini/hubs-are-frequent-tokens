"""

Making a table over acurracies for hubs  and non-hubs for models 


"""



import os 

import pandas as pd 

from src.file_handling import naming, save_load_hdf5
from src.plots_tables import make_table



def get_hub_accuracy_latex_table(result_folder: str):
    """ 
         Get table with hub and non-hub accuracies
    """

    model_names = ['pythia', 'olmo', 'opt', 'mistral', 'llama']        
    hub_datasets = ['pile', 'wikitext', 'bookcorpus']

    h_5_data_name = naming.get_hdf5_dataset_name()

    hub_acc_dict = {
        'model_name': [],
        'context': [],
        'general acc': [],
        'hub acc': [],
        'non-hub acc': []
    }

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

            hub_acc_dict['model_name'].append(model_name)
            hub_acc_dict['context'].append(dataset_name)
            hub_acc_dict['general acc'].append(general_accuracy)
            hub_acc_dict['hub acc'].append(hub_accuracy)
            hub_acc_dict['non-hub acc'].append(not_hub_accuracy)

    hub_acc_df = pd.DataFrame(hub_acc_dict)

    latex_str = make_table.get_hub_acc_table_as_latex_strings_from_df(hub_acc_df)

    return latex_str


