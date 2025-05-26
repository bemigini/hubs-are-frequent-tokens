"""

run.py: Run Script for hubness and token frequency experiments.


Usage:
    run.py save-next-token-probs --output-folder=<file> --vocab-main-folder=<file> --all-emb-folder=<file> [options] 
    run.py pred-hub-Nk-vs-freq-plot --output-folder=<file> [options] 
    run.py tt-hub-Nk-vs-freq-plot --output-folder=<file> --vocab-main-folder=<file> [options]
    run.py cc-hub-Nk-info --output-folder=<file> [options]
    run.py distance-distributions --output-folder=<file> --vocab-main-folder=<file> [options] 
    run.py plot-k-occurrence --output-folder=<file> --vocab-main-folder=<file> [options] 
    run.py get-hub-examples --output-folder=<file> --vocab-main-folder=<file> --copora-folder=<file> [options] 
    
Options:
    -h --help                               show this screen.
    --output-folder=<file>                  folder to save output in
    --config-folder=<file>                  folder to load config files from    
    --vocab-main-folder=<file>              the main folder for loading vocabularies
    --copora-folder=<file>                  folder to get coporas from
    --num-neighbours=<int>                  number of neighbours to use in the neighbourhoods [default: 10]
    --num-top-hubs=<int>                    number of largest hubs to take for examples [default: 20]
    --skip-calcs=<string>                   calculations/plots to skip separated by _ e.g. ct_tt will skip context-token and token-token plots, e.g [default: None]
    --cuda                                  use GPU 
    --log=<string>                          log level to use [default: info]
    
"""



import logging

from typing import Dict

from docopt import docopt

from src.experiments import context_to_context_hubs
from src.experiments import distribution_of_distances
from src.experiments import hub_examples
from src.experiments import plot_k_occurrence_distributions
from src.experiments import prediction_hubs_vs_frequent_tokens
from src.experiments import token_to_token_hubs

from src import predictions


def save_next_token_probabilities(args: Dict):
    """ Save next token probabilities for models """

    output_folder = args['--output-folder'] if args['--output-folder'] else '.'
    vocab_main_folder = args['--vocab-main-folder'] if args['--vocab-main-folder'] else ''
    all_embeddings_folder = args['--all-emb-folder'] if args['--all-emb-folder'] else ''

    device = "cuda" if args['--cuda'] else "cpu"

    logging.info('save_next_token_probabilities_for_models')

    predictions.save_next_token_probabilities_for_models(
        result_folder=output_folder, vocab_main_folder=vocab_main_folder, 
        device=device, all_embeddings_folder=all_embeddings_folder)


def pred_hub_N_k_vs_token_freqs(args: Dict):
    """ Make and save prediction hub k-occurrence vs token frequency plots """
    
    output_folder = args['--output-folder'] if args['--output-folder'] else '.'
    num_neighbours = int(args['--num-neighbours']) if args['--num-neighbours'] else 10

    logging.info('plot_hub_N_k_vs_token_frequencies')

    prediction_hubs_vs_frequent_tokens.plot_hub_N_k_vs_token_frequencies(
        result_folder=output_folder,
        num_neighbours=num_neighbours)
    
    prediction_hubs_vs_frequent_tokens.plot_hub_N_k_vs_token_train_frequencies(
        result_folder=output_folder,
        num_neighbours=num_neighbours)
    
    prediction_hubs_vs_frequent_tokens.plot_hub_N_k_vs_token_frequencies_checkpoints(
        result_folder=output_folder,
        num_neighbours=num_neighbours)


def token_token_hub_N_k_vs_token_freqs(args: Dict):
    """ Make and save token to token hub k-occurrence vs token frequency plots """
    
    output_folder = args['--output-folder'] if args['--output-folder'] else '.'
    num_neighbours = int(args['--num-neighbours']) if args['--num-neighbours'] else 10
    vocab_main_folder = args['--vocab-main-folder'] if args['--vocab-main-folder'] else ''

    device = "cuda" if args['--cuda'] else "cpu"

    logging.info('plot_token_to_token_hub_N_k_vs_token_frequencies_for_all')

    token_to_token_hubs.plot_token_to_token_hub_N_k_vs_token_frequencies_for_all(
        result_folder=output_folder,
        num_neighbours=num_neighbours,
        vocab_main_folder = vocab_main_folder,
        device=device)


def context_context_hub_N_k_info(args: Dict):
    """ Make and save context to context hub k-occurrence vs token frequency plots """
    
    output_folder = args['--output-folder'] if args['--output-folder'] else '.'
    num_neighbours = int(args['--num-neighbours']) if args['--num-neighbours'] else 10

    device = "cuda" if args['--cuda'] else "cpu"

    logging.info('save_context_context_hub_N_k_info_for_all')

    context_to_context_hubs.save_context_context_hub_N_k_info_for_all(
        result_folder=output_folder,
        num_neighbours=num_neighbours,
        device=device)

    context_to_context_hubs.save_context_context_hub_N_k_info_for_pythia_checkpoints(
        result_folder=output_folder,
        num_neighbours=num_neighbours,
        device=device)


def distance_distribution_histograms(args: Dict):
    """ Make and save distance distribution histograms """
    output_folder = args['--output-folder'] if args['--output-folder'] else '.'
    vocab_main_folder = args['--vocab-main-folder'] if args['--vocab-main-folder'] else ''

    skip_calcs_str = args['--skip-calcs'] if args['--skip-calcs'] else ''
    skip_calcs = skip_calcs_str.split('_')

    device = "cuda" if args['--cuda'] else "cpu"

    logging.info('make_histograms_of_distances')

    distribution_of_distances.make_histograms_of_distances(
        result_folder=output_folder,
        vocab_main_folder=vocab_main_folder,
        device=device,
        skip_calcs= skip_calcs)
    
    distribution_of_distances.make_histograms_of_distances_pythia_checkpoints(
        result_folder=output_folder,
        vocab_main_folder=vocab_main_folder,
        device=device,
        skip_calcs= skip_calcs)


def plot_k_occurrences(args: Dict):
    """ Make and save k-occurrence plots """

    output_folder = args['--output-folder'] if args['--output-folder'] else '.'
    vocab_main_folder = args['--vocab-main-folder'] if args['--vocab-main-folder'] else ''

    device = "cuda" if args['--cuda'] else "cpu"

    logging.info('make_line_plots_of_k_occurrence')

    plot_k_occurrence_distributions.make_line_plots_of_k_occurrence(
        result_folder=output_folder, vocab_main_folder=vocab_main_folder, device=device
    )

    plot_k_occurrence_distributions.make_line_plots_of_k_occurrence_pythia_checkpoints(
        result_folder=output_folder, vocab_main_folder=vocab_main_folder, device=device
    )


def get_hub_examples(args: Dict):
    """ Get and save hub examples """

    output_folder = args['--output-folder'] if args['--output-folder'] else '.'
    vocab_main_folder = args['--vocab-main-folder'] if args['--vocab-main-folder'] else ''
    copora_folder = args['--copora-folder'] if args['--copora-folder'] else '.'

    num_top_hubs = int(args['--num-top-hubs']) if args['--num-top-hubs'] else 20

    logging.info('get_examples_of_top_hubs')
    
    hub_examples.get_examples_of_top_hubs(
        result_folder=output_folder, vocab_main_folder=vocab_main_folder, 
        copora_folder=copora_folder,
        num_top_hubs=num_top_hubs)


def main():
    """ Main function """      
    args = docopt(__doc__)
    
    log_level = args['--log'] if args['--log'] else ''
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=numeric_level)   
    
    
    if args['save-next-token-probs']:
        save_next_token_probabilities(args)
    elif args['pred-hub-Nk-vs-freq-plot']:
        pred_hub_N_k_vs_token_freqs(args)
    elif args['tt-hub-Nk-vs-freq-plot']:
        token_token_hub_N_k_vs_token_freqs(args)
    elif args['cc-hub-Nk-info']:
        context_context_hub_N_k_info(args)
    elif args['distance-distributions']:
        distance_distribution_histograms(args)
    elif args['plot-k-occurrence']:
        plot_k_occurrences(args)
    elif args['get-hub-examples']:
        get_hub_examples(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
