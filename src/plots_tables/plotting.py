"""

Functions for making plots


"""


import logging

from typing import List

import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from src.plots_tables import util



def make_log_vs_log_scatterplot_w_spearman(
    N_ks: NDArray, hub_token_frequencies: NDArray, 
    plot_title: str, plot_file_name: str, corr_placement: str) -> None:
    """ Make and save hub rank vs token frequency plots """    
    label_fontsize = 20
    eps = 1e-9    

    if ((hub_token_frequencies > 0).sum() + (N_ks > 0).sum()) == 0:
        logging.warning(
            'No positive data for %s. freqs: %s, N_ks: %s',
            plot_file_name,
            (hub_token_frequencies > 0).sum(),
            (N_ks > 0).sum())
        return
    
    # We add small value to frequencies to be able to do a log scale 
    # and still see the zero frequency tokens on the plot
    hub_token_frequencies = np.array(hub_token_frequencies) + eps

    var_token_frequencies = np.var(hub_token_frequencies)
    if var_token_frequencies == 0:
        spearman_rank_corr_stat = 0.0
    else:
        spearman_rank_corr = stats.spearmanr(N_ks, hub_token_frequencies)
        spearman_rank_corr_stat = spearman_rank_corr.statistic

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    fig.dpi = 300
    ax.scatter(N_ks, hub_token_frequencies, s = 3)
    ax.set_xlabel('k-occurrence', fontsize=label_fontsize)
    ax.set_ylabel('frequency', fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize-2)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(plot_title, fontsize = label_fontsize)
    if corr_placement == 'lower_right':
        fig_text_x = 0.85
        fig_text_y = 0.1
    elif corr_placement == 'upper_right':
        fig_text_x = 0.85
        fig_text_y = 0.9
    else:
        fig_text_x = 0.5
        fig_text_y = 0.025

    ax.text(
        fig_text_x, fig_text_y,
        r'$\rho =$' + f'{spearman_rank_corr_stat:.2f}', 
        transform = plt.gca().transAxes,
        ha="center", fontsize=label_fontsize-2, bbox={"alpha":0.5, "pad":3})
    fig.tight_layout()
    util.save_plot(plot_file_name)


def make_three_steps_log_vs_log_scatterplot_w_spearman(
    N_ks: NDArray, hub_token_frequencies: NDArray, 
    model_names: List[str], plot_file_name: str, corr_placement: str) -> None:
    """ Make and save hub rank vs token frequency plots """    
    label_fontsize = 20
    eps = 1e-9

    corrs = []
    for i, current_freq in enumerate(hub_token_frequencies):
        if ((current_freq > 0).sum() + (N_ks[i] > 0).sum()) == 0:
            logging.warning(
                'No positive data for %s. freqs: %s, N_ks: %s',
                plot_file_name,
                (current_freq > 0).sum(),
                (N_ks[i] > 0).sum())
            return
        
        var_token_frequencies = np.var(current_freq)
        if var_token_frequencies == 0:
            spearman_rank_corr_stat = 0.0
        else:
            spearman_rank_corr = stats.spearmanr(N_ks[i], current_freq)
            spearman_rank_corr_stat = spearman_rank_corr.statistic
        
        corrs.append(spearman_rank_corr_stat)
    
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 5))
    fig.dpi = 300

    for i, _ in enumerate(axs):
        # We add small value to frequencies to be able to do a log scale 
        # and still see the zero frequency tokens on the plot
        axs[i].scatter(N_ks[i], hub_token_frequencies[i] + eps, s = 3)
        
        if i > 0:
            axs[i].sharey(axs[0])
        
        if i == 1:
            axs[i].set_xlabel('k-occurrence', fontsize=label_fontsize)
        
        if i == 0:
            axs[i].set_ylabel('frequency', fontsize=label_fontsize)
        
        axs[i].tick_params(axis='both', which='major', labelsize=label_fontsize-2)
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')

        if i == 1:
            current_title = f'Pythia training step \n {model_names[i]}'
        else:
            current_title = f' \n {model_names[i]}'
        
        axs[i].set_title(current_title, fontsize = label_fontsize)
        if corr_placement == 'lower_right':
            fig_text_x = 0.68
            fig_text_y = 0.1
        elif corr_placement == 'upper_right':
            fig_text_x = 0.68
            fig_text_y = 0.9
        else:
            fig_text_x = 0.5
            fig_text_y = 0.025

        axs[i].text(
            fig_text_x, fig_text_y,
            r'$\rho =$' + f'{corrs[i]:.2f}', 
            transform = axs[i].transAxes,
            ha="center", fontsize=label_fontsize-2, bbox={"alpha":0.5, "pad":3})
    
    for ax in fig.get_axes():
        ax.label_outer()

    fig.tight_layout()

    util.save_plot(plot_file_name)


def make_distance_hist_plot(
    distances: NDArray, max_dist:float, 
    plot_title: str, plot_file_name:str,
    min_dist: float = 0, bins: int = 100) -> None:
    """ Make a histogram of distances """
    fontsize = 20
    
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.dpi = 300
    ax.hist(distances.flatten(), bins=bins, range=(min_dist, max_dist))
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.set_yscale('log')
    ax.set_ylim(bottom = 1.0)
    ax.set_title(plot_title, fontsize=fontsize)

    ax.set_xlabel('distance', fontsize=fontsize)
    ax.set_ylabel('count', fontsize=fontsize)
    fig.tight_layout()

    util.save_plot(plot_file_name)


def make_two_in_one_distance_hist_plot(
    distances: NDArray, max_dist:float, 
    plot_titles: str, plot_file_name:str,
    y_max: float,
    min_dist: float = 0, bins: int = 100) -> None:
    """ Make a histogram of distances """
    fontsize = 20
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 5))
    fig.dpi = 300

    for i, _ in enumerate(axs):
        axs[i].hist(distances[i].flatten(), bins=bins, range=(min_dist, max_dist))

        if i > 0:
            axs[i].sharey(axs[0])

        axs[i].set_yscale('log')
        if y_max > 0:
            axs[i].set_ylim(bottom = 1.0, top = y_max)
        else:
            axs[i].set_ylim(bottom = 1.0)

        if i == 0:
            axs[i].set_ylabel('count', fontsize=fontsize)
        axs[i].set_xlabel('distances', fontsize=fontsize)
        
        axs[i].set_title(plot_titles[i], fontsize = fontsize)
        axs[i].tick_params(axis='both', which='major', labelsize=fontsize-2)

    for ax in fig.get_axes():
        ax.label_outer()

    fig.tight_layout()

    util.save_plot(plot_file_name)
    

def plot_k_occurrence_hist_from_N_k(
    N_k_result: NDArray, k_skew: float, 
    add_k_skew: bool,
    log_scale: bool,
    plot_title: str, save_to: str = '') -> None:
    """ Plot the k-occurrence histogram """
    fontsize = 12
    num_bins = max(N_k_result) + 1

    plt.hist(N_k_result, bins=num_bins, range=(0, num_bins))
    if log_scale:
        plt.yscale('log')
        
    plt.title(plot_title)
    if add_k_skew:
        plt.figtext(
            0.5, 0.025, f'k-skew: {k_skew:.2f}', 
            ha="center", fontsize=fontsize, bbox={"alpha":0.5, "pad":3})

    if save_to != '':
        util.save_plot(save_to)
    else:
        plt.show()


def plot_k_occurrence_line_from_N_k(
    N_k_result: NDArray, k_skew: float, 
    add_k_skew: bool,
    log_scale: bool,
    plot_title: str, save_to: str = '') -> None:
    """ Plot the k-occurrence line plot """
    fontsize = 20
    unique, counts = np.unique(N_k_result, return_counts=True)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.dpi = 300
    ax.plot(unique, counts, marker = 'o', markersize= 3)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.set_title(plot_title, fontsize = fontsize)
    ax.set_xlabel('k-occurrence', fontsize=fontsize)
    ax.set_ylabel('number of points', fontsize=fontsize)

    if add_k_skew:
        fig_text_x = 0.8
        fig_text_y = 0.9
        ax.text(
        fig_text_x, fig_text_y,
        f'k-skew: {k_skew:.2f}', 
        transform = plt.gca().transAxes,
        ha="center", fontsize=fontsize-2, bbox={"alpha":0.5, "pad":3})
        
    if log_scale:
        ax.set_yscale('log')
        ax.set_xscale('log')
    
    fig.tight_layout()

    if save_to != '':
        util.save_plot(save_to)
    else:
        fig.show()


def make_four_in_one_plot(
    file_name: str, all_N_ks, all_hub_token_frequencies, corrs) -> None:
    """ Make four in one plot """
    fontsize = 18
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 5))
    fig.dpi = 300
    axs[0, 0].scatter(all_N_ks[0], all_hub_token_frequencies[0], s = 3)
    axs[0, 0].set_title('Freq Pile10k', fontsize = fontsize)
    axs[0, 0].set_ylabel('Preds Pile10k', fontsize = fontsize)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=fontsize-2)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xscale('log')

    axs[0, 1].sharex(axs[0, 0])
    axs[0, 1].sharey(axs[0, 0])
    axs[0, 1].tick_params(axis='both', which='major', labelsize=fontsize-2)
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_xscale('log')
    axs[0, 1].scatter(all_N_ks[1], all_hub_token_frequencies[1], s = 3)
    axs[0, 1].set_title('Freq Bookcorpus', fontsize = fontsize)

    axs[1, 0].sharex(axs[0, 0])
    axs[1, 0].sharey(axs[0, 0])
    axs[1, 0].tick_params(axis='both', which='major', labelsize=fontsize-2)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xscale('log')
    axs[1, 0].scatter(all_N_ks[2], all_hub_token_frequencies[2], s = 3)
    axs[1, 0].set_ylabel('Preds Bookcorpus', fontsize = fontsize)

    axs[1, 1].sharex(axs[0, 0])
    axs[1, 1].sharey(axs[0, 0])
    axs[1, 1].tick_params(axis='both', which='major', labelsize=fontsize-2)
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xscale('log')
    axs[1, 1].scatter(all_N_ks[3], all_hub_token_frequencies[3], s = 3)

    for ax in fig.get_axes():
        ax.label_outer()

    
    # Add correlation text
    axs[0, 0].text(0.95, 0.1, r'$\rho =$' + corrs[0], 
    fontsize = fontsize, ha="right", va="bottom", bbox={"alpha":0.5, "pad":3}, 
    transform = axs[0, 0].transAxes)

    axs[0, 1].text(0.95, 0.1, r'$\rho =$' + corrs[1], 
    fontsize = fontsize, ha="right", va="bottom", bbox={"alpha":0.5, "pad":3}, 
    transform = axs[0, 1].transAxes)

    axs[1, 0].text(0.95, 0.1, r'$\rho =$' + corrs[2], 
    fontsize = fontsize, ha="right", va="bottom", bbox={"alpha":0.5, "pad":3}, 
    transform = axs[1, 0].transAxes)

    axs[1, 1].text(0.95, 0.1, r'$\rho =$' + corrs[3], 
    fontsize = fontsize, ha="right", va="bottom", bbox={"alpha":0.5, "pad":3}, 
    transform = axs[1, 1].transAxes)

    fig.tight_layout()

    if file_name != '':
        util.save_plot(file_name)
    else:    
        fig.show()
