"""

Making plots on synthetic data for illustrative purposes


"""



from typing import Tuple

import matplotlib.pyplot as plt

import numpy as np 
from numpy.typing import NDArray

from sklearn.neighbors import NearestNeighbors

from scipy.stats import skew

import torch

from tqdm import tqdm



random_seed = 0
rng = np.random.default_rng(random_seed)

high_dimension = 300
low_dimension = 3
num_neighbours = 10


# high dimensional data
num_query_points = 10000 
high_dim_queries = rng.standard_normal((num_query_points, high_dimension))

num_other_points = 10000 
high_dim_other_points = rng.standard_normal((num_other_points, high_dimension))

# Low dimensional data to compare 
low_dim_queries = rng.standard_normal((num_query_points, low_dimension))
low_dim_other_points = rng.standard_normal((num_other_points, low_dimension))


# Consider Euclidean distances
def get_euclidean_dist(queries, points):
    """ Get the euclidean distances between query points and others """
    distances = np.linalg.norm(queries - points, axis = 1)
    
    return distances

high_dim_euc_dist = get_euclidean_dist(high_dim_queries, high_dim_other_points)
high_dim_max_dist = np.max(high_dim_euc_dist)

high_dim_hist_title = f'Hist of Euclidean distances in {high_dimension} dimensions'
high_dim_hist_path = 'high_dim_euc_dist_hist.png'


low_dim_euc_dist = get_euclidean_dist(low_dim_queries, low_dim_other_points)
low_dim_max_dist = np.max(low_dim_euc_dist)

low_dim_hist_title = f'Hist of Euclidean distances in {low_dimension} dimensions'
low_dim_hist_path = 'low_dim_euc_dist_hist.png'

# For comparison in the same plot
def plot_comparison(hd_dist, hd_max_dist, ld_dist):
    fontsize = 18
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.dpi = 300
    ax.hist(hd_dist.flatten(), bins=200, range=(0, hd_max_dist + 0.1), label='300')
    ax.hist(ld_dist.flatten(), bins=200, range=(0, hd_max_dist + 0.1), color='green', label = '3')
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.set_yscale('log')
    ax.set_xlabel('distances', fontsize = fontsize)
    ax.set_ylabel('count', fontsize = fontsize)
    ax.legend(fontsize = fontsize, loc = 'lower center')
    fig.tight_layout()

    fig.savefig('high_low_dim_euc_dist_hist.png', dpi = 300)
    plt.clf()

plot_comparison(high_dim_euc_dist, high_dim_max_dist, low_dim_euc_dist)


def plot_distance_distribution_hist(
        distances, max_distance, 
        plot_title: str, plot_path: str):
    fontsize = 17
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(distances.flatten(), bins=200, range=(0, max_distance + 0.1))
    ax.set_yscale('log')
    ax.set_xlabel('distances', fontsize = fontsize)
    ax.set_ylabel('count', fontsize = fontsize)
    ax.set_title(plot_title, fontsize = fontsize)
    fig.tight_layout()
    
    fig.savefig(plot_path, dpi = 300)
    plt.clf()


# Plot distance distributions high and low dimension
plot_distance_distribution_hist(
    high_dim_euc_dist, high_dim_max_dist, high_dim_hist_title, high_dim_hist_path)

plot_distance_distribution_hist(
    low_dim_euc_dist, high_dim_max_dist, low_dim_hist_title, low_dim_hist_path)


# Compare with when using softmaxed dotproduct distance
def do_softmax_dots(queries: torch.Tensor, points: torch.Tensor):
    """ Calculate softmax dots  """
    # Do dot product in batches 
    softmax_dots = torch.zeros((queries.shape[0], points.shape[0]))
    batch_size = torch.tensor(128)
    batch_nums = int(torch.ceil(queries.shape[0]/batch_size))
    torch_softmax = torch.nn.Softmax(dim = 1)

    for idx in tqdm(torch.arange(batch_nums)):
        current_batch = queries[idx*batch_size : (idx + 1)*batch_size]
        current_dots = torch.matmul(current_batch, points.T).squeeze()
        softmax_dots[idx*batch_size : (idx + 1)*batch_size] = torch_softmax(current_dots)
    
    return softmax_dots


high_dim_sofmax_dots = do_softmax_dots(
    torch.from_numpy(high_dim_queries), torch.from_numpy(high_dim_other_points))
high_dim_sofmax_dots = high_dim_sofmax_dots.detach().to('cpu').numpy()
high_dim_softmax_dist = 1 - high_dim_sofmax_dots

high_dim_sd_hist_title = f'Softmax dot distances in {high_dimension} dimensions'
high_dim_sd_hist_path = 'high_dim_softmax_dot_dist_hist.png'


low_dim_sofmax_dots = do_softmax_dots(
    torch.from_numpy(low_dim_queries), torch.from_numpy(low_dim_other_points))
low_dim_softmax_dist = 1 - low_dim_sofmax_dots

low_dim_sd_hist_title = f'Softmax dot distances in {low_dimension} dimensions'
low_dim_sd_hist_path = 'low_dim_softmax_dot_dist_hist.png'


plot_distance_distribution_hist(
    high_dim_softmax_dist, 1, high_dim_sd_hist_title, high_dim_sd_hist_path)

plot_distance_distribution_hist(
    low_dim_softmax_dist, 1, low_dim_sd_hist_title, low_dim_sd_hist_path)





# Consider k-occurrences
def N_k_x(
        x_index: int, 
        nearest_k_for_every_point: NDArray) -> float:
    """ Get the k-occurrence of point with index [x_index] """
    # If x is in the k nearest neighbours for a point it only occurs once
    # So we can sum all the occurences of x in the array of nearest neighbours. 
    among_k_nearest = (nearest_k_for_every_point == x_index).sum()
    
    return among_k_nearest


def get_k_skew(N_ks: NDArray) -> float:
    """ Get the skewness of the k-occurrence distribution """
    return skew(N_ks)


def N_k_for_fitted(
        k: int, 
        vectors: NDArray, 
        fitted_n_neighbour: NearestNeighbors) -> NDArray:
    """ Get the k-occurence of [vectors] using a fitted NearestNeighbors model """
    
    nearest_k = fitted_n_neighbour.kneighbors(
        vectors, 
        n_neighbors = k, 
        return_distance = False)
    
    num_vectors = vectors.shape[0]
    
    N_k_result = np.zeros(num_vectors).astype(int)
    for i in tqdm(range(num_vectors)):
        N_k_result[i] = N_k_x(
            x_index = i, 
            nearest_k_for_every_point = nearest_k)
    
    return N_k_result


def N_k(
        k: int, 
        vectors: NDArray) -> Tuple[NDArray, float]:
    """ Get the k-occurence and k-skew of [vectors]"""
    
    nn_vectors = np.copy(vectors)
    knn_metric = 'euclidean'
    
    nearest_neighbours = NearestNeighbors(n_neighbors = k, metric = knn_metric)
    nearest_neighbours.fit(nn_vectors)
    
    N_k_result = N_k_for_fitted(k, nn_vectors, nearest_neighbours)

    k_skew = get_k_skew(N_k_result)
    
    return N_k_result, k_skew


low_dim_N_k, low_dim_k_skew = N_k(num_neighbours, low_dim_queries)

high_dim_N_k, high_dim_k_skew = N_k(num_neighbours, high_dim_queries)


def plot_k_occurrence_line_from_N_k(
    N_k_result: NDArray, k_skew: float, 
    add_k_skew: bool,
    log_scale: bool,
    plot_title: str, save_to: str = '') -> None:
    """ Plot the k-occurrence line plot """
    fontsize = 18
    unique, counts = np.unique(N_k_result, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(unique, counts, marker = 'o', markersize= 3)
    ax.set_title(plot_title, fontsize = fontsize)
    ax.set_xlabel('k-occurrence', fontsize=fontsize)
    ax.set_ylabel('number of points', fontsize=fontsize)

    if add_k_skew:
        fig_text_x = 0.83
        fig_text_y = 0.9
        plt.figtext(
        fig_text_x, fig_text_y,
        f'k-skew: {k_skew:.2f}', 
        transform = plt.gca().transAxes,
        ha="center", fontsize=fontsize-2, bbox={"alpha":0.5, "pad":3})
        
    if log_scale:
        ax.set_yscale('log')
        ax.set_xscale('log')

    fig.tight_layout()
    if save_to != '':        
        fig.savefig(save_to, dpi = 300)
        plt.clf()
    else:
        plt.show()


low_dim_N_k_plot_title=f'K-occurrence in {low_dimension} dimensions'
low_dim_N_k_plot_path = 'low_dim_euc_dist_k_occurrence.png'

plot_k_occurrence_line_from_N_k(
    low_dim_N_k, low_dim_k_skew, 
    add_k_skew=True, log_scale=True, 
    plot_title=low_dim_N_k_plot_title, save_to=low_dim_N_k_plot_path)


high_dim_N_k_plot_title=f'K-occurrence in {high_dimension} dimensions'
high_dim_N_k_plot_path = 'high_dim_euc_dist_k_occurrence.png'

plot_k_occurrence_line_from_N_k(
    high_dim_N_k, high_dim_k_skew, 
    add_k_skew=True, log_scale=True, 
    plot_title=high_dim_N_k_plot_title, save_to=high_dim_N_k_plot_path)



def plot_hist_and_k_occurrence_comparison(
        ld_dist, hd_dist, hd_max_dist, 
        hd_N_k_result: NDArray, ld_N_k_result: NDArray, 
        k_skews: Tuple[float, float], 
        save_to: str = ''
    ):
    """ Plot a comparison of histograms and k-occurrences """
    fontsize = 18
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    fig.dpi = 300

    axes[0].hist(ld_dist.flatten(), bins=200, range=(0, hd_max_dist + 0.1), color='green', label = '3')
    axes[0].hist(hd_dist.flatten(), bins=200, range=(0, hd_max_dist + 0.1), label='300')
    
    axes[0].set_yscale('log')
    axes[0].set_xlabel('distances', fontsize = fontsize)
    axes[0].set_ylabel('count', fontsize = fontsize)
    axes[0].tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes[0].legend(fontsize = fontsize, loc = 'lower center')

    hd_unique, hd_counts = np.unique(hd_N_k_result, return_counts=True)
    ld_unique, ld_counts = np.unique(ld_N_k_result, return_counts=True)

    axes[1].plot(ld_unique, ld_counts, marker = 'o', markersize= 4, linewidth=2.0,
                 color='green', label=f'k-skew: {k_skews[1]:.2f}')
    axes[1].plot(hd_unique, hd_counts, marker = 'o', markersize= 4, linewidth=2.0, 
                 label=f'k-skew: {k_skews[0]:.2f}')
    
    axes[1].set_xlabel('k-occurrence', fontsize=fontsize)
    axes[1].set_ylabel('number of points', fontsize=fontsize)
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes[1].legend(fontsize = fontsize, loc = 'upper right')

    fig.tight_layout()
    #fig.show()

    fig.savefig(save_to, dpi = 300)
    plt.clf()
    

plot_hist_and_k_occurrence_comparison(
    ld_dist = low_dim_euc_dist,
    hd_dist = high_dim_euc_dist, hd_max_dist = high_dim_max_dist,     
    hd_N_k_result = high_dim_N_k,
    ld_N_k_result = low_dim_N_k,
    k_skews = (high_dim_k_skew, low_dim_k_skew),
    save_to = 'high_low_dim_dist_hist_k_occurrence_comparison.png'
    )
    
        
    
    


    

