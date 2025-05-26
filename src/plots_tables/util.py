"""

Utility functions for making plots and tables

"""


import os

import matplotlib.pyplot as plt


def get_plot_folder() -> str:
    """ Get the folder for plots """
    return 'plots'


def save_plot(plot_file_name: str) -> None:
    """ Save the current plot """
    plot_folder = get_plot_folder()
    plot_path = os.path.join(plot_folder, plot_file_name)
    plt.savefig(plot_path, dpi = 300)
    plt.clf()
