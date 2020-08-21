import sys

import matplotlib

matplotlib.use("Agg")

size = 20

matplotlib.rc("font", size=size)  # controls default text sizes
matplotlib.rc("axes", titlesize=size)  # fontsize of the axes title
matplotlib.rc("axes", labelsize=size)  # fontsize of the x and y labels
matplotlib.rc("xtick", labelsize=size)  # fontsize of the tick labels
matplotlib.rc("ytick", labelsize=size)  # fontsize of the tick labels
matplotlib.rc("legend", fontsize=size)  # legend fontsize
matplotlib.rc("figure", titlesize=size)  # fontsize of the figure title
matplotlib.rc("figure", autolayout=True)

from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

from CADETMatch.cache import cache

from pathlib import Path
import pandas
import numpy
import scipy.interpolate
import itertools

from cadet import Cadet, H5
from addict import Dict

# parallelization
import multiprocessing

import os
import warnings
import CADETMatch.util as util
import logging
import CADETMatch.loggerwriter as loggerwriter

from matplotlib.colors import ListedColormap
import matplotlib.cm

cm_plot = matplotlib.cm.gist_rainbow

def get_color(idx, max_colors, cmap):
    return cmap(1.0 * float(idx) / max_colors)

def main(map_function):
    cache.setup_dir(sys.argv[1])
    util.setupLog(cache.settings["resultsDirLog"], "mixing.log")
    cache.setup(sys.argv[1])

    multiprocessing.get_logger().info("mixing graphing directory %s", os.getcwd())

    mcmcDir = Path(cache.settings["resultsDirMCMC"])
    mcmc_h5 = mcmcDir / "mcmc.h5"
    if mcmc_h5.exists():
        mcmc_store = H5()
        mcmc_store.filename = mcmc_h5.as_posix()
        mcmc_store.load(paths=["/full_chain", "/train_full_chain", "/bounds_full_chain",
                               "/full_chain_transform", "/train_full_chain_transform", "/bounds_full_chain_transform"])

        progress_path = Path(cache.settings["resultsDirBase"]) / "result.h5"

        graph_dir = cache.settings["resultsDirSpace"] / "mcmc"

        mixing = graph_dir / "mixing"
        mixing.mkdir(parents=True, exist_ok=True)

        input_headers = cache.parameter_headers_actual

        chain_names = ("full_chain", "train_full_chain", "bounds_full_chain", 
                       "full_chain_transform", "train_full_chain_transform", "bounds_full_chain_transform")

        for chain in chain_names:
            if chain in mcmc_store.root:
                plot_chain(input_headers, mcmc_store.root[chain], chain, mixing / ("mixing_%s" % chain))


def plot_chain(headers, chain, chain_name, graph_dir):
    graph_dir.mkdir(parents=True, exist_ok=True)

    for i in range(chain.shape[2]):

        fig = figure.Figure(figsize=[15, 7])
        canvas = FigureCanvas(fig)
        graph = fig.add_subplot(1, 1, 1)

        chain_length = chain.shape[1]
        x = numpy.linspace(0, chain_length -1, chain_length)

        lines = []
        for j in range(chain.shape[0]):
            graph.plot(x, chain[j, :, i], color = get_color(j, chain.shape[0] - 1, cm_plot))

        graph.set_xlabel("chain length")
        graph.set_ylabel("value")
        filename = "mixing_%s.png" % (headers[i])
        filename = filename.replace(":", "_").replace("/", "_")

        graph.set_title("Mixing graph %s" % headers[i])
        fig.savefig((graph_dir / filename).as_posix())


if __name__ == "__main__":
    map_function = util.getMapFunction()
    main(map_function)
    sys.exit()
