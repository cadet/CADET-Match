import sys

import matplotlib
import matplotlib.style as mplstyle

mplstyle.use("fast")

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

import itertools
import logging

# parallelization
import multiprocessing
import os
import warnings
from pathlib import Path

import numpy
import scipy.interpolate
from addict import Dict
from cadet import H5, Cadet
from emcee import autocorr
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

import CADETMatch.loggerwriter as loggerwriter
import CADETMatch.util as util
from CADETMatch.cache import cache


def main(map_function):
    cache.setup_dir(sys.argv[1])
    util.setupLog(cache.settings["resultsDirLog"], "autocorr.log")
    cache.setup(sys.argv[1])

    multiprocessing.get_logger().info("autocorr graphing directory %s", os.getcwd())

    mcmcDir = Path(cache.settings["resultsDirMCMC"])
    mcmc_h5 = mcmcDir / "mcmc.h5"
    if mcmc_h5.exists():
        resultDir = Path(cache.settings["resultsDirBase"])
        result_lock = resultDir / "result.lock"

        mcmc_store = H5()
        mcmc_store.filename = mcmc_h5.as_posix()
        mcmc_store.load(
            paths=["/full_chain", "/train_full_chain", "/bounds_full_chain"],
            lock=True
        )

        progress_path = Path(cache.settings["resultsDirBase"]) / "result.h5"

        graph_dir = cache.settings["resultsDirSpace"] / "mcmc"

        input_headers = cache.parameter_headers_actual

        for chain in ("full_chain", "train_full_chain", "bounds_full_chain"):
            if chain in mcmc_store.root:
                plot_chain(
                    input_headers, mcmc_store.root[chain], chain, graph_dir / chain
                )


def plot_chain(headers, chain, chain_name, graph_dir):
    graph_dir.mkdir(parents=True, exist_ok=True)

    for i in range(chain.shape[2]):

        fig = figure.Figure(figsize=[15, 7])
        canvas = FigureCanvas(fig)
        graph = fig.add_subplot(1, 1, 1)

        lines = []
        for j in range(chain.shape[0]):
            lines.append(autocorr.function_1d(chain[j, :, i]))
            graph.plot(lines[-1])
        graph.plot(numpy.mean(numpy.array(lines), axis=0), "k", linewidth=4)

        graph.set_xlabel("time")
        graph.set_ylabel("correlation")
        filename = "correlation_%s.png" % (headers[i])
        filename = filename.replace(":", "_").replace("/", "_")

        sum_lines = numpy.sum(numpy.array(lines), axis=0)
        sum_lines = sum_lines / chain.shape[0]
        taus = 2.0 * numpy.cumsum(sum_lines) - 1.0
        tau = taus[autocorr.auto_window(taus, 5)]

        graph.set_title("Correlation graph %s  Tau: %.2f" % (headers[i], tau))
        fig.savefig((graph_dir / filename).as_posix())

        fig = figure.Figure(figsize=[15, 7])
        canvas = FigureCanvas(fig)
        graph = fig.add_subplot(1, 1, 1)
        graph.set_title("Tau graph %s  Tau: %.2f" % (headers[i], tau))
        graph.set_xlabel("time")
        graph.set_ylabel("tau")
        graph.plot(taus, "k", linewidth=4)
        filename = "tau_%s.png" % (headers[i])
        filename = filename.replace(":", "_").replace("/", "_")
        fig.savefig((graph_dir / filename).as_posix())


if __name__ == "__main__":
    map_function = util.getMapFunction()
    main(map_function)
    sys.exit()
