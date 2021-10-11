import multiprocessing
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.style as mplstyle
import numpy
import pandas
import scipy
from addict import Dict
from cadet import H5, Cadet
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity

import CADETMatch.evo as evo
import CADETMatch.kde_util as kde_util
import CADETMatch.smoothing as smoothing
import CADETMatch.util as util
from CADETMatch.cache import cache

mplstyle.use("fast")

matplotlib.use("Agg")

import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

cm_plot = matplotlib.cm.gist_rainbow

import itertools
import logging

import CADETMatch.loggerwriter as loggerwriter

import arviz as az

def get_color(idx, max_colors, cmap):
    return cmap(1.0 * float(idx) / max_colors)


saltIsotherms = {
    b"STERIC_MASS_ACTION",
    b"SELF_ASSOCIATION",
    b"MULTISTATE_STERIC_MASS_ACTION",
    b"SIMPLE_MULTISTATE_STERIC_MASS_ACTION",
    b"BI_STERIC_MASS_ACTION",
}

size = 20

plt.rc("font", size=size)  # controls default text sizes
plt.rc("axes", titlesize=size)  # fontsize of the axes title
plt.rc("axes", labelsize=size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=size)  # fontsize of the tick labels
plt.rc("ytick", labelsize=size)  # fontsize of the tick labels
plt.rc("legend", fontsize=size)  # legend fontsize
plt.rc("figure", titlesize=size)  # fontsize of the figure title
plt.rc("figure", autolayout=True)

class ArvizSampler:
    def __init__(self, chain, prob):
        self.chain = chain.swapaxes(0, 1)
        self.prob = prob.swapaxes(0, 1)
    
    def get_chain(self):
        return self.chain

    def get_log_prob(self):
        return self.prob

def flatten(chain):
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])
    return flat_chain

def reduce_data(data_chain, probability, size):
    sampler = ArvizSampler(data_chain, probability)
    emcee_data = az.from_emcee(sampler)
    hdi = az.hdi(emcee_data, hdi_prob=0.99).to_array().values

    multiprocessing.get_logger().info("hdi %s", hdi)

    data= flatten(data_chain)

    selected = (data > hdi[:,0]) & (data < hdi[:,1])
    selected = numpy.all(selected, axis=1)

    data = data[selected]

    scaler = preprocessing.RobustScaler().fit(data)

    data = scaler.transform(data)

    if size < data.shape[0]:
        indexes = numpy.random.choice(data.shape[0], size, replace=False)
        data_reduced = data[indexes]
    else:
        data_reduced = data

    return data_reduced, scaler


def get_prior(data_chain, probability, headers):
    multiprocessing.get_logger().info("setting up scaler and reducing data")
    data_reduced, scaler = reduce_data(data_chain, probability, 30000)

    multiprocessing.get_logger().info("finished setting up scaler and reducing data")
    multiprocessing.get_logger().info("data_reduced shape %s", data_reduced.shape)

    kde_ga = KernelDensity(kernel="gaussian")

    kde_ga, bandwidth, store = kde_util.get_bandwidth(kde_ga, data_reduced)

    mcmcDir = Path(cache.settings["resultsDirMCMC"])

    plot_bandwidth(store, mcmcDir)

    multiprocessing.get_logger().info("mle bandwidth: %.2g", bandwidth)

    multiprocessing.get_logger().info("fitting kde with mle bandwidth")

    kde_ga.fit(data_reduced)

    multiprocessing.get_logger().info("finished fitting")

    return kde_ga, scaler

def plot_bandwidth(store, mcmcDir):
    plt.figure(figsize=[10, 10])
    plt.scatter(store[:, 0], store[:, 1])
    plt.xlabel("bandwidth")
    plt.ylabel("cross_val_score")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(str(mcmcDir / "prior_log_bandwidth.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=[10, 10])
    plt.scatter(store[:, 0], 1 - store[:, 1])
    plt.xlabel("bandwidth")
    plt.ylabel("1 - cross_val_score")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(str(mcmcDir / "prior_1-log_bandwidth.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=[10, 10])
    plt.scatter(store[:, 0], store[:, 1])
    plt.xlabel("bandwidth")
    plt.ylabel("cross_val_score")
    plt.savefig(str(mcmcDir / "prior_bandwidth.png"), bbox_inches="tight")
    plt.close()


def process_mle(chain, probability, cache):
    kde, scaler = get_prior(chain, probability, cache.parameter_headers_actual)

    mcmcDir = Path(cache.settings["resultsDirMCMC"])
    joblib.dump(scaler, mcmcDir / "kde_prior_scaler.joblib")
    joblib.dump(kde, mcmcDir / "kde_prior.joblib")


def main():
    cache.setup_dir(sys.argv[1])
    util.setupLog(cache.settings["resultsDirLog"], "prior.log")
    cache.setup(sys.argv[1])

    mcmcDir = Path(cache.settings["resultsDirMCMC"])
    mcmc_h5 = mcmcDir / "mcmc.h5"

    resultDir = Path(cache.settings["resultsDirBase"])
    result_lock = resultDir / "result.lock"

    mcmc_store = H5()
    mcmc_store.filename = mcmc_h5.as_posix()
    mcmc_store.load(paths=["/full_chain", "/mcmc_acceptance", "/run_probability"], lock=True)

    process_mle(mcmc_store.root.full_chain, 
                mcmc_store.root.run_probability,
                cache)


if __name__ == "__main__":
    main()

