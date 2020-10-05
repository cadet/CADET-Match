import numpy
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import scipy
import multiprocessing
import sys

import cadet
import CADETMatch.util as util
import CADETMatch.evo as evo
import pandas
from addict import Dict
from CADETMatch.cache import cache
import CADETMatch.smoothing as smoothing
import CADETMatch.kde_util as kde_util
from pathlib import Path
import warnings
import joblib

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import matplotlib
import matplotlib.style as mplstyle
mplstyle.use('fast')

matplotlib.use("Agg")

from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import matplotlib.pyplot as plt

import matplotlib.cm

cm_plot = matplotlib.cm.gist_rainbow

import logging
import CADETMatch.loggerwriter as loggerwriter

import itertools


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

atol = 1e-3
rtol = 1e-3


def reduce_data(data, size, bw_size):
    # size reduces data for normal usage, bw_size reduces the size just for bandwidth estimation
    lb, ub = numpy.percentile(data, [5, 95], 0)
    selected = (data >= lb) & (data <= ub)

    selected = numpy.all(selected, 1)

    data = data[selected, :]

    scaler = preprocessing.RobustScaler().fit(data)

    data = scaler.transform(data)

    shape = data.shape

    if size < data.shape[0]:
        indexes = numpy.random.choice(data.shape[0], size, replace=False)
        data_reduced = data[indexes]
    else:
        data_reduced = data

    if bw_size < data.shape[0]:
        indexes = numpy.random.choice(data.shape[0], bw_size, replace=False)
        bw_data_reduced = data[indexes]
    else:
        bw_data_reduced = data

    return data, data_reduced, bw_data_reduced, scaler


def goal_kde(x, kde):
    test_value = numpy.array(x).reshape(1, -1)
    score = kde.score_samples(test_value)
    return -score[0]

def get_mle(data):
    multiprocessing.get_logger().info("setting up scaler and reducing data")
    data, data_reduced, data_reduced_bw, scaler = reduce_data(data, 32000, 10000)
    multiprocessing.get_logger().info("finished setting up scaler and reducing data")
    multiprocessing.get_logger().info("data_reduced shape %s", data_reduced.shape)

    BOUND_LOW_num = numpy.min(data, 0)
    BOUND_UP_num = numpy.max(data, 0)

    BOUND_LOW_trans = list(BOUND_LOW_num)
    BOUND_UP_trans = list(BOUND_UP_num)

    kde_ga = KernelDensity(kernel="gaussian", atol=atol, rtol=rtol)

    kde_ga, bandwidth, store = kde_util.get_bandwidth(kde_ga, data_reduced_bw)

    multiprocessing.get_logger().info("mle bandwidth: %.2g", bandwidth)

    multiprocessing.get_logger().info("fitting kde with mle bandwidth")
    kde_ga.fit(data)
    multiprocessing.get_logger().info("finished fitting and starting mle search")

    result_kde = scipy.optimize.differential_evolution(
        goal_kde, bounds=list(zip(BOUND_LOW_trans, BOUND_UP_trans)), args=(kde_ga,), disp=True, popsize=100
    )

    multiprocessing.get_logger().info("finished mle search")

    x = list(scaler.inverse_transform(numpy.array(result_kde.x).reshape(1, -1))[0])

    multiprocessing.get_logger().info("mle found %s", x)

    return x, kde_ga, scaler


def fitness(individual):
    return evo.fitness(individual, sys.argv[1])


def addChain(axis, *args):
    temp = [arg for arg in args if len(arg)]
    if len(temp) > 1:
        return numpy.concatenate(temp, axis=axis)
    else:
        return numpy.array(temp[0])


def process_mle(chain, gen, cache):
    mcmc_dir = Path(cache.settings["resultsDirMCMC"])

    mcmc_csv = Path(cache.settings["resultsDirMCMC"]) / "prob.csv"
    mle_h5 = Path(cache.settings["resultsDirMCMC"]) / "mle.h5"

    h5 = cadet.H5()
    h5.filename = mle_h5.as_posix()
    if mle_h5.exists():
        h5.load()

        if 0:  # h5.root.generations[-1] == gen:
            multiprocessing.get_logger().info("new information is not yet available and mle will quit")
            return

    multiprocessing.get_logger().info("process mle chain shape before %s", chain.shape)
    # This step cleans up bad entries
    chain = chain[~numpy.all(chain == 0, axis=1)]
    multiprocessing.get_logger().info("process mle chain shape after cleaning 0 entries %s", chain.shape)


    mle_x, kde, scaler = get_mle(chain)

    multiprocessing.get_logger().info("mle_x: %s", mle_x)

    mcmcDir = Path(cache.settings["resultsDirMCMC"])
    joblib.dump(scaler, mcmcDir / "kde_prior_scaler.joblib")
    joblib.dump(kde, mcmcDir / "kde_prior.joblib")

    mle_ind = util.convert_individual_inputorder(mle_x, cache)

    multiprocessing.get_logger().info("mle_ind: %s", mle_ind)

    temp = [
        mle_x,
    ]

    multiprocessing.get_logger().info("chain shape: %s", chain.shape)

    # run simulations for 5% 50% 95% and MLE vs experimental data
    percentile_splits = [5, 10, 50, 90, 95]
    percentile = numpy.percentile(chain, percentile_splits, 0)

    multiprocessing.get_logger().info("percentile: %s %s", percentile.shape, percentile)

    for row in percentile:
        temp.append(list(row))

    cadetValues = [util.convert_individual_inputorder(i, cache) for i in temp]
    cadetValues = numpy.array(cadetValues)

    multiprocessing.get_logger().info("cadetValues: %s %s", cadetValues.shape, cadetValues)

    map_function = util.getMapFunction()

    fitnesses = list(map_function(fitness, temp))

    simulations = {}
    for scores, csv_record, meta_score, results, individual in fitnesses:
        for name, value in results.items():
            sims = simulations.get(name, [])
            sims.append(value["simulation"])

            simulations[name] = sims

    multiprocessing.get_logger().info("type %s  value %s", type(cadetValues), cadetValues)

    pd = pandas.DataFrame(cadetValues, columns=cache.parameter_headers_actual)
    labels = ["MLE", "5", "10", "50", "90", "95"]
    pd.insert(0, "name", labels)
    pd.to_csv(mcmc_csv, index=False)

    h5.root.stat_labels = cache.parameter_headers_actual
    h5.root.percentile_splits = percentile_splits

    mle_x = numpy.array(mle_x)

    h5.root.mles = addChain(1, h5.root.mles, mle_x[:, numpy.newaxis])
    h5.root.stats = addChain(2, h5.root.stats, percentile[:, :, numpy.newaxis])
    h5.root.generations = addChain(0, h5.root.generations, [gen,])
    h5.root.stat_MLE = mle_ind

    h5.save()

    plot_mle(simulations, cache, labels)


def plot_mle(simulations, cache, labels):
    mcmc_dir = Path(cache.settings["resultsDirMCMC"])
    target = cache.target
    settings = cache.settings
    for experiment in settings["experiments"]:
        experimentName = experiment["name"]

        file_name = "%s_stats.png" % experimentName
        dst = mcmc_dir / file_name

        units_used = cache.target[experimentName]["units_used"]

        numPlotsSeq = [len(units_used)]
        # Shape and ShapeDecay have a chromatogram + derivative
        for feature in experiment["features"]:
            if feature["type"] in ("Shape", "ShapeDecay", "ShapeFront", "ShapeBack"):
                numPlotsSeq.append(2)
            elif feature["type"] in ("AbsoluteTime", "AbsoluteHeight"):
                pass
            else:
                numPlotsSeq.append(1)

        numPlots = sum(numPlotsSeq)

        exp_time = target[experimentName]["time"]
        exp_value = target[experimentName]["valueFactor"]

        fig = figure.Figure(figsize=[15, 15 * numPlots])
        canvas = FigureCanvas(fig)

        for idx, unit in enumerate(units_used):
            graph_simulations(simulations[experimentName], labels, unit, fig.add_subplot(numPlots, 1, idx + 1))

        graphIdx = 2
        for idx, feature in enumerate(experiment["features"]):
            featureName = feature["name"]
            featureType = feature["type"]

            feat = target[experimentName][featureName]

            selected = feat["selected"]
            exp_time = feat["time"][selected]
            exp_value = feat["value"][selected]

            if featureType in (
                "similarity",
                "similarityDecay",
                "similarityHybrid",
                "similarityHybrid2",
                "similarityHybrid2_spline",
                "similarityHybridDecay",
                "similarityHybridDecay2",
                "curve",
                "breakthrough",
                "dextran",
                "dextranHybrid",
                "dextranHybrid2",
                "dextranHybrid2_spline",
                "similarityCross",
                "similarityCrossDecay",
                "breakthroughCross",
                "SSE",
                "LogSSE",
                "breakthroughHybrid",
                "breakthroughHybrid2",
                "Shape",
                "ShapeDecay",
                "Dextran",
                "DextranAngle",
                "DextranTest",
                "ShapeDecaySimple",
                "ShapeFront",
                "ShapeBack",
            ):

                graph = fig.add_subplot(numPlots, 1, graphIdx)  # additional +1 added due to the overview plot

                for idx, (sim, label) in enumerate(zip(simulations[experimentName], labels)):
                    sim_time, sim_value = util.get_times_values(sim, target[experimentName][featureName])

                    if idx == 0:
                        linewidth = 2
                    else:
                        linewidth = 1

                    graph.plot(
                        sim_time,
                        sim_value,
                        "--",
                        label=label,
                        color=get_color(idx, len(simulations[experimentName]) + 1, cm_plot),
                        linewidth=linewidth,
                    )

                graph.plot(
                    exp_time,
                    exp_value,
                    "-",
                    label="Experiment",
                    color=get_color(len(simulations[experimentName]), len(simulations[experimentName]) + 1, cm_plot),
                    linewidth=2,
                )
                graphIdx += 1

            if featureType in (
                "similarity",
                "similarityDecay",
                "similarityHybrid",
                "similarityHybrid2",
                "similarityHybrid2_spline",
                "similarityHybridDecay",
                "similarityHybridDecay2",
                "curve",
                "breakthrough",
                "dextran",
                "dextranHybrid",
                "dextranHybrid2",
                "dextranHybrid2_spline",
                "similarityCross",
                "similarityCrossDecay",
                "breakthroughCross",
                "SSE",
                "LogSSE",
                "breakthroughHybrid",
                "breakthroughHybrid2",
                "Shape",
                "ShapeDecay",
                "Dextran",
                "DextranAngle",
                "DextranTest",
                "DextranQuad",
                "Dextran3",
                "DextranShape",
                "ShapeFront",
                "ShapeBack",
            ):

                graph = fig.add_subplot(numPlots, 1, graphIdx)  # additional +1 added due to the overview plot
                for idx, (sim, label) in enumerate(zip(simulations[experimentName], labels)):
                    sim_time, sim_value = util.get_times_values(sim, target[experimentName][featureName])
                    sim_spline = smoothing.smooth_data_derivative(
                        sim_time, sim_value, feat["critical_frequency"], feat["smoothing_factor"], feat["critical_frequency_der"]
                    )

                    if idx == 0:
                        linewidth = 2
                    else:
                        linewidth = 1

                    graph.plot(
                        sim_time,
                        sim_spline,
                        "--",
                        label=label,
                        color=get_color(idx, len(simulations[experimentName]) + 1, cm_plot),
                        linewidth=linewidth,
                    )

                exp_spline = smoothing.smooth_data_derivative(
                    exp_time, exp_value, feat["critical_frequency"], feat["smoothing_factor"], feat["critical_frequency_der"]
                )
                graph.plot(
                    exp_time,
                    exp_spline,
                    "-",
                    label="Experiment",
                    color=get_color(len(simulations[experimentName]), len(simulations[experimentName]) + 1, cm_plot),
                    linewidth=2,
                )
                graphIdx += 1

            graph.legend()

        fig.savefig(str(dst))


def graph_simulations(simulations, simulation_labels, unit, graph):
    linestyles = ["-", "--", "-.", ":"]
    for idx_sim, (simulation, label_sim) in enumerate(zip(simulations, simulation_labels)):

        comps = []

        ncomp = int(simulation.root.input.model.unit_001.ncomp)
        isotherm = bytes(simulation.root.input.model.unit_001.adsorption_model)

        hasSalt = isotherm in saltIsotherms

        solution_times = simulation.root.output.solution.solution_times

        hasColumn = any("column" in i for i in simulation.root.output.solution[unit].keys())
        hasPort = any("port" in i for i in simulation.root.output.solution[unit].keys())

        if hasColumn:
            for i in range(ncomp):
                comps.append(simulation.root.output.solution[unit]["solution_column_outlet_comp_%03d" % i])
        elif hasPort:
            for i in range(ncomp):
                comps.append(simulation.root.output.solution[unit]["solution_outlet_port_000_comp_%03d" % i])
        else:
            for i in range(ncomp):
                comps.append(simulation.root.output.solution[unit]["solution_outlet_comp_%03d" % i])

        if hasSalt:
            graph.set_title("Output")
            graph.plot(solution_times, comps[0], "b-", label="Salt")
            graph.set_xlabel("time (s)")

            # Make the y-axis label, ticks and tick labels match the line color.
            graph.set_ylabel("mM Salt", color="b")
            graph.tick_params("y", colors="b")

            axis2 = graph.twinx()
            for idx, comp in enumerate(comps[1:]):
                axis2.plot(
                    solution_times,
                    comp,
                    linestyles[idx],
                    color=get_color(idx_sim, len(simulation_labels), cm_plot),
                    label="P%s %s" % (idx, label_sim),
                )
            axis2.set_ylabel("mM Protein", color="r")
            axis2.tick_params("y", colors="r")

            lines, labels = graph.get_legend_handles_labels()
            lines2, labels2 = axis2.get_legend_handles_labels()
            axis2.legend(lines + lines2, labels + labels2, loc=0)
        else:
            graph.set_title("Output")

            for idx, comp in enumerate(comps):
                graph.plot(
                    solution_times,
                    comp,
                    linestyles[idx],
                    color=get_color(idx_sim, len(simulation_labels), cm_plot),
                    label="P%s %s" % (idx, label_sim),
                )
            graph.set_ylabel("mM Protein", color="r")
            graph.tick_params("y", colors="r")
            graph.set_xlabel("time (s)")

            lines, labels = graph.get_legend_handles_labels()
            graph.legend(lines, labels, loc=0)


def main():
    cache.setup_dir(sys.argv[1])
    util.setupLog(cache.settings["resultsDirLog"], "mle.log")
    cache.setup(sys.argv[1])

    mcmcDir = Path(cache.settings["resultsDirMCMC"])
    mcmc_h5 = mcmcDir / "mcmc.h5"

    with h5py.File(mcmc_h5, "r") as h5:
        flat_chain = h5["/flat_chain"][()]
        mcmc_acceptance = h5["/mcmc_acceptance"][()]
        process_mle(flat_chain, len(mcmc_acceptance), cache)


if __name__ == "__main__":
    main()
