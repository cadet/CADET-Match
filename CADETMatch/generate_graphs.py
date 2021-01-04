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
import pandas
import scipy.interpolate
from addict import Dict
from cadet import H5, Cadet
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

import CADETMatch.loggerwriter as loggerwriter
import CADETMatch.smoothing as smoothing
import CADETMatch.util as util
from CADETMatch.cache import cache

saltIsotherms = {
    b"STERIC_MASS_ACTION",
    b"SELF_ASSOCIATION",
    b"MULTISTATE_STERIC_MASS_ACTION",
    b"SIMPLE_MULTISTATE_STERIC_MASS_ACTION",
    b"BI_STERIC_MASS_ACTION",
}


import matplotlib.cm
from matplotlib.colors import ListedColormap

cmap = matplotlib.cm.winter

# Get the colormap colors
my_cmap = cmap(numpy.arange(cmap.N))

# Set alpha
my_cmap[:, -1] = 1.0

# Create new colormap
my_cmap = ListedColormap(my_cmap)

cm_plot = matplotlib.cm.gist_rainbow


def get_color(idx, max_colors, cmap):
    return cmap(1.0 * float(idx) / max_colors)


def main(map_function):
    cache.setup_dir(sys.argv[1])
    util.setupLog(cache.settings["resultsDirLog"], "graph.log")
    cache.setup(sys.argv[1])

    multiprocessing.get_logger().info("graphing directory %s", os.getcwd())

    # full generation 1 = 2D, 2 = 2D + 3D
    fullGeneration = int(sys.argv[2])

    # load all data, all data that can be changed by the main process MUST be loaded and this point and the files locked
    # this allows the main process to wait for data loading and then continue once complete

    resultDir = Path(cache.settings["resultsDirBase"])
    progress = resultDir / "progress.csv"
    result_lock = resultDir / "result.lock"

    result_h5 = resultDir / "result.h5"

    progress_df = pandas.read_csv(progress)

    result_data = H5()
    result_data.filename = result_h5.as_posix()
    result_data.load(
        paths=[
            "/output",
            "/output_meta",
            "/is_extended_input",
            "/probability",
            "/input_transform_extended",
            "/input_transform",
            "/mean",
            "/confidence",
            "/distance_correct",
        ],
        lock=True
    )

    graphMeta(cache, map_function)
    graphProgress(cache, map_function, progress_df)

    if fullGeneration:
        graphSpace(fullGeneration, cache, map_function, result_data)
        graphExperiments(cache, map_function)


def graphDistance(cache, map_function, data):
    output_distance = cache.settings["resultsDirSpace"] / "distance"
    output_distance_meta = output_distance / "meta"

    output_distance.mkdir(parents=True, exist_ok=True)

    output_distance_meta.mkdir(parents=True, exist_ok=True)

    parameter_headers_actual = cache.parameter_headers_actual
    score_headers = cache.score_headers
    meta_headers = cache.meta_headers

    temp = []

    if (
        "mean" in data.root
        and "confidence" in data.root
        and "distance_correct" in data.root
    ):
        for idx_parameter, parameter in enumerate(parameter_headers_actual):
            for idx_score, score in enumerate(score_headers):
                temp.append(
                    (
                        output_distance,
                        parameter,
                        idx_parameter,
                        score,
                        idx_score,
                        data.root.distance_correct[:, idx_parameter],
                        data.root.output[:, idx_score],
                        data.root.mean[:, idx_parameter],
                        data.root.confidence[:, idx_parameter],
                    )
                )

        for idx_parameter, parameter in enumerate(parameter_headers_actual):
            for idx_score, score in enumerate(meta_headers):
                temp.append(
                    (
                        output_distance_meta,
                        parameter,
                        idx_parameter,
                        score,
                        idx_score,
                        data.root.distance_correct[:, idx_parameter],
                        data["output_meta"][:, idx_score],
                        data.root.mean[:, idx_parameter],
                        data.root.confidence[:, idx_parameter],
                    )
                )

        list(map_function(plot_2d_scatter, temp))


def plot_2d_scatter(args):
    (
        output_directory,
        parameter,
        parameter_idx,
        score,
        score_idx,
        parameter_data,
        score_data,
        mean_data,
        confidence_data,
    ) = args
    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)
    graph.scatter(parameter_data, score_data, c=score_data, cmap=my_cmap)
    graph.set_xlabel(parameter)
    graph.set_ylabel(score)
    filename = "%s_%s.png" % (parameter_idx, score_idx)
    fig.savefig(str(output_directory / filename))

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)
    graph.scatter(mean_data, score_data, c=score_data, cmap=my_cmap)
    graph.set_xlabel(parameter)
    graph.set_ylabel(score)
    filename = "%s_%s_mean.png" % (parameter_idx, score_idx)
    fig.savefig(str(output_directory / filename))

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)
    graph.scatter(confidence_data, score_data, c=score_data, cmap=my_cmap)
    graph.set_xlabel(parameter)
    graph.set_ylabel(score)
    filename = "%s_%s_confidence.png" % (parameter_idx, score_idx)
    fig.savefig(str(output_directory / filename))


def graphExperiments(cache, map_function):
    multiprocessing.get_logger().info("starting simulation graphs")
    directory = Path(cache.settings["resultsDirEvo"])

    # find all items in directory
    pathsH5 = directory.glob("*.h5")
    pathsPNG = directory.glob("*.png")

    # make set of items based on removing everything after _
    existsH5 = {str(path.name).split("_", 1)[0] for path in pathsH5}
    existsPNG = {str(path.name).split("_", 1)[0] for path in pathsPNG}

    toGenerate = existsH5 - existsPNG

    args = zip(
        toGenerate,
        itertools.repeat(sys.argv[1]),
        itertools.repeat(cache.settings["resultsDirEvo"]),
        itertools.repeat("%s_%s_EVO.png"),
    )

    list(map_function(plotExperiments, args))
    multiprocessing.get_logger().info("ending simulation graphs")


def graphMeta(cache, map_function):
    multiprocessing.get_logger().info("starting meta graphs")
    directory = Path(cache.settings["resultsDirMeta"])

    # find all items in directory
    pathsH5 = directory.glob("*.h5")
    pathsPNG = directory.glob("*.png")

    # make set of items based on removing everything after _
    existsH5 = {str(path.name).split("_", 1)[0] for path in pathsH5}
    existsPNG = {str(path.name).split("_", 1)[0] for path in pathsPNG}

    toGenerate = existsH5 - existsPNG

    args = zip(
        toGenerate,
        itertools.repeat(sys.argv[1]),
        itertools.repeat(cache.settings["resultsDirMeta"]),
        itertools.repeat("%s_%s_meta.png"),
    )

    list(map_function(plotExperiments, args))
    multiprocessing.get_logger().info("ending meta graphs")


def graph_simulation_unit(simulation, unit, graph):
    ncomp = int(simulation.root.input.model[unit].ncomp)
    isotherm = bytes(simulation.root.input.model[unit].adsorption_model)

    # This does not work correctly since we could be reading an outlet or tubing that does not have an isotherm
    # hasSalt = isotherm in saltIsotherms

    solution_times = simulation.root.output.solution.solution_times

    comps = []

    hasColumn = any("column" in i for i in simulation.root.output.solution[unit].keys())
    hasPort = any("port" in i for i in simulation.root.output.solution[unit].keys())

    if hasColumn:
        for i in range(ncomp):
            comps.append(
                simulation.root.output.solution[unit][
                    "solution_column_outlet_comp_%03d" % i
                ]
            )
    elif hasPort:
        for i in range(ncomp):
            comps.append(
                simulation.root.output.solution[unit][
                    "solution_outlet_port_000_comp_%03d" % i
                ]
            )
    else:
        for i in range(ncomp):
            comps.append(
                simulation.root.output.solution[unit]["solution_outlet_comp_%03d" % i]
            )

    # for now lets assume that if your first component is much higher concentration than the other components then the first component is salt
    max_comps = [max(comp) for comp in comps]

    hasSalt = any([max_comps[0] > (comp * 10) for comp in max_comps[1:]])

    if hasSalt:
        graph.set_title("Output %s" % unit)
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
                "-",
                color=get_color(idx, len(comps) - 1, cm_plot),
                label="P%s" % idx,
            )
        axis2.set_ylabel("mM Protein", color="r")
        axis2.tick_params("y", colors="r")

        lines, labels = graph.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis2.legend(lines + lines2, labels + labels2, loc=0)
    else:
        graph.set_title("Output %s" % unit)

        # colors = ['r', 'g', 'c', 'm', 'y', 'k']
        for idx, comp in enumerate(comps):
            graph.plot(
                solution_times,
                comp,
                "-",
                color=get_color(idx, len(comps), cm_plot),
                label="P%s" % idx,
            )
        graph.set_ylabel("mM Protein", color="r")
        graph.tick_params("y", colors="r")
        graph.set_xlabel("time (s)")

        lines, labels = graph.get_legend_handles_labels()
        graph.legend(lines, labels, loc=0)


def convert_frac(start, stop, values):
    x = []
    y = []
    for xl, xu, val in zip(start, stop, values):
        x.append(xl)
        y.append(val)
        x.append(xu)
        y.append(val)
    return x, y


def plotExperiments(args):
    save_name_base, json_path, directory, file_pattern = args
    if json_path != cache.json_path:
        cache.setup(json_path)

    target = cache.target
    settings = cache.settings
    for experiment in settings["experiments"]:
        experimentName = experiment["name"]

        dst = Path(directory, file_pattern % (save_name_base, experimentName))

        if dst.exists():
            # this item has already been plotted, this means that plots are occurring faster than generations are so kill this version
            sys.exit()

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
        exp_value = target[experimentName]["value"]

        fig = figure.Figure(figsize=[24, 12 * numPlots])
        canvas = FigureCanvas(fig)

        simulation = Cadet()
        h5_path = Path(directory) / (
            file_pattern.replace("png", "h5") % (save_name_base, experimentName)
        )
        simulation.filename = h5_path.as_posix()
        simulation.load()

        results = {}
        results["simulation"] = simulation

        for idx, unit in enumerate(units_used):
            graph_simulation_unit(
                simulation, unit, fig.add_subplot(numPlots, 1, idx + 1)
            )

        graphIdx = idx + 2
        for idx, feature in enumerate(experiment["features"]):
            featureName = feature["name"]
            featureType = feature["type"]

            feat = target[experimentName][featureName]

            selected = feat["selected"]
            exp_time = feat["time"][selected]
            exp_value = feat["value"][selected]

            sim_time, sim_value = get_times_values(
                simulation, target[experimentName][featureName]
            )

            if featureType in (
                "similarity",
                "similarityDecay",
                "curve",
                "SSE",
                "Shape",
                "ShapeDecay",
                "Dextran",
                "DextranShape",
                "ShapeDecaySimple",
                "ShapeSimple",
                "DextranSSE",
                "ShapeFront",
                "ShapeBack",
                "ShapeNoDer",
                "ShapeDecayNoDer",
            ):

                graph = fig.add_subplot(
                    numPlots, 1, graphIdx
                )  # additional +1 added due to the overview plot
                graph.plot(sim_time, sim_value, "r--", label="Simulation")
                graph.plot(exp_time, exp_value, "g:", label="Experiment")
                graphIdx += 1

            if featureType in (
                "Shape",
                "ShapeDecay",
                "ShapeFront",
                "ShapeBack",
            ):
                exp_spline = smoothing.smooth_data_derivative(
                    exp_time,
                    exp_value,
                    feat["critical_frequency"],
                    feat["smoothing_factor"],
                    feat["critical_frequency_der"],
                )
                sim_spline = smoothing.smooth_data_derivative(
                    sim_time,
                    sim_value,
                    feat["critical_frequency"],
                    feat["smoothing_factor"],
                    feat["critical_frequency_der"],
                )

                graph = fig.add_subplot(
                    numPlots, 1, graphIdx
                )  # additional +1 added due to the overview plot
                graph.plot(sim_time, sim_spline, "r--", label="Simulation")
                graph.plot(exp_time, exp_spline, "g:", label="Experiment")
                graphIdx += 1

            if featureType in (
                "fractionationSlide",
                "fractionationSSE",
            ):
                cache.scores[featureType].run(results, feat)

                graph_exp = results["graph_exp"]
                graph_sim = results["graph_sim"]
                graph_sim_offset = results.get("graph_sim_offset", {})

                findMax = 0
                max_comp = {}
                for idx, (key, value) in enumerate(graph_sim.items()):
                    (time_start, time_stop, values) = zip(*value)
                    max_value = max(values)
                    findMax = max(findMax, max_value)
                    max_comp[key] = max_value

                for idx, (key, value) in enumerate(graph_exp.items()):
                    (time_start, time_stop, values) = zip(*value)
                    max_value = max(values)
                    findMax = max(findMax, max_value)
                    max_comp[key] = max(max_comp[key], max_value)

                graph = fig.add_subplot(
                    numPlots, 1, graphIdx
                )  # additional +1 added due to the overview plot
                factors = []
                for idx, (key, value) in enumerate(graph_sim.items()):
                    (time_start, time_stop, values) = zip(*value)
                    values = numpy.array(values)
                    mult = 1.0
                    if max(values) < (0.2 * findMax):
                        mult = findMax / (2 * max_comp[key])
                        values = values * mult
                    factors.append(mult)

                    time_offset = graph_sim_offset.get(key, None)
                    if time_offset is not None:
                        if time_offset > 0:
                            time_str = "time early (s): %.3g" % abs(time_offset)
                        else:
                            time_str = "time late (s): %.3g" % abs(time_offset)
                    else:
                        time_str = ""

                    label = "Simulation Comp: %s Mult:%.2f %s" % (key, mult, time_str)

                    x, y = convert_frac(time_start, time_stop, values)

                    graph.plot(
                        x,
                        y,
                        "--",
                        color=get_color(idx, len(graph_sim), cm_plot),
                        label=label,
                    )

                for idx, (key, value) in enumerate(graph_exp.items()):
                    (time_start, time_stop, values) = zip(*value)
                    values = numpy.array(values)
                    mult = factors[idx]
                    values = values * mult

                    x, y = convert_frac(time_start, time_stop, values)

                    graph.plot(
                        x,
                        y,
                        ":",
                        color=get_color(idx, len(graph_sim), cm_plot),
                        label="Experiment Comp: %s Mult:%.2f" % (key, mult),
                    )
                graphIdx += 1
            graph.legend()
        fig.set_size_inches((24, 12 * numPlots))
        fig.savefig(str(dst))


def graphSpace(fullGeneration, cache, map_function, results):
    multiprocessing.get_logger().info("starting space graphs")
    output_2d = cache.settings["resultsDirSpace"] / "2d"
    output_3d = cache.settings["resultsDirSpace"] / "3d"
    output_prob = cache.settings["resultsDirSpace"] / "probability"

    output_2d.mkdir(parents=True, exist_ok=True)
    output_3d.mkdir(parents=True, exist_ok=True)
    output_prob.mkdir(parents=True, exist_ok=True)

    if results.root.is_extended_input:
        input = results.root.input_transform_extended
    else:
        input = results.root.input_transform

    if isinstance(results.root.probability, Dict):
        probability = None
    else:
        probability = numpy.squeeze(results.root.probability)
        probability_idx = numpy.max(probability) / probability < 1e5
        probability = probability[probability_idx]

    output = results.root.output
    output_meta = results.root.output_meta

    input_headers = cache.parameter_headers
    output_headers = cache.score_headers
    meta_headers = cache.meta_headers

    input_indexes = list(range(len(input_headers)))
    output_indexes = list(range(len(output_headers)))
    meta_indexes = list(range(len(meta_headers)))

    comp_two = list(itertools.combinations(input_indexes, 2))
    comp_one = list(itertools.combinations(input_indexes, 1))

    # 2d plots
    if fullGeneration >= 1:
        multiprocessing.get_logger().info("starting 2d space graphs")
        graphDistance(cache, map_function, results)

        seq = []
        for (x,), y in itertools.product(comp_one, output_indexes):
            seq.append(
                [
                    output_2d.as_posix(),
                    input_headers[x],
                    output_headers[y],
                    input[:, x],
                    output[:, y],
                ]
            )

        for (x,), y in itertools.product(comp_one, meta_indexes):
            seq.append(
                [
                    output_2d.as_posix(),
                    input_headers[x],
                    meta_headers[y],
                    input[:, x],
                    output_meta[:, y],
                ]
            )

        if probability is not None:
            for (x,) in comp_one:
                seq.append(
                    [
                        output_prob.as_posix(),
                        input_headers[x],
                        "probability",
                        input[probability_idx, x],
                        probability,
                    ]
                )

        list(map_function(plot_2d, seq))
        multiprocessing.get_logger().info("ending 2d space graphs")

    # 3d plots
    if fullGeneration >= 2:
        multiprocessing.get_logger().info("starting 3d space graphs")
        seq = []
        for (x, y), z in itertools.product(comp_two, output_indexes):
            seq.append(
                [
                    output_3d.as_posix(),
                    input_headers[x],
                    input_headers[y],
                    output_headers[z],
                    input[:, x],
                    input[:, y],
                    output[:, z],
                ]
            )

        for (x, y), z in itertools.product(comp_two, meta_indexes):
            seq.append(
                [
                    output_3d.as_posix(),
                    input_headers[x],
                    input_headers[y],
                    meta_headers[z],
                    input[:, x],
                    input[:, y],
                    output_meta[:, z],
                ]
            )
        list(map_function(plot_3d, seq))
        multiprocessing.get_logger().info("ending 3d space graphs")
    multiprocessing.get_logger().info("ending space graphs")


def plot_3d(arg):
    "This leaks memory and is run in a separate short-lived process, do not integrate into the matching main process"
    directory_path, header1, header2, header3, data1, data2, scores = arg
    directory = Path(directory_path)

    x_str = "%s"
    x = data1
    if max(x) / min(x) > 100:
        x_str = "log10(%s)"
        x = numpy.log10(x)

    y_str = "%s"
    y = data2
    if max(y) / min(y) > 100:
        y_str = "log10(%s)"
        y = numpy.log10(y)

    z_str = "%s"
    z = scores
    if max(z) / min(z) > 100:
        z_str = "log10(%s)"
        z = numpy.log10(z)

    fig = figure.Figure(figsize=[20, 20])
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=z, cmap=my_cmap)
    ax.set_xlabel(x_str % header1)
    ax.set_ylabel(y_str % header2)
    ax.set_zlabel(z_str % header3)
    filename = "%s_%s_%s.png" % (header1, header2, header3)
    filename = filename.replace(":", "_").replace("/", "_")
    fig.savefig(str(directory / filename))


def plot_2d(arg):
    directory_path, header_x, scoreName, data, scores = arg

    multiprocessing.get_logger().info("2d graph %s %s start" % (header_x, scoreName))

    plot_2d_single(directory_path, header_x, scoreName, data, scores)

    multiprocessing.get_logger().info("2d graph %s %s stop" % (header_x, scoreName))


def plot_2d_single(directory_path, header_x, scoreName, data, scores):
    directory = Path(directory_path)

    fig = figure.Figure(figsize=[10, 10])
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)

    format = "%s"
    if numpy.max(data) / numpy.min(data[data > 0]) > 100.0:
        keep = data > 0
        data = numpy.log10(data[keep])
        scores = scores[keep]
        format = "log10(%s)"

    format_scores = "%s"
    if numpy.max(scores) / numpy.min(scores[scores > 0]) > 100.0:
        keep = scores > 0
        scores = numpy.log10(scores[keep])
        data = data[keep]
        format_scores = "log10(%s)"

    data_norm = (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))
    scores_norm = (scores - numpy.min(scores)) / (numpy.max(scores) - numpy.min(scores))
    scatter_data = numpy.array([data_norm, scores_norm]).T
    scatter_data_round = numpy.round(scatter_data, 2)
    scatter_data_round, idx, counts = numpy.unique(
        scatter_data_round, axis=0, return_index=True, return_counts=True
    )

    counts_norm = (counts - numpy.min(counts)) / (
        numpy.max(counts) - numpy.min(counts)
    ) * 100 + 10

    graph.scatter(data[idx], scores[idx], c=scores[idx], s=counts_norm, cmap=cmap)

    graph.set_xlabel(format % header_x)
    graph.set_ylabel(format_scores % scoreName)
    graph.set_xlim(min(data), max(data), auto=True)
    filename = "%s_%s.png" % (header_x, scoreName)
    filename = filename.replace(":", "_").replace("/", "_")

    fig.savefig(str(directory / filename))


def graphProgress(cache, map_function, df):
    multiprocessing.get_logger().info("starting progress graphs")
    output = cache.settings["resultsDirProgress"]

    x = [
        "Generation",
    ]
    y = ["Meta Front", "Meta Min", "Meta Product", "Meta Mean", "Meta SSE", "Meta RMSE"]

    temp = []
    for x, y in itertools.product(x, y):
        if x != y:
            temp.append((df[[x, y]], output))

    list(map_function(singleGraphProgress, temp))
    multiprocessing.get_logger().info("ending progress graphs")


def singleGraphProgress(arg):
    df, output = arg

    i, j = list(df)

    fig = figure.Figure(figsize=(10, 10))
    canvas = FigureCanvas(fig)

    graph = fig.add_subplot(1, 1, 1)

    x = df[i].to_numpy()
    y = df[j].to_numpy()

    title_str = "%s vs %s"
    y_label = "%s"
    if numpy.max(y) / numpy.min(y) > 100.0:
        y = numpy.log10(y)
        title_str = "%s vs log10(%s)"
        y_label = "log10(%s)"

    graph.plot(x, y)
    y_max = max(y)
    y_min = min(y)
    graph.set_ylim((1.1 * y_min, 1.1 * y_max))
    graph.set_title(title_str % (i, j))
    graph.set_xlabel(i)
    graph.set_ylabel(y_label % j)

    filename = "%s vs %s.png" % (i, j)
    file_path = output / filename
    fig.savefig(str(file_path))


def get_times_values(simulation, target, selected=None):

    times = simulation.root.output.solution.solution_times

    isotherm = target["isotherm"]

    if isinstance(isotherm, list):
        values = numpy.sum([simulation[i] for i in isotherm], 0)
    else:
        values = simulation[isotherm]

    if selected is None:
        selected = target["selected"]

    return times[selected], values[selected] * target["factor"]


if __name__ == "__main__":
    map_function = util.getMapFunction()
    main(map_function)
    sys.exit()
