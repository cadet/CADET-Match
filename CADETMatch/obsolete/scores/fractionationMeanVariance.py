import numpy
import pandas
from addict import Dict

import CADETMatch.score as score
import CADETMatch.util as util

name = "fractionationMeanVariance"
settings = Dict()
settings.adaptive = True
settings.badScore = 0
settings.meta_mask = True
settings.count = None


def run(sim_data, feature):
    simulation = sim_data["simulation"]
    funcs = feature["funcs"]
    components = feature["components"]
    numComponents = len(components)
    samplesPerComponent = feature["samplesPerComponent"]
    start = feature["start"]
    stop = feature["stop"]
    multiplier = 1.0 / samplesPerComponent

    time_centers = (start + stop) / 2.0

    times = simulation.root.output.solution.solution_times

    scores = []

    sim_values_sse = []
    exp_values_sse = []

    graph_sim = {}
    graph_exp = {}
    for (
        start,
        stop,
        component,
        values,
        func_mean_time,
        func_variance_time,
        func_mean_value,
        func_variance_value,
    ) in funcs:
        time_center = (start + stop) / 2.0

        sim_values = util.fractionate(
            start,
            stop,
            times,
            simulation.root.output.solution[feature["unit"]][
                "solution_outlet_comp_%03d" % component
            ],
        )

        (
            mean_sim_time,
            variance_sim_time,
            skew_sim_time,
            mean_sim_value,
            variance_sim_value,
            skew_sim_value,
        ) = util.fracStat(time_center, sim_values)

        exp_values_sse.extend(values)
        sim_values_sse.extend(sim_values)

        scores.append(func_mean_time(mean_sim_time))
        scores.append(func_variance_time(variance_sim_time))
        scores.append(func_mean_value(mean_sim_value))
        scores.append(func_variance_value(variance_sim_value))

        graph_sim[component] = list(zip(time_center, sim_values))
        graph_exp[component] = list(zip(time_center, values))

    sim_data["graph_exp"] = graph_exp
    sim_data["graph_sim"] = graph_sim

    return (
        scores,
        util.sse(numpy.array(sim_values_sse), numpy.array(exp_values_sse)),
        len(sim_values_sse),
        time_centers,
        numpy.array(sim_values_sse),
        numpy.array(exp_values_sse),
        [1.0 - i for i in scores],
    )


def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    data = pandas.read_csv(feature["fraction_csv"])
    rows, cols = data.shape

    headers = data.columns.values.tolist()

    start = numpy.array(data.iloc[:, 0])
    stop = numpy.array(data.iloc[:, 1])

    time_center = (start + stop) / 2.0

    temp["start"] = start
    temp["stop"] = stop

    smallestTime = min(data["Stop"] - data["Start"])
    abstolFraction = abstol * smallestTime

    funcs = []

    for idx, component in enumerate(headers[2:], 2):
        value = numpy.array(data.iloc[:, idx])

        (
            mean_time,
            variance_time,
            skew_time,
            mean_value,
            variance_value,
            skew_value,
        ) = util.fracStat(time_center, value)

        func_mean_time = score.time_function(CV_time, mean_time, diff_input=False)
        func_variance_time = score.value_function(variance_time)

        func_mean_value = score.value_function(mean_value, abstolFraction)
        func_variance_value = score.value_function(variance_value, abstolFraction / 1e5)

        funcs.append(
            (
                start,
                stop,
                int(component),
                value,
                func_mean_time,
                func_variance_time,
                func_mean_value,
                func_variance_value,
            )
        )

    settings.count = 4 * len(funcs)
    temp["funcs"] = funcs
    temp["components"] = [int(i) for i in headers[2:]]
    temp["samplesPerComponent"] = rows
    temp["unit"] = feature["unit_name"]
    return temp


def headers(experimentName, feature):
    data = pandas.read_csv(feature["fraction_csv"])
    rows, cols = data.shape

    data_headers = data.columns.values.tolist()

    temp = []
    for component in data_headers[2:]:
        temp.append(
            "%s_%s_Component_%s_time_mean"
            % (experimentName, feature["name"], component)
        )
        temp.append(
            "%s_%s_Component_%s_time_var" % (experimentName, feature["name"], component)
        )
        temp.append(
            "%s_%s_Component_%s_value_mean"
            % (experimentName, feature["name"], component)
        )
        temp.append(
            "%s_%s_Component_%s_value_var"
            % (experimentName, feature["name"], component)
        )
    return temp
