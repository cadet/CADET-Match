import sys

import numpy
import pandas
import scipy.interpolate
from addict import Dict

import CADETMatch.score as score
import CADETMatch.util as util

"""DO NOT USE THIS SCORE. This score only exists for the purpose of a paper to confirm that
this is not an appropriate method to solve fractionation problems.
"""

name = "fractionationSSE"


def get_settings(feature):
    settings = Dict()
    settings.adaptive = False
    settings.badScore = sys.float_info.max
    settings.meta_mask = True
    settings.count = 1
    settings.graph_der = 0
    settings.graph = 0
    settings.graph_frac = 1
    return settings


def run(sim_data, feature):
    "similarity, value, start stop"
    simulation = sim_data["simulation"]
    start = feature["start"]
    stop = feature["stop"]
    comps = feature["comps"]
    data = feature["data"]

    time_center = (start + stop) / 2.0

    times = simulation.root.output.solution.solution_times

    sim_values_sse = []
    exp_values_sse = []

    graph_sim = {}
    graph_exp = {}

    for component in comps:
        exp_values = numpy.array(data[str(component)])
        selected = numpy.isfinite(exp_values)
        sim_value = simulation.root.output.solution[feature["unit"]][
            "solution_outlet_comp_%03d" % component
        ]

        spline = scipy.interpolate.InterpolatedUnivariateSpline(times, sim_value, ext=1)

        fractions = util.fractionate_spline(start[selected], stop[selected], spline)

        exp_values_sse.extend(exp_values[selected])
        sim_values_sse.extend(fractions)

        graph_sim[component] = list(zip(start[selected], stop[selected], fractions))
        graph_exp[component] = list(
            zip(start[selected], stop[selected], exp_values[selected])
        )

    # sort lists
    for key, value in graph_sim.items():
        value.sort()
    for key, value in graph_exp.items():
        value.sort()

    sim_data["graph_exp"] = graph_exp
    sim_data["graph_sim"] = graph_sim

    sse = util.sse(numpy.array(sim_values_sse), numpy.array(exp_values_sse))

    return (
        [
            sse,
        ],
        sse,
        len(sim_values_sse),
        time_center,
        numpy.array(sim_values_sse),
        numpy.array(exp_values_sse)
    )


def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    temp = {}
    data = pandas.read_csv(feature["fraction_csv"])

    temp["comps"] = [int(i) for i in data.columns.values.tolist()[2:]]

    temp["data"] = data
    temp["start"] = numpy.array(data.iloc[:, 0])
    temp["stop"] = numpy.array(data.iloc[:, 1])
    temp["unit"] = feature["unit_name"]

    start = numpy.array(data.iloc[:, 0])
    stop = numpy.array(data.iloc[:, 1])

    smallestTime = min(start - stop)
    abstolFraction = abstol * smallestTime

    headers = data.columns.values.tolist()
    temp["peak_max"] = data.iloc[:, 2:].max().min()
    return temp


def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature["name"])
    temp = ["%s_SSE" % name]
    return temp
