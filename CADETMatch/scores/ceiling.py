from addict import Dict

import CADETMatch.calc_coeff as calc_coeff
import CADETMatch.score as score
import CADETMatch.util as util

name = "Ceiling"


def get_settings(feature):
    settings = Dict()
    settings.adaptive = True
    settings.badScore = 0
    settings.meta_mask = True
    settings.count = 1
    settings.graph_der = 0
    settings.graph = 1
    settings.graph_frac = 0
    return settings


def run(sim_data, feature):
    "similarity, value, start stop"
    selected = feature["selected"]
    exp_data_values = feature["value"][selected]
    sim_time_values, sim_data_values = util.get_times_values(
        sim_data["simulation"], feature
    )

    max_value = max(sim_data_values)

    if max_value <= feature["max_value"]:
        value = 1.0
    else:
        value = feature["value_function"](max_value)

    temp = [
        value,
    ]

    return (
        temp,
        util.sse(sim_data_values, exp_data_values),
        len(sim_data_values),
        sim_time_values,
        sim_data_values,
        exp_data_values,
        [1.0 - i for i in temp],
    )


def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    temp = {}
    temp["value_function"] = score.value_function(feature["max_value"], abstol)
    return temp


def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature["name"])
    temp = ["%s_Height" % name]
    return temp
