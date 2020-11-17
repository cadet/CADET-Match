import numpy
import scipy.interpolate
import scipy.stats
from addict import Dict

import CADETMatch.score as score
import CADETMatch.smoothing as smoothing
import CADETMatch.util as util

name = "ShapeDecayNoDer"


def get_settings(feature):
    settings = Dict()
    settings.adaptive = True
    settings.badScore = 0
    settings.meta_mask = True
    settings.count = 3
    return settings


def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(
        sim_data["simulation"], feature
    )
    selected = feature["selected"]

    exp_data_values = feature["value"][selected]
    exp_time_values = feature["time"][selected]

    [high, low] = util.find_peak(exp_time_values, sim_data_values)

    time_high, value_high = high

    pearson, diff_time = score.pearson_spline(
        exp_time_values, sim_data_values, feature["smooth_value"]
    )

    temp = [
        pearson,
        feature["value_function"](value_high),
        feature["time_function"](numpy.abs(diff_time)),
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
    name = "%s_%s" % (sim.root.experiment_name, feature["name"])
    s, crit_fs, crit_fs_der = smoothing.find_smoothing_factors(
        selectedTimes, selectedValues, name, cache
    )

    temp = {}
    temp["peak"] = util.find_peak(selectedTimes, selectedValues)[0]
    temp["time_function"] = score.time_function_decay(feature["time"][-1])
    temp["value_function"] = score.value_function(temp["peak"][1], abstol)
    temp["peak_max"] = max(selectedValues)
    temp["smoothing_factor"] = s
    temp["critical_frequency"] = crit_fs
    temp["critical_frequency_der"] = crit_fs_der
    temp["smooth_value"] = smoothing.smooth_data(
        selectedTimes, selectedValues, crit_fs, s
    )
    return temp


def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature["name"])
    temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name]
    return temp
