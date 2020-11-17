import multiprocessing
import sys

import numpy
import numpy.linalg
import scipy.stats
from addict import Dict

import CADETMatch.score as score
import CADETMatch.smoothing as smoothing
import CADETMatch.util as util

name = "DextranSSE"


def get_settings(feature):
    settings = Dict()
    settings.adaptive = False
    settings.badScore = -sys.float_info.max
    settings.meta_mask = True
    settings.count = 1
    return settings


def run(sim_data, feature):
    "special score designed for dextran. This looks at only the front side of the peak up to the maximum slope and pins a value at the elbow in addition to the top"
    exp_time_values = feature["time"]

    min_value = feature["min_value"]
    max_value = feature["max_value"]

    selected = feature["selected"]

    sim_time_values, sim_data_values = util.get_times_values(
        sim_data["simulation"], feature
    )

    if (
        max(sim_data_values) < max_value
    ):  # the system has no point higher than the value we are looking for
        # remove hard failure
        max_value = max(sim_data_values)

    exp_time_values = exp_time_values[selected]
    exp_data_zero = feature["exp_data_zero"]

    sim_data_values_smooth = smoothing.smooth_data(
        sim_time_values,
        sim_data_values,
        feature["critical_frequency"],
        feature["smoothing_factor"],
    )
    (
        sim_data_zero,
        sim_min_time,
        sim_min_value,
        sim_max_time,
        sim_max_value,
    ) = score.cut_zero(sim_time_values, sim_data_values_smooth, min_value, max_value)

    sse = util.sse(sim_data_zero, exp_data_zero)

    data = (
        [
            -sse,
        ],
        sse,
        len(sim_data_zero),
        sim_time_values,
        sim_data_zero,
        exp_data_zero,
        [
            sse,
        ],
    )

    return data


def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    temp = {}
    # change the stop point to be where the max positive slope is along the searched interval
    name = "%s_%s" % (sim.root.experiment_name, feature["name"])
    s, crit_fs, crit_fs_der = smoothing.find_smoothing_factors(
        selectedTimes, selectedValues, name, cache
    )
    smooth_value, values_der = smoothing.full_smooth(
        selectedTimes, selectedValues, crit_fs, s, crit_fs_der
    )

    spline_der = scipy.interpolate.InterpolatedUnivariateSpline(
        selectedTimes, values_der, ext=1
    )
    spline = scipy.interpolate.InterpolatedUnivariateSpline(
        selectedTimes, smooth_value, ext=1
    )

    min_time, min_value, max_time, max_value = score.find_cuts(
        selectedTimes, smooth_value, spline, spline_der
    )

    (
        exp_data_zero,
        exp_min_time,
        exp_min_value,
        exp_max_time,
        exp_max_value,
    ) = score.cut_zero(selectedTimes, smooth_value, min_value, max_value)

    if exp_min_time is not None:
        min_time = exp_min_time
        min_value = exp_min_value
        max_time = exp_max_time
        max_value = exp_max_value

    multiprocessing.get_logger().info(
        "Dextran %s  start: %s   stop: %s  max value: %s",
        name,
        min_time,
        max_time,
        max_value,
    )

    temp["min_time"] = feature["start"]
    temp["max_time"] = feature["stop"]
    temp["min_value"] = min_value
    temp["max_value"] = max_value
    temp["exp_data_zero"] = exp_data_zero
    temp["peak_max"] = max_value
    temp["smoothing_factor"] = s
    temp["critical_frequency"] = crit_fs
    temp["critical_frequency_der"] = crit_fs_der
    temp["smooth_value"] = smooth_value
    return temp


def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature["name"])
    temp = [
        "%s_SSE" % name,
    ]
    return temp
