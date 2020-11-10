import CADETMatch.util as util
import CADETMatch.score as score
import scipy.stats
import scipy.interpolate
import numpy
from addict import Dict
import CADETMatch.smoothing as smoothing

name = "ShapeDecaySimple"


def get_settings(feature):
    settings = Dict()
    settings.adaptive = True
    settings.badScore = 0
    settings.meta_mask = True
    settings.count = 3
    return settings


def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data["simulation"], feature)
    selected = feature["selected"]

    exp_data_values = feature["value"][selected]
    exp_time_values = feature["time"][selected]

    sim_data_values_smooth, sim_data_values_der_smooth = smoothing.full_smooth(
        exp_time_values, sim_data_values, feature["critical_frequency"], feature["smoothing_factor"], feature["critical_frequency_der"]
    )

    [high, low] = util.find_peak(exp_time_values, sim_data_values_smooth)

    time_high, value_high = high

    pearson, diff_time = score.pearson_spline(exp_time_values, sim_data_values_smooth, feature["smooth_value"])

    pearson_der = score.pearson_offset(diff_time, exp_time_values, sim_data_values_der_smooth, feature["smooth_value_der"])

    temp = [pearson, feature["time_function"](numpy.abs(diff_time)), pearson_der]
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
    s, crit_fs, crit_fs_der = smoothing.find_smoothing_factors(selectedTimes, selectedValues, name, cache)
    exp_data_values_smooth, exp_data_values_der_smooth = smoothing.full_smooth(selectedTimes, selectedValues, crit_fs, s, crit_fs_der)

    temp = {}
    temp["peak"] = util.find_peak(selectedTimes, exp_data_values_smooth)[0]

    temp["time_function"] = score.time_function_decay(feature["time"][-1])
    temp["peak_max"] = max(selectedValues)
    temp["smoothing_factor"] = s
    temp["critical_frequency"] = crit_fs
    temp["critical_frequency_der"] = crit_fs_der
    temp["smooth_value"] = exp_data_values_smooth
    temp["smooth_value_der"] = exp_data_values_der_smooth
    return temp


def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature["name"])
    temp = ["%s_Similarity" % name, "%s_Time" % name, "%s_Derivative_Similarity" % name]
    return temp
