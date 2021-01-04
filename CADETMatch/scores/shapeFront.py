import numpy
import scipy.interpolate
import scipy.stats
from addict import Dict

import CADETMatch.score as score
import CADETMatch.smoothing as smoothing
import CADETMatch.util as util

# Used to match the front side of a peak (good for breakthrough curves, the drops the derivative lower score)

name = "ShapeFront"


def get_settings(feature):
    settings = Dict()
    settings.adaptive = True
    settings.badScore = 0
    settings.meta_mask = True

    derivative = feature.get("derivative", 1)

    settings.graph = 1
    settings.graph_frac = 0

    if derivative:
        settings.count = 5
        settings.graph_der = 1
    else:
        settings.count = 3
        settings.graph_der = 0

    return settings


def slice_front(times, seq, seq_der, feature):
    "create an array of zeros with only the slice we need in it"
    # resample to 10hz
    new_times = numpy.linspace(times[0], times[-1], int(times[-1] - times[0]) * 10)

    spline = scipy.interpolate.InterpolatedUnivariateSpline(times, seq, ext=1)
    spline_der = scipy.interpolate.InterpolatedUnivariateSpline(times, seq_der, ext=1)

    seq_resample = spline(new_times)
    seq_der_resample = spline_der(new_times)

    new_seq = numpy.zeros(new_times.shape)
    new_seq_der = numpy.zeros(new_times.shape)

    max_value = numpy.max(seq_resample)

    max_percent = feature.get("max_percent", 0.98)
    min_percent = feature.get("min_percent", 0.02)

    select_max = max_percent * max_value
    select_min = min_percent * max_value

    max_index = numpy.argmax(seq_resample)
    idx_min = numpy.argmin((seq_resample[:max_index] - select_min) ** 2)
    idx_max = numpy.argmin((seq_resample[:max_index] - select_max) ** 2)

    min_value = seq_resample[idx_min]
    max_value = seq_resample[idx_max]
    new_seq[idx_min : idx_max + 1] = seq_resample[idx_min : idx_max + 1]
    new_seq_der[idx_min : idx_max + 1] = seq_der_resample[idx_min : idx_max + 1]
    return new_times, new_seq, new_seq_der, min_value, max_value


def slice_front_values(new_times, times, seq, seq_der, min_value, max_value, feature):
    "create an array of zeros with only the slice we need in it"
    spline = scipy.interpolate.InterpolatedUnivariateSpline(times, seq, ext=1)
    spline_der = scipy.interpolate.InterpolatedUnivariateSpline(times, seq_der, ext=1)

    seq_resample = spline(new_times)
    seq_der_resample = spline_der(new_times)

    new_seq = numpy.zeros(new_times.shape)
    new_seq_der = numpy.zeros(new_times.shape)

    max_index = numpy.argmax(seq_resample)
    idx_min = numpy.argmin((seq_resample[:max_index] - min_value) ** 2)
    idx_max = numpy.argmin((seq_resample[:max_index] - max_value) ** 2)

    new_seq[idx_min : idx_max + 1] = seq_resample[idx_min : idx_max + 1]
    new_seq_der[idx_min : idx_max + 1] = seq_der_resample[idx_min : idx_max + 1]
    return new_seq, new_seq_der


def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(
        sim_data["simulation"], feature
    )
    selected = feature["selected"]

    exp_data_values = feature["value"][selected]
    exp_time_values = feature["time"][selected]
    exp_data_values_der_smooth = feature["exp_data_values_der_smooth"]
    exp_data_values_smooth = feature["exp_data_values_smooth"]
    new_times = feature["new_times"]
    min_value = feature["min_value"]
    max_value = feature["max_value"]

    sim_data_values_smooth, sim_data_values_der_smooth = smoothing.full_smooth(
        exp_time_values,
        sim_data_values,
        feature["critical_frequency"],
        feature["smoothing_factor"],
        feature["critical_frequency_der"],
    )

    ret = slice_front_values(
        new_times,
        exp_time_values,
        sim_data_values_smooth,
        sim_data_values_der_smooth,
        min_value,
        max_value,
        feature,
    )
    sim_data_values_smooth_cut, sim_data_values_der_smooth_cut = ret

    [high, low] = util.find_peak(exp_time_values, sim_data_values_smooth)

    time_high, value_high = high

    pearson, diff_time = score.pearson_spline(
        new_times, sim_data_values_smooth_cut, exp_data_values_smooth
    )

    derivative = feature.get("derivative", 1)

    if derivative:
        pearson_der = score.pearson_offset(
            diff_time,
            new_times,
            sim_data_values_der_smooth_cut,
            exp_data_values_der_smooth,
        )
        [highs_der, lows_der] = util.find_peak(
            exp_time_values, sim_data_values_der_smooth
        )

    temp = [
        pearson,
        feature["value_function"](value_high),
        feature["time_function"](numpy.abs(diff_time)),
    ]
    if derivative:
        temp.extend([pearson_der, feature["value_function_high"](highs_der[1])])

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
    exp_data_values_smooth, exp_data_values_der_smooth = smoothing.full_smooth(
        selectedTimes, selectedValues, crit_fs, s, crit_fs_der
    )

    [high, low] = util.find_peak(selectedTimes, exp_data_values_der_smooth)

    temp = {}
    temp["peak"] = util.find_peak(selectedTimes, exp_data_values_smooth)[0]

    ret = slice_front(
        selectedTimes, exp_data_values_smooth, exp_data_values_der_smooth, feature
    )
    (
        selectedTimes,
        exp_data_values_smooth,
        exp_data_values_der_smooth,
        min_value,
        max_value,
    ) = ret

    decay = feature.get("decay", 0)

    if decay:
        temp["time_function"] = score.time_function_decay(feature["time"][-1])
    else:
        temp["time_function"] = score.time_function(feature["time"][-1], 0.1 * CV_time)

    temp["value_function"] = score.value_function(temp["peak"][1], abstol)
    temp["value_function_high"] = score.value_function(
        high[1], numpy.abs(high[1]) / 1000
    )
    temp["peak_max"] = max(selectedValues)
    temp["smoothing_factor"] = s
    temp["critical_frequency"] = crit_fs
    temp["critical_frequency_der"] = crit_fs_der
    temp["new_times"] = selectedTimes
    temp["exp_data_values_smooth"] = exp_data_values_smooth
    temp["exp_data_values_der_smooth"] = exp_data_values_der_smooth
    temp["min_value"] = min_value
    temp["max_value"] = max_value
    return temp


def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature["name"])
    derivative = feature.get("derivative", 1)
    temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name]

    if derivative:
        temp.extend(["%s_Derivative_Similarity" % name, "%s_Der_High_Value" % name])

    return temp
