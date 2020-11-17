import numpy
import scipy.stats
from addict import Dict

import CADETMatch.score as score
import CADETMatch.util as util

name = "dextranHybrid"
settings = Dict()
settings.adaptive = True
settings.badScore = 0
settings.meta_mask = True
settings.count = 6
settings.failure = (
    [0.0] * settings.count,
    1e6,
    1,
    numpy.array([0.0]),
    numpy.array([0.0]),
    numpy.array([1e6]),
    [1.0] * settings.count,
)


def run(sim_data, feature):
    "special score designed for dextran. This looks at only the front side of the peak up to the maximum slope and pins a value at the elbow in addition to the top"
    exp_time_values = feature["time"]
    exp_data_values = feature["value"]

    start = feature["min_time"]
    stop = feature["max_time"]

    selected = (exp_time_values >= start) & (exp_time_values <= stop)

    sim_time_values, sim_data_values = util.get_times_values(
        sim_data["simulation"], feature, selected=selected
    )

    lower_index = numpy.argmax(sim_data_values >= 0.05 * max(sim_data_values))
    lower_time = sim_time_values[lower_index]
    lower_value = sim_data_values[lower_index]

    exp_time_values = exp_time_values[selected]
    exp_data_values = exp_data_values[selected]

    score_corr, diff_time = score.cross_correlate(
        exp_time_values, sim_data_values, exp_data_values
    )

    sim_spline = util.create_spline(exp_time_values, sim_data_values).derivative(1)
    exp_spline = util.create_spline(exp_time_values, exp_data_values).derivative(1)

    exp_der_data_values = exp_spline(exp_time_values)
    sim_der_data_values = sim_spline(exp_time_values)

    temp = [
        score.pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0]),
        score.pear_corr(
            scipy.stats.pearsonr(sim_der_data_values, exp_der_data_values)[0]
        ),
        feature["offsetTimeFunction"](diff_time),
        feature["valueFunction"](sim_data_values[-1]),
        feature["offsetTimeLowerFunction"](lower_time),
        feature["valueLowerFunction"](lower_value),
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


def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    # change the stop point to be where the max positive slope is along the searched interval
    exp_spline = util.create_spline(selectedTimes, selectedValues).derivative(1)

    values = exp_spline(selectedTimes)
    max_index = numpy.argmax(values)
    max_time = selectedTimes[max_index]
    max_value = selectedValues[max_index]

    lower_index = numpy.argmax(selectedValues >= 0.05 * max_value)
    lower_time = selectedTimes[lower_index]
    lower_value = selectedValues[lower_index]

    temp["min_time"] = feature["start"]
    temp["max_time"] = max_time
    temp["max_value"] = max_value
    temp["offsetTimeFunction"] = score.time_function_decay(
        CV_time / 10.0, max_time, diff_input=True
    )
    temp["valueFunction"] = score.value_function(max_value, abstol)
    temp["offsetTimeLowerFunction"] = score.time_function_decay(
        CV_time / 10.0, lower_time, diff_input=False
    )
    temp["valueLowerFunction"] = score.value_function(lower_value, abstol)
    return temp


def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature["name"])
    temp = [
        "%s_Front_Similarity" % name,
        "%s_Derivative_Similarity" % name,
        "%s_Time" % name,
        "%s_Value" % name,
        "%s_10P_Time" % name,
        "%s_10P_Value" % name,
    ]
    return temp
