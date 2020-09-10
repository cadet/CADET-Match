import CADETMatch.util as util
import CADETMatch.calc_coeff as calc_coeff
import CADETMatch.score as score
import numpy
from addict import Dict
import scipy

name = "AbsoluteTime"


def get_settings(feature):
    settings = Dict()
    settings.adaptive = True
    settings.badScore = 0
    settings.meta_mask = False
    settings.count = 1
    return settings


"""This score is NOT for optimization. It is needed for the MCMC algorithm in order to handle assymetric distributions"""


def run(sim_data, feature):
    "similarity, value, start stop"
    selected = feature["selected"]
    exp_data_values = feature["value"][selected]
    sim_time_values, sim_data_values = util.get_times_values(sim_data["simulation"], feature)

    ys = [0.0, 1.0]
    xs = [0.0, feature["exp_time"]]

    a, b = calc_coeff.linear_coeff(xs[0], ys[0], xs[1], ys[1])

    sim_time = find_time(sim_time_values, sim_data_values)

    value = calc_coeff.linear(sim_time, a, b)

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
    exp_time = find_time(selectedTimes, selectedValues)

    temp = {}
    temp["exp_time"] = exp_time
    return temp


def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature["name"])
    temp = ["%s_Time" % name]
    return temp


def find_time(times, values):
    spline = scipy.interpolate.InterpolatedUnivariateSpline(times, values, ext=3)

    arg_max = numpy.argmax(values)
    max_value = values[arg_max]
    search_value = 0.9 * max_value

    time = score.find_target(spline, search_value, times[:arg_max], values[:arg_max])

    return time
