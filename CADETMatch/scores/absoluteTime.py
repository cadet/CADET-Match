import CADETMatch.util as util
import CADETMatch.calc_coeff as calc_coeff
import numpy
from addict import Dict

name = "AbsoluteTime"
settings = Dict()
settings.adaptive = True
settings.badScore = 0
settings.meta_mask = False
settings.count = 1

"""This score is NOT for optimization. It is needed for the MCMC algorithm in order to handle assymetric distributions"""

def run(sim_data, feature):
    "similarity, value, start stop"
    selected = feature['selected']
    exp_data_values = feature['value'][selected]
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    
    ys = [0.0, 1.0]
    xs = [0.0, sim_time_values[numpy.argmax(exp_data_values)]]

    a, b = calc_coeff.linear_coeff(xs[0], ys[0], xs[1], ys[1])
    
    value = calc_coeff.linear(sim_time_values[numpy.argmax(sim_data_values)], a, b)

    temp = [value,]

    return (temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), 
            sim_time_values, sim_data_values, exp_data_values, [1.0 - i for i in temp])

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    return {}

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_Time" % name]
    return temp


