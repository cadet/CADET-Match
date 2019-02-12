import util
import sys
import numpy
from addict import Dict

name = "LogSSE"
settings = Dict()
settings.adaptive = False
settings.badScore = -sys.float_info.max
settings.meta_mask = True
settings.count = 1
settings.failure = [0.0] * settings.count, 1e6, 1, [], [1.0] * settings.count

def run(sim_data, feature):
    "log of SSE score, not composable, negative so score is maximized"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_time_values = feature['time'][selected]
    exp_data_values = feature['value'][selected]

    temp = [-numpy.log(util.sse(sim_data_values, exp_data_values)),]

    return temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), sim_data_values - exp_data_values, [-i for i in temp]

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    return {}

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_SSE" % name]
    return temp
