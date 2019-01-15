import util
import sys
import numpy

name = "LogSSE"
adaptive = False
badScore = -sys.float_info.max

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
