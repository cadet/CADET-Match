import CADETMatch.util as util
import sys
import numpy
from addict import Dict

name = "LogSSE"

def get_settings(feature):
    settings = Dict()
    settings.adaptive = False
    settings.badScore = -sys.float_info.max
    settings.meta_mask = True
    settings.count = 1
    return settings

def run(sim_data, feature):
    "log of SSE score, not composable, negative so score is maximized"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_time_values = feature['time'][selected]
    exp_data_values = feature['value'][selected]

    temp = [-numpy.log(util.sse(sim_data_values, exp_data_values)),]

    return (temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), 
            sim_time_values, sim_data_values, exp_data_values, [-i for i in temp])

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    temp = {}
    temp['peak_max'] = max(selectedValues)
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_SSE" % name]
    return temp
