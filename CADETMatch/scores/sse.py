import util
import score
import scipy.stats
import sys

name = "SSE"
adaptive = False
badScore = -sys.float_info.max

def run(sim_data,  feature):
    "sum square error score, this score is NOT composable with other scores, use negative so score is maximized like other scores"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_time_values = feature['time'][selected]
    exp_data_values = feature['value'][selected]

    return [-util.sse(sim_data_values, exp_data_values),], util.sse(sim_data_values, exp_data_values)

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    return {}

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_SSE" % name]
    return temp

