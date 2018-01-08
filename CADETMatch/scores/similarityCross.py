import util
import score
import scipy.stats

name = "similarityCross"
adaptive = True
badScore = 0

def run(sim_data,  feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]

    score_corr, diff_time = score.cross_correlate(exp_time_values, sim_data_values, exp_data_values)

    [high, low] = util.find_peak(exp_time_values, sim_data_values)

    time_high, value_high = high

    temp = [score_corr, feature['value_function'](value_high), feature['time_function'](diff_time)]
    return temp, util.sse(sim_data_values, exp_data_values)

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    temp['peak'] = util.find_peak(selectedTimes, selectedValues)[0]
    temp['time_function'] = score.time_function(CV_time, temp['peak'][0], diff_input = False)
    temp['value_function'] = score.value_function(temp['peak'][1], abstol)
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name]
    return temp



