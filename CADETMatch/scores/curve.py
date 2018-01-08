import util
import score
import scipy.stats

name = "curve"

def run(sim_data,  feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]

    return [score.pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0])], util.sse(sim_data_values, exp_data_values)

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    return {}

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp  = ["%s_Similarity" % name]
    return temp

