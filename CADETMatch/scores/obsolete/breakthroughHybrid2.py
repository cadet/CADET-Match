import CADETMatch.util as util
import CADETMatch.score as score
import scipy.stats
import scipy.interpolate
from addict import Dict

name = "breakthroughHybrid2"
settings = Dict()
settings.adaptive = True
settings.badScore = 0
settings.meta_mask = True
settings.count = 7
settings.failure = [0.0] * settings.count, 1e6, 1, [], [1.0] * settings.count

def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)

    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]

    [start, stop] = util.find_breakthrough(exp_time_values, sim_data_values)

    pearson, diff_time = score.pearson(exp_time_values, sim_data_values, exp_data_values)

    sim_spline = util.create_spline(exp_time_values, sim_data_values).derivative(1)
    exp_spline = util.create_spline(exp_time_values, exp_data_values).derivative(1)
    
    exp_data_values_der = exp_spline(exp_time_values)
    sim_data_values_der = sim_spline(exp_time_values)

    pearson_der, diff_time_der = score.pearson(exp_time_values, sim_data_values_der, exp_data_values_der)

    [highs, lows] = util.find_peak(exp_time_values, sim_data_values_der)
    
    temp = [pearson, 
            feature['value_function'](start[1]), 
            feature['time_function'](diff_time),
            pearson_der,
            feature['offsetDerTimeFunction'](diff_time_der),
            feature['value_function_high'](highs[1]),
            feature['value_function_low'](lows[1]),]
    return (temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), 
            sim_time_values, sim_data_values, exp_data_values, [1.0 - i for i in temp])

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    
    exp_spline = util.create_spline(selectedTimes, selectedValues).derivative(1)

    [high, low] = util.find_peak(selectedTimes, exp_spline(selectedTimes))


    temp = {}
    temp['break'] = util.find_breakthrough(selectedTimes, selectedValues)
    temp['time_function'] = score.time_function(CV_time, None, diff_input=True)
    temp['value_function'] = score.value_function(temp['break'][0][1], abstol)
    temp['offsetDerTimeFunction'] = score.time_function_decay(CV_time, None, diff_input=True)
    temp['value_function_high'] = score.value_function(high[1], abstol, 0.1)
    temp['value_function_low'] = score.value_function(low[1], abstol, 0.1)
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name,
            "%s_Derivative_Similarity" % name, "%s_DerTime" % name,
            "%s_Der_High_Value" % name, "%s_Der_Low_Value" % name]
    return temp

