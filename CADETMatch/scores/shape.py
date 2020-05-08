import CADETMatch.util as util
import CADETMatch.score as score
import scipy.stats
import scipy.interpolate
import numpy
from addict import Dict
import CADETMatch.smoothing as smoothing

name = "Shape"

def get_settings(feature):
    settings = Dict()
    settings.adaptive = True
    settings.badScore = 0
    settings.meta_mask = True

    derivative = feature.get('derivative', 1)

    if derivative:
        settings.count = 6
    else:
        settings.count = 3
    return settings

def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]
    exp_data_values_spline = feature['exp_data_values_spline']

    sim_data_values_spline = smoothing.smooth_data_derivative(exp_time_values, sim_data_values, feature['critical_frequency'], 
                                                              feature['smoothing_factor'], feature['critical_frequency_der'])
     
    [high, low] = util.find_peak(exp_time_values, sim_data_values)

    time_high, value_high = high

    pearson, diff_time = score.pearson_spline(exp_time_values, sim_data_values, feature['smooth_value'])

    derivative = feature.get('derivative', 1)

    if derivative:
        pearson_der, diff_time_der = score.pearson_spline(exp_time_values, sim_data_values_spline, exp_data_values_spline)
        [highs_der, lows_der] = util.find_peak(exp_time_values, sim_data_values_spline)

    
    temp = [pearson, 
            feature['value_function'](value_high), 
            feature['time_function'](numpy.abs(diff_time))]
    
    if derivative:
        temp.extend([pearson_der, feature['value_function_high'](highs_der[1]),             
            feature['value_function_low'](lows_der[1]),])

    return (temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), 
            sim_time_values, sim_data_values, exp_data_values, [1.0 - i for i in temp])

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    name = '%s_%s' % (sim.root.experiment_name,   feature['name'])
    s, crit_fs, crit_fs_der = smoothing.find_smoothing_factors(selectedTimes, selectedValues, name, cache)
    values = smoothing.smooth_data_derivative(selectedTimes, selectedValues, crit_fs, s, crit_fs_der)

    [high, low] = util.find_peak(selectedTimes, values)

    temp = {}
    temp['peak'] = util.find_peak(selectedTimes, selectedValues)[0]
    
    decay = feature.get('decay', 0)

    if decay:
        temp['time_function'] = score.time_function_decay_cv(CV_time, selectedTimes, temp['peak'][0])
    else:
        temp['time_function'] = score.time_function_cv(CV_time, selectedTimes, temp['peak'][0])

    temp['value_function'] = score.value_function(temp['peak'][1], abstol)
    temp['value_function_high'] = score.value_function(high[1], numpy.abs(high[1])/1000)
    temp['value_function_low'] = score.value_function(low[1], numpy.abs(low[1])/1000)
    temp['peak_max'] = max(selectedValues)
    temp['smoothing_factor'] = s
    temp['critical_frequency'] = crit_fs
    temp['critical_frequency_der'] = crit_fs_der
    temp['smooth_value'] = smoothing.smooth_data(selectedTimes, selectedValues, crit_fs, s)
    temp['exp_data_values_spline'] = values
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    derivative = feature.get('derivative', 1)
    temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name]
    
    if derivative:
        temp.extend(["%s_Derivative_Similarity" % name,
            "%s_Der_High_Value" % name, "%s_Der_Low_Value" % name])
    return temp


