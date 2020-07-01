import CADETMatch.util as util
import CADETMatch.score as score
import scipy.stats
import scipy.interpolate
import numpy
from addict import Dict
import CADETMatch.smoothing as smoothing

name = "ShapeDecay"

def get_settings(feature):
    settings = Dict()
    settings.adaptive = True
    settings.badScore = 0
    settings.meta_mask = True
    settings.count = 6
    return settings

def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]
    exp_data_values_spline = feature['exp_data_values_spline']

    sim_data_values_smooth, sim_data_values_der_smooth = smoothing.full_smooth(exp_time_values, sim_data_values, feature['critical_frequency'], 
                                                              feature['smoothing_factor'], feature['critical_frequency_der'])

    [high, low] = util.find_peak(exp_time_values, sim_data_values_smooth)

    time_high, value_high = high

    pearson, diff_time = score.pearson_spline(exp_time_values, sim_data_values_smooth, feature['smooth_value'])

    pearson_der, diff_time_der = score.pearson_spline(exp_time_values, sim_data_values_der_smooth, exp_data_values_spline)

    [highs_der, lows_der] = util.find_peak(exp_time_values, sim_data_values_der_smooth)

    
    temp = [pearson, 
            feature['value_function'](value_high), 
            feature['time_function'](numpy.abs(diff_time)),
            pearson_der,
            feature['value_function_high'](highs_der[1]),             
            feature['value_function_low'](lows_der[1]),]
    return (temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), 
            sim_time_values, sim_data_values, exp_data_values, [1.0 - i for i in temp])

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    name = '%s_%s' % (sim.root.experiment_name,   feature['name'])
    s, crit_fs, crit_fs_der = smoothing.find_smoothing_factors(selectedTimes, selectedValues, name, cache)
    exp_data_values_smooth, exp_data_values_der_smooth = smoothing.full_smooth(selectedTimes, selectedValues, crit_fs, s, crit_fs_der)
    
    [high, low] = util.find_peak(selectedTimes, exp_data_values_der_smooth)

    temp = {}
    temp['peak'] = util.find_peak(selectedTimes, selectedValues)[0]    
    temp['time_function'] = score.time_function_decay(feature['time'][-1])
    temp['value_function'] = score.value_function(temp['peak'][1], abstol)
    temp['value_function_high'] = score.value_function(high[1], numpy.abs(high[1])/1000)
    temp['value_function_low'] = score.value_function(low[1], numpy.abs(low[1])/1000)
    temp['peak_max'] = max(selectedValues)
    temp['smoothing_factor'] = s
    temp['critical_frequency'] = crit_fs
    temp['critical_frequency_der'] = crit_fs_der
    temp['smooth_value'] = exp_data_values_smooth
    temp['exp_data_values_spline'] = exp_data_values_der_smooth
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name,
            "%s_Derivative_Similarity" % name,
            "%s_Der_High_Value" % name, "%s_Der_Low_Value" % name]
    return temp


