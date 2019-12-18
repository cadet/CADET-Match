import CADETMatch.util as util
import CADETMatch.score as score
import scipy.stats
import scipy.interpolate
import numpy
from addict import Dict
import CADETMatch.smoothing as smoothing

#Used to match the front side of a peak (good for breakthrough curves, the drops the derivative lower score)

name = "ShapeFront"
settings = Dict()
settings.adaptive = True
settings.badScore = 0
settings.meta_mask = True
settings.count = 5
settings.failure = [0.0] * settings.count, 1e6, 1, numpy.array([0.0]), numpy.array([0.0]), numpy.array([1e6]), [1.0] * settings.count

def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]

    sim_data_values_spline = smoothing.smooth_data_derivative(exp_time_values, sim_data_values, feature['critical_frequency'], feature['smoothing_factor'])
    exp_data_values_spline = smoothing.smooth_data_derivative(exp_time_values, exp_data_values, feature['critical_frequency'], feature['smoothing_factor'])
     
    [high, low] = util.find_peak(exp_time_values, sim_data_values)

    time_high, value_high = high

    pearson, diff_time = score.pearson_spline(exp_time_values, sim_data_values, feature['smooth_value'])

    pearson_der, diff_time_der = score.pearson_spline(exp_time_values, sim_data_values_spline, exp_data_values_spline)

    [highs_der, lows_der] = util.find_peak(exp_time_values, sim_data_values_spline)

    
    temp = [pearson, 
            feature['value_function'](value_high), 
            feature['time_function'](numpy.abs(diff_time)),
            pearson_der,
            feature['value_function_high'](highs_der[1])]
    return (temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), 
            sim_time_values, sim_data_values, exp_data_values, [1.0 - i for i in temp])

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    name = '%s_%s' % (sim.root.experiment_name,   feature['name'])
    s, crit_fs = smoothing.find_smoothing_factors(selectedTimes, selectedValues, name, cache)
    values = smoothing.smooth_data_derivative(selectedTimes, selectedValues, crit_fs, s)

    [high, low] = util.find_peak(selectedTimes, values)

    temp = {}
    temp['peak'] = util.find_peak(selectedTimes, selectedValues)[0]
    temp['time_function'] = score.time_function_cv(CV_time, selectedTimes, temp['peak'][0])
    temp['value_function'] = score.value_function(temp['peak'][1], abstol)
    temp['value_function_high'] = score.value_function(high[1], numpy.abs(high[1])/1000)
    temp['peak_max'] = max(selectedValues)
    temp['smoothing_factor'] = s
    temp['critical_frequency'] = crit_fs
    temp['smooth_value'] = smoothing.smooth_data(selectedTimes, selectedValues, crit_fs, s)
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name,
            "%s_Derivative_Similarity" % name,
            "%s_Der_High_Value" % name]
    return temp



