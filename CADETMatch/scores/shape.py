import util
import score
import scipy.stats
import scipy.interpolate
import numpy

name = "Shape"
adaptive = True
badScore = 0

def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]

    try:
        sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=util.smoothing_factor(sim_data_values)).derivative(1)
        exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=util.smoothing_factor(exp_data_values)).derivative(1)
    except:  #I know a bare exception is based but it looks like the exception is not exposed inside UnivariateSpline
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1e6, 1, [], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
     
    [high, low] = util.find_peak(exp_time_values, sim_data_values)

    time_high, value_high = high

    pearson, diff_time = score.pearson_spline(exp_time_values, sim_data_values, exp_data_values)

    exp_data_values = exp_spline(exp_time_values)
    sim_data_values = sim_spline(exp_time_values)

    pearson_der, diff_time_der = score.pearson_spline(exp_time_values, sim_data_values, exp_data_values)

    [highs_der, lows_der] = util.find_peak(exp_time_values, sim_data_values)

    
    temp = [pearson, 
            feature['value_function'](value_high), 
            feature['time_function'](numpy.abs(diff_time)),
            pearson_der,
            feature['value_function_high'](highs_der[1]),             
            feature['value_function_low'](lows_der[1]),]
    return temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), sim_data_values - exp_data_values, [1.0 - i for i in temp]

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    
    exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, util.smoothing(selectedTimes, selectedValues), s=util.smoothing_factor(selectedValues)).derivative(1)

    [high, low] = util.find_peak(selectedTimes, exp_spline(selectedTimes))

    temp = {}
    temp['peak'] = util.find_peak(selectedTimes, selectedValues)[0]
    temp['time_function'] = score.time_function(CV_time, temp['peak'][0], diff_input=True)
    temp['value_function'] = score.value_function(temp['peak'][1], abstol)
    temp['value_function_high'] = score.value_function(high[1], abstol, 0.1)
    temp['value_function_low'] = score.value_function(low[1], abstol, 0.1)
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name,
            "%s_Derivative_Similarity" % name,
            "%s_Der_High_Value" % name, "%s_Der_Low_Value" % name]
    return temp


