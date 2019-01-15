import util
import score
import scipy.stats
import scipy.interpolate

name = "breakthroughHybrid2"
adaptive = True
badScore = 0

def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)

    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]

    [start, stop] = util.find_breakthrough(exp_time_values, sim_data_values)

    pearson, diff_time = score.pearson(exp_time_values, sim_data_values, exp_data_values)


    try:
        sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=util.smoothing_factor(sim_data_values)).derivative(1)
        exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=util.smoothing_factor(exp_data_values)).derivative(1)
    except:  #I know a bare exception is based but it looks like the exception is not exposed inside UnivariateSpline
        return [0.0] * 7, 1e6, [], [1.0] * 7

    exp_data_values_der = exp_spline(exp_time_values)
    sim_data_values_der = sim_spline(exp_time_values)

    pearson_der, diff_time_der = score.pearson(exp_time_values, sim_data_values_der, exp_data_values_der)

    [highs, lows] = util.find_peak(exp_time_values_der, sim_data_values_der)
    
    temp = [pearson, 
            feature['value_function'](start[1]), 
            feature['time_function'](diff_time),
            pearson_der,
            feature['offsetDerTimeFunction'](diff_time_der),
            feature['value_function_high'](highs[1]),
            feature['value_function_low'](lows[1]),]
    return temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), sim_data_values - exp_data_values, [1.0 - i for i in temp]

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    
    exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, util.smoothing(selectedTimes, selectedValues), s=util.smoothing_factor(selectedValues)).derivative(1)

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

