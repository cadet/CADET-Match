import util
import score
import scipy.stats
import scipy.interpolate

name = "derivative_similarity"
adaptive = True
badScore = 0

def run(sim_data, feature):
    "Order is Pearson, Value High, Time High, Value Low, Time Low"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]

    try:
        sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=util.smoothing_factor(sim_data_values)).derivative(1)
        exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=util.smoothing_factor(exp_data_values)).derivative(1)
    except:  #I know a bare exception is based but it looks like the exception is not exposed inside UnivariateSpline
        return [0.0, 0.0, 0.0, 0.0, 0.0], 1e6

    exp_data_values = exp_spline(exp_time_values)
    sim_data_values = sim_spline(exp_time_values)

    [highs, lows] = util.find_peak(exp_time_values, sim_spline(exp_time_values))

    return [score.pear_corr(scipy.stats.pearsonr(sim_spline(exp_time_values), exp_spline(exp_time_values))[0]), 
            feature['value_function_high'](highs[1]), 
            feature['time_function_high'](highs[0]),
            feature['value_function_low'](lows[1]), 
            feature['time_function_low'](lows[0]),], util.sse(sim_data_values, exp_data_values)

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, util.smoothing(selectedTimes, selectedValues), s=util.smoothing_factor(selectedValues)).derivative(1)

    [high, low] = util.find_peak(selectedTimes, exp_spline(selectedTimes))

    temp['peak_high'] = high
    temp['peak_low'] = low

    temp['time_function_high'] = score.time_function(CV_time, high[0])
    temp['value_function_high'] = score.value_function(high[1], abstol, 0.1)
    temp['time_function_low'] = score.time_function(CV_time, low[0])
    temp['value_function_low'] = score.value_function(low[1], abstol, 0.1)
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_Derivative_Similarity" % name, "%s_High_Value" % name, "%s_High_Time" % name, "%s_Low_Value" % name, "%s_Low_Time" % name]
    return temp