import scipy.stats
import numpy
import scipy.optimize
import scipy.interpolate
import util

def logistic(x, a, b):
    return  1.0-1.0/(1.0+numpy.exp(a*(x-b)))

def exponential(x,a):
    return numpy.exp(a*x)

def time_function(CV_time, peak_time):
    x = numpy.array([0, CV_time/2.0, 2.0*CV_time, 4.0*CV_time])
    y = numpy.array([1.0, 0.98, 0.5, 0.01])
    
    args = scipy.optimize.curve_fit(logistic, x, y, [-0.1,2.0*CV_time])[0]
    
    scale = 1.0/logistic(0.0, *args)

    def wrapper(x):

        diff = numpy.abs(x - peak_time)
        return logistic(diff, *args)*scale

    return wrapper

def value_function(peak_height):
    x = numpy.array([0.0, 1.0])
    y = numpy.array([1.0, 0.01])
    
    args = scipy.optimize.curve_fit(exponential, x, y, [-5])[0]

    scale = 1.0/exponential(0.0, *args)
    
    def wrapper(x):
        diff = numpy.abs(x-peak_height)/numpy.abs(peak_height)
        return exponential(diff, *args) * scale

    return wrapper

def pear_corr(cr):
    if cr < 0.5:
        out = 1.0/3.0 * cr + 1.0/3.0
    else:
        out = cr
    return out

def scoreSimilarity(sim_data, experimental_data, feature):
    "Order is Pearson, Value, Time"
    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]
    exp_time_values = experimental_data['time'][selected]
    sim_data_values = sim_data['value'][selected]

    [high, low] = util.find_peak(exp_time_values, sim_data_values)

    time_high, value_high = high
    
    sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=1e-4)
    exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=1e-4)

    temp = [pear_corr(scipy.stats.pearsonr(sim_spline(exp_time_values), exp_spline(exp_time_values))[0]), feature['value_function'](value_high), feature['time_function'](time_high)]
    return temp

def scoreDerivativeSimilarity(sim_data, experimental_data, feature):
    "Order is Pearson, Value High, Time High, Value Low, Time Low"
    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]
    exp_time_values = experimental_data['time'][selected]
    sim_data_values = sim_data['value'][selected]

    #spline_exp = scipy.interpolate.splrep(exp_time_values, util.smoothing(exp_time_values, exp_data_values))
    #spline_sim = scipy.interpolate.splrep(exp_time_values, util.smoothing(exp_time_values, sim_data_values))

    sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=1e-4).derivative(1)
    exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=1e-4).derivative(1)

    #spline_derivative_exp = scipy.interpolate.splev(exp_time_values, spline_exp, der=1)
    #spline_derivative_sim = scipy.interpolate.splev(exp_time_values, spline_sim, der=1)

    [highs, lows] = util.find_peak(exp_time_values, sim_spline(exp_time_values))

    return [pear_corr(scipy.stats.pearsonr(sim_spline(exp_time_values), exp_spline(exp_time_values))[0]), 
            feature['value_function_high'](highs[1]), 
            feature['time_function_high'](highs[0]),
            feature['value_function_low'](lows[1]), 
            feature['time_function_low'](lows[0]),]

def scoreCurve(sim_data, experimental_data, feature):
    "Just Pearson score"
    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]
    sim_data_values = sim_data['value'][selected]

    return [pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0])]
