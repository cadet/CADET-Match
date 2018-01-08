import scipy.stats
import numpy
import scipy.optimize
import scipy.interpolate
import scipy.signal
import numpy.linalg
import util

def logistic(x, a, b):
    return  1.0-1.0/(1.0+numpy.exp(a*(x-b)))

def exponential(x, a, b):
    return a * scipy.exp(b*x)

def linear(x, a, b):
    return a*x+b

def cross_correlate(exp_time_values, sim_data_values, exp_data_values):
    corr = scipy.signal.correlate(sim_data_values, exp_data_values)/(numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values))

    index = numpy.argmax(corr)

    score = corr[index]

    endTime = exp_time_values[-1]

    try:
        if index > len(exp_time_values):
            simTime = exp_time_values[-(index - len(exp_time_values))]
        elif index < len(exp_time_values):
            simTime = exp_time_values[-(len(exp_time_values) - index)]
        else:
            simTime = endTime
    except IndexError:
        #This means the curve has to be moved outside of the time range and so just set it to the end of the range
        simTime = endTime

    diff_time = endTime - simTime
    return score, diff_time

def time_function_decay(CV_time, peak_time, diff_input=False):
    x_exp = numpy.array([0, 10.0*CV_time])
    y_exp = numpy.array([1, 0.5])

    args_exp = scipy.optimize.curve_fit(exponential, x_exp, y_exp, [1, -0.1], method='trf')[0]

    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        value = exponential(diff, *args_exp)


        #value = float(fun(diff))

        #clip values
        #value = min(value, 1.0)
        #value = max(value, 0.0)

        return value

    return wrapper


def time_function(CV_time, peak_time, diff_input=False):
    #x = numpy.array([0, CV_time/2, 2*CV_time, 5*CV_time, 8*CV_time, 12*CV_time])
    #y = numpy.array([1.0, 0.97, 0.5, 0.15, 0.01, 0])
    #fun = scipy.interpolate.UnivariateSpline(x,y, s=1e-6, ext=1)

    #args = scipy.optimize.curve_fit(logistic, x, y, [-0.1,2.0*CV_time])[0]
    x_exp = numpy.array([CV_time/2.0, 10.0*CV_time])
    y_exp = numpy.array([0.97, 0.5])

    x_lin = numpy.array([0, CV_time/2.0])
    y_lin = numpy.array([1, 0.97])

    args_exp = scipy.optimize.curve_fit(exponential, x_exp, y_exp, [1, -0.1], method='trf')[0]
    args_lin = scipy.optimize.curve_fit(linear, x_lin, y_lin, [1, -0.1], method='trf')[0]
    
    #scale = 1.0/logistic(0.0, *args)

    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        if diff < CV_time/2.0:
            value = linear(diff, *args_lin)
        else:
            value = exponential(diff, *args_exp)


        #value = float(fun(diff))

        #clip values
        #value = min(value, 1.0)
        #value = max(value, 0.0)

        return value

    return wrapper

def value_function(peak_height, tolerance=1e-8, bottom_score = 0.01):
    #if the peak height is 0 or less than the tolerance it needs to be treated as a special case to prevent divide by zero problems
    x = numpy.array([0.0, 1.0])
    y = numpy.array([1.0, bottom_score])
    
    args = scipy.optimize.curve_fit(exponential, x, y, [1, -0.1])[0]

    scale = 1.0/exponential(0.0, *args)
    
    if numpy.abs(peak_height) < tolerance:
        print("peak height less than tolerance", tolerance, peak_height)
        def wrapper(x):
            if numpy.abs(x) < tolerance:
                return 1.0
            else:
                diff = numpy.abs(x-tolerance)/numpy.abs(tolerance)
                return exponential(diff, *args) * scale
    else:
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

def scoreDerivativeSimilarity(sim_data,  feature):
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

    return [pear_corr(scipy.stats.pearsonr(sim_spline(exp_time_values), exp_spline(exp_time_values))[0]), 
            feature['value_function_high'](highs[1]), 
            feature['time_function_high'](highs[0]),
            feature['value_function_low'](lows[1]), 
            feature['time_function_low'](lows[0]),], util.sse(sim_data_values, exp_data_values)

def scoreDerivativeSimilarityHybrid(sim_data,  feature):
    "Order is Pearson, Value High, Time High, Value Low, Time Low"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]

    try:
        sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=util.smoothing_factor(sim_data_values)).derivative(1)
        exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=util.smoothing_factor(exp_data_values)).derivative(1)
    except:  #I know a bare exception is based but it looks like the exception is not exposed inside UnivariateSpline
        return [0.0, 0.0, 0.0, 0.0,], 1e6

    exp_data_values = exp_spline(exp_time_values)
    sim_data_values = sim_spline(exp_time_values)

    score, diff_time = cross_correlate(exp_time_values, sim_data_values, exp_data_values)

    [highs, lows] = util.find_peak(exp_time_values, sim_data_values)

    return [pear_corr(scipy.stats.pearsonr(sim_spline(exp_time_values), exp_spline(exp_time_values))[0]),
            feature['time_function'](diff_time),
            feature['value_function_high'](highs[1]),             
            feature['value_function_low'](lows[1]),], util.sse(sim_data_values, exp_data_values)

def scoreDerivativeSimilarityCross(sim_data,  feature):
    "Order is Pearson, Value High, Time High, Value Low, Time Low"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]

    try:
        sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=util.smoothing_factor(sim_data_values)).derivative(1)
        exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=util.smoothing_factor(exp_data_values)).derivative(1)
    except:  #I know a bare exception is based but it looks like the exception is not exposed inside UnivariateSpline
        return [0.0, 0.0, 0.0, 0.0,], 1e6

    exp_data_values = exp_spline(exp_time_values)
    sim_data_values = sim_spline(exp_time_values)

    score, diff_time = cross_correlate(exp_time_values, sim_data_values, exp_data_values)
    
    [highs, lows] = util.find_peak(exp_time_values, sim_data_values)

    return [pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0]), 
            feature['time_function'](diff_time),
            feature['value_function_high'](highs[1]),             
            feature['value_function_low'](lows[1]),], util.sse(sim_data_values, exp_data_values)

def scoreDerivativeSimilarityCrossAlt(sim_data,  feature):
    "Order is Pearson, Value High, Time High, Value Low, Time Low"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]

    try:
        sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=util.smoothing_factor(sim_data_values)).derivative(1)
        exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=util.smoothing_factor(exp_data_values)).derivative(1)
    except:  #I know a bare exception is based but it looks like the exception is not exposed inside UnivariateSpline
        return [0.0, 0.0], 1e6

    exp_data_values = exp_spline(exp_time_values)
    sim_data_values = sim_spline(exp_time_values)

    score, diff_time = cross_correlate(exp_time_values, sim_data_values, exp_data_values)
    
    return [pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0]), 
            feature['time_function'](diff_time),], util.sse(sim_data_values, exp_data_values)

def scoreCurve(sim_data,  feature):
    "Just Pearson score"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]

    return [pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0])], util.sse(sim_data_values, exp_data_values)