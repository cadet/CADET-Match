import scipy.stats
import numpy
import scipy.optimize
import scipy.interpolate
import scipy.signal
import numpy.linalg

def logistic(x, a, b):
    return  1.0-1.0/(1.0+numpy.exp(a*(x-b)))

def exponential(x, a, b):
    return a * scipy.exp(b*x)

def linear(x, a, b):
    return a*x+b

def cross_correlate(exp_time_values, sim_data_values, exp_data_values):
    corr = scipy.signal.correlate(exp_data_values, sim_data_values)/(numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values))

    index = numpy.argmax(corr)

    score = corr[index]

    endTime = exp_time_values[-1]

    startIndex = -numpy.abs(len(exp_time_values) - index)
    startTime = exp_time_values[startIndex]
    diff_time = endTime - startTime

    return score, diff_time

def pearson(exp_time_values, sim_data_values, exp_data_values):
    corr = scipy.signal.correlate(exp_data_values, sim_data_values)/(numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values))

    index = numpy.argmax(corr)

    score = corr[index]

    endTime = exp_time_values[-1]

    startIndex = -numpy.abs(len(exp_time_values) - index)
    startTime = exp_time_values[startIndex]
    diff_time = endTime - startTime
    diff_index = len(exp_time_values) - index - 1

    sim_data_values_copy = numpy.copy(sim_data_values)
    sim_data_values_copy = numpy.roll(sim_data_values_copy, -diff_index)

    pear = scipy.stats.pearsonr(exp_data_values, sim_data_values)

    return pear_corr(pear[0]), diff_time

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
