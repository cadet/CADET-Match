import scipy.stats
import numpy
import scipy.optimize
import scipy.interpolate
import scipy.signal
import numpy.linalg
import CADETMatch.calc_coeff as calc_coeff
import multiprocessing
import sys
import SALib.sample.sobol_sequence

def roll(x, shift):
    if shift > 0:
        temp = numpy.pad(x,(shift,0), mode='constant')
        return temp[:-shift]
    elif shift < 0:
        temp = numpy.pad(x,(0,numpy.abs(shift)), mode='constant')
        return temp[numpy.abs(shift):]
    else:
        return x


def roll_spline(times, values, shift):
    "this function does approximately what the roll function does except that is used the spline so the shift does not have to be an integer and the points are resampled"

    spline = scipy.interpolate.InterpolatedUnivariateSpline(times, values, ext=3)

    times_new = times - shift

    values_new = spline(times_new)

    return values_new


def cross_correlate(exp_time_values, sim_data_values, exp_data_values):
    corr = scipy.signal.correlate(exp_data_values, sim_data_values)/(numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values))

    #need +1 due to how correlate works
    index = numpy.argmax(corr) + 1

    roll_index = index - len(exp_time_values)

    score = corr[index]

    sim_time_values = roll(exp_time_values, shift=int(numpy.ceil(roll_index)))

    diff_time = numpy.abs(exp_time_values[int(len(exp_time_values)/2)] - sim_time_values[int(len(exp_time_values)/2)])

    return score, diff_time

def pearson(exp_time_values, sim_data_values, exp_data_values):
    corr = scipy.signal.correlate(exp_data_values, sim_data_values)/(numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values))

    #need +1 due to how correlate works
    index = numpy.argmax(corr) + 1

    roll_index = index - len(exp_time_values)

    try:
        score = corr[index]
    except IndexError:
        score = 0.0
        multiprocessing.get_logger().warn("Index error in pearson score at index %s out of %s entries", index, len(corr))

    endTime = exp_time_values[-1]

    sim_time_values = roll(exp_time_values, shift=int(numpy.ceil(roll_index)))

    diff_time = numpy.abs(exp_time_values[int(len(exp_time_values)/2)] - sim_time_values[int(len(exp_time_values)/2)])

    sim_data_values_copy = roll(sim_data_values, shift=int(numpy.ceil(roll_index)))

    pear = scipy.stats.pearsonr(exp_data_values, sim_data_values_copy)
    
    return pear_corr(pear[0]), diff_time

def pearson_spline(exp_time_values, sim_data_values, exp_data_values):
    #resample to a much smaller time step to get a more precise offset
    dt = 1e-2
    times = numpy.arange(exp_time_values[0], exp_time_values[-1], dt)
    
    sim_resample = scipy.interpolate.InterpolatedUnivariateSpline(exp_time_values, sim_data_values, ext=3)(times)
    exp_resample = scipy.interpolate.InterpolatedUnivariateSpline(exp_time_values, exp_data_values, ext=3)(times)    
    
    corr = scipy.signal.correlate(exp_resample, sim_resample)  #/(numpy.linalg.norm(sim_resample) * numpy.linalg.norm(exp_resample))

    index = numpy.argmax(corr)
    
    dt = (index - len(times) + 1) * dt
    
    #calculate pearson correlation at the new time
    spline = scipy.interpolate.InterpolatedUnivariateSpline(exp_time_values, sim_data_values, ext=3)
    sim_data_values_copy = spline(exp_time_values - dt)
    pear = scipy.stats.pearsonr(exp_data_values, sim_data_values_copy)[0]
    score = pear_corr(pear)

    return score, dt

def time_function_decay(CV_time, peak_time, diff_input=False):
    x_exp = numpy.array([0, 1.0*CV_time])
    y_exp = numpy.array([1, 0.5])

    #a, b = calc_coeff.exponential_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])

    a, b = calc_coeff.linear_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])
    
    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        #value = max(0.0, calc_coeff.exponential(diff, a, b))
        value = max(0.0, calc_coeff.linear(diff, a, b))

        return value

    return wrapper

def time_function_decay_cv(CV_time, times, peak_time):

    time = numpy.max(numpy.abs(numpy.array([times[-1] - peak_time, peak_time - times[0]])))
    cv_time = time /CV_time
    
    x_exp = numpy.array([0, cv_time])
    y_exp = numpy.array([1, 0.0])

    a, b = calc_coeff.linear_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])
    
    def wrapper(x):
        diff = x/CV_time
        value = max(0.0, calc_coeff.linear(diff, a, b))

        return value

    return wrapper

def time_function_cv(CV_time, times, peak_time):

    time = numpy.max(numpy.abs(numpy.array([times[-1] - peak_time, peak_time - times[0]])))
    cv_time = time /CV_time

    x_lin_1 = numpy.array([0.0, 0.1])
    y_lin_1 = numpy.array([1.0, 0.95])
    
    x_lin_2 = numpy.array([0.1, cv_time])
    y_lin_2 = numpy.array([0.95, 0.0])

    a1, b1 = calc_coeff.linear_coeff(x_lin_1[0], y_lin_1[0], x_lin_1[1], y_lin_1[1])
    a2, b2 = calc_coeff.linear_coeff(x_lin_2[0], y_lin_2[0], x_lin_2[1], y_lin_2[1])
    
    def wrapper(x):
        diff_cv = x/CV_time
        if diff_cv <= 0.1:
            value = max(0.0, calc_coeff.linear(diff_cv, a1, b1))
        else:
            value = max(0.0, calc_coeff.linear(diff_cv, a2, b2))
        return value
    return wrapper


def time_function_decay_exp(CV_time, peak_time, diff_input=False):
    x_exp = numpy.array([0, 1.0*CV_time])
    y_exp = numpy.array([1, 0.5])

    a, b = calc_coeff.exponential_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])

    #a, b = calc_coeff.linear_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])
    
    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        value = max(0.0, calc_coeff.exponential(diff, a, b))
        #value = max(0.0, calc_coeff.linear(diff, a, b))

        return value

    return wrapper

def time_function(CV_time, peak_time, diff_input=False):
    x_lin = numpy.array([0, 4*CV_time])
    #y_lin = numpy.array([1, 0.0])

    y_lin = numpy.array([1, 0.05])

    a, b = calc_coeff.exponential_coeff(x_lin[0], y_lin[0], x_lin[1], y_lin[1])
    #a, b = calc_coeff.linear_coeff(x_lin[0], y_lin[0], x_lin[1], y_lin[1])

    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        #if diff < CV_time/2.0:
        #value = max(0.0, calc_coeff.linear(diff, a, b))
        value = max(0.0, calc_coeff.exponential(diff, a, b))

        return value

    return wrapper

def time_function2(CV_time, peak_time, diff_input=False):
    #x_exp = numpy.array([CV_time/2.0, 10.0*CV_time])
    x_exp = numpy.array([60.0, 2.0*CV_time])
    y_exp = numpy.array([0.97, 0.5])

    #x_lin = numpy.array([0, CV_time/2.0])
    x_lin = numpy.array([0, 60.0])
    y_lin = numpy.array([1, 0.97])

    a_exp, b_exp = calc_coeff.exponential_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])
    a_lin, b_lin = calc_coeff.linear_coeff(x_lin[0], y_lin[0], x_lin[1], y_lin[1])

    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        #if diff < CV_time/2.0:
        if diff < 60.0:
            value = calc_coeff.linear(diff, a_lin, b_lin)
        else:
            value = calc_coeff.exponential(diff, a_exp, b_exp)

        return value

    return wrapper

def value_function(peak_height, tolerance=1e-8, bottom_score = 0.0):
    #if the peak height is 0 or less than the tolerance it needs to be treated as a special case to prevent divide by zero problems
    x = numpy.array([0.0, 1.0])
    y = numpy.array([1.0, bottom_score])

    #a, b = calc_coeff.exponential_coeff(x[0], y[0], x[1], y[1])
    a, b = calc_coeff.linear_coeff(x[0], y[0], x[1], y[1])
    
    if numpy.abs(peak_height) < tolerance:
        multiprocessing.get_logger().warn("peak height less than tolerance %s %s", tolerance, peak_height)
        def wrapper(x):
            if numpy.abs(x) < tolerance:
                return 1.0
            else:
                diff = numpy.abs(x-tolerance)/numpy.abs(tolerance)
                #return max(0, calc_coeff.exponential(diff, a, b))
                return max(0, calc_coeff.linear(diff, a, b))
    else:
        def wrapper(x):
            diff = numpy.abs(x-peak_height)/numpy.abs(peak_height)
            #return max(0, calc_coeff.exponential(diff, a, b))
            return max(0, calc_coeff.linear(diff, a, b))

    return wrapper

def value_function_exp(peak_height, tolerance=1e-8, bottom_score = 0.05):
    #if the peak height is 0 or less than the tolerance it needs to be treated as a special case to prevent divide by zero problems
    x = numpy.array([0.0, 1.0])
    y = numpy.array([1.0, bottom_score])

    a, b = calc_coeff.exponential_coeff(x[0], y[0], x[1], y[1])
    #a, b = calc_coeff.linear_coeff(x[0], y[0], x[1], y[1])
    
    if numpy.abs(peak_height) < tolerance:
        multiprocessing.get_logger().warn("peak height less than tolerance %s %s", tolerance, peak_height)
        def wrapper(x):
            if numpy.abs(x) < tolerance:
                return 1.0
            else:
                diff = numpy.abs(x-tolerance)/numpy.abs(tolerance)
                return max(0, calc_coeff.exponential(diff, a, b))
                #return max(0, calc_coeff.linear(diff, a, b))
    else:
        def wrapper(x):
            diff = numpy.abs(x-peak_height)/numpy.abs(peak_height)
            return max(0, calc_coeff.exponential(diff, a, b))
            #return max(0, calc_coeff.linear(diff, a, b))

    return wrapper

def slope_function(peak_slope):
    #if the peak height is 0 or less than the tolerance it needs to be treated as a special case to prevent divide by zero problems
    x = numpy.array([0.0, 4.0])
    y = numpy.array([1.0, 0.0])

    a, b = calc_coeff.linear_coeff(x[0], y[0], x[1], y[1])
    
    def wrapper(x):
        diff = numpy.abs(x-peak_slope)/numpy.abs(peak_slope)
        return max(0, calc_coeff.linear(diff, a, b))

    return wrapper

def pear_corr(cr):
    #handle the case where a nan is returned
    if numpy.isnan(cr):
        return 0.0
    return (1+cr)/2