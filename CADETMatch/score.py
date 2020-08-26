import scipy.stats
import numpy
import scipy.optimize
import scipy.interpolate
import scipy.signal
import numpy.linalg
import CADETMatch.calc_coeff as calc_coeff
import CADETMatch.util as util
import multiprocessing
import sys
import SALib.sample.sobol_sequence
import math


def roll_spline(times, values, shift):
    "this function does approximately what the roll function does except that is used the spline so the shift does not have to be an integer and the points are resampled"

    spline = scipy.interpolate.InterpolatedUnivariateSpline(times, values, ext=3)

    times_new = times - shift

    values_new = spline(times_new)

    return values_new


def pearson_spline(exp_time_values, sim_data_values, exp_data_values):
    # resample to a much smaller time step to get a more precise offset
    dt_approx = 1e-2
    points = math.ceil((exp_time_values[-1] - exp_time_values[0])/dt_approx)
    times = numpy.linspace(exp_time_values[0], exp_time_values[-1], points)
    dt = times[1] - times[0]

    sim_spline = scipy.interpolate.InterpolatedUnivariateSpline(exp_time_values, sim_data_values, ext=3)
    sim_resample = sim_spline(times)
    exp_spline = scipy.interpolate.InterpolatedUnivariateSpline(exp_time_values, exp_data_values, ext=3)
    exp_resample = exp_spline(times)

    corr = scipy.signal.correlate(exp_resample, sim_resample)

    index = numpy.argmax(corr)
    
    dt_pre = (index - len(times) + 1) * dt
    
    indexes = numpy.array([index-1, index, index+1])
    x = (indexes - len(times) + 1) * dt
    y = corr[indexes]
    
    poly, res = numpy.polynomial.Polynomial.fit(x,y,2, full=True)
    
    dt = poly.deriv().roots()[0]
    
    # calculate pearson correlation at the new time
    sim_data_values_copy = sim_spline(exp_time_values - dt)
    try:
        pear = scipy.stats.pearsonr(exp_data_values, sim_data_values_copy)[0]
    except ValueError:
        multiprocessing.get_logger().warn(
            "Pearson correlation failed to do NaN or InF in array  exp_array: [%s]   sim_array: [%s]",
            list(exp_data_values),
            list(sim_data_values_copy),
        )
        pear = 0
    score = pear_corr(pear)

    return score, dt


def pearson_spline_fun(exp_time_values, exp_data_values, sim_spline):
    # resample to a much smaller time step to get a more precise offset
    dt_approx = 1e-2
    points = math.ceil((exp_time_values[-1] - exp_time_values[0])/dt_approx)
    times = numpy.linspace(exp_time_values[0], exp_time_values[-1], points)
    dt = times[1] - times[0]

    sim_resample = sim_spline(times)
    exp_spline = scipy.interpolate.InterpolatedUnivariateSpline(exp_time_values, exp_data_values, ext=3)
    exp_resample = exp_spline(times)

    corr = scipy.signal.correlate(exp_resample, sim_resample)

    index = numpy.argmax(corr)
    
    dt_pre = (index - len(times) + 1) * dt
    
    indexes = numpy.array([index-1, index, index+1])
    x = (indexes - len(times) + 1) * dt
    y = corr[indexes]
    
    poly, res = numpy.polynomial.Polynomial.fit(x,y,2, full=True)
    
    dt = poly.deriv().roots()[0]
    
    # calculate pearson correlation at the new time
    sim_data_values_copy = sim_spline(exp_time_values - dt)
    try:
        pear = scipy.stats.pearsonr(exp_data_values, sim_data_values_copy)[0]
    except ValueError:
        multiprocessing.get_logger().warn(
            "Pearson correlation failed to do NaN or InF in array  exp_array: [%s]   sim_array: [%s]",
            list(exp_data_values),
            list(sim_data_values_copy),
        )
        pear = 0
    score_local = pear_corr(pear)

    return score_local, dt


def time_function_decay(max_time):
    x_exp = numpy.array([0, max_time])
    y_exp = numpy.array([1, 0.0])

    a, b = calc_coeff.linear_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])

    def wrapper(diff):
        value = max(0.0, calc_coeff.linear(diff, a, b))

        return value

    return wrapper


def time_function(max_time, delay=10):
    x_lin_1 = numpy.array([0.0, delay])
    y_lin_1 = numpy.array([1.0, 0.95])

    x_lin_2 = numpy.array([delay, max_time])
    y_lin_2 = numpy.array([0.95, 0.0])

    a1, b1 = calc_coeff.linear_coeff(x_lin_1[0], y_lin_1[0], x_lin_1[1], y_lin_1[1])
    a2, b2 = calc_coeff.linear_coeff(x_lin_2[0], y_lin_2[0], x_lin_2[1], y_lin_2[1])

    def wrapper(diff):
        if diff <= delay:
            value = max(0.0, calc_coeff.linear(diff, a1, b1))
        else:
            value = max(0.0, calc_coeff.linear(diff, a2, b2))
        return value

    return wrapper


def value_function(peak_height, tolerance=1e-8, bottom_score=0.0):
    # if the peak height is 0 or less than the tolerance it needs to be treated as a special case to prevent divide by zero problems
    x = numpy.array([0.0, 1.0])
    y = numpy.array([1.0, bottom_score])

    # a, b = calc_coeff.exponential_coeff(x[0], y[0], x[1], y[1])
    a, b = calc_coeff.linear_coeff(x[0], y[0], x[1], y[1])

    if numpy.abs(peak_height) < tolerance:
        multiprocessing.get_logger().warn("peak height less than tolerance %s %s", tolerance, peak_height)

        def wrapper(x):
            if numpy.abs(x) < tolerance:
                return 1.0
            else:
                diff = numpy.abs(x - tolerance) / numpy.abs(tolerance)
                # return max(0, calc_coeff.exponential(diff, a, b))
                return max(0, calc_coeff.linear(diff, a, b))

    else:

        def wrapper(x):
            diff = numpy.abs(x - peak_height) / numpy.abs(peak_height)
            # return max(0, calc_coeff.exponential(diff, a, b))
            return max(0, calc_coeff.linear(diff, a, b))

    return wrapper


def value_function_exp(peak_height, tolerance=1e-8, bottom_score=0.05):
    # if the peak height is 0 or less than the tolerance it needs to be treated as a special case to prevent divide by zero problems
    x = numpy.array([0.0, 1.0])
    y = numpy.array([1.0, bottom_score])

    a, b = calc_coeff.exponential_coeff(x[0], y[0], x[1], y[1])
    # a, b = calc_coeff.linear_coeff(x[0], y[0], x[1], y[1])

    if numpy.abs(peak_height) < tolerance:
        multiprocessing.get_logger().warn("peak height less than tolerance %s %s", tolerance, peak_height)

        def wrapper(x):
            if numpy.abs(x) < tolerance:
                return 1.0
            else:
                diff = numpy.abs(x - tolerance) / numpy.abs(tolerance)
                return max(0, calc_coeff.exponential(diff, a, b))
                # return max(0, calc_coeff.linear(diff, a, b))

    else:

        def wrapper(x):
            diff = numpy.abs(x - peak_height) / numpy.abs(peak_height)
            return max(0, calc_coeff.exponential(diff, a, b))
            # return max(0, calc_coeff.linear(diff, a, b))

    return wrapper


def slope_function(peak_slope):
    # if the peak height is 0 or less than the tolerance it needs to be treated as a special case to prevent divide by zero problems
    x = numpy.array([0.0, 4.0])
    y = numpy.array([1.0, 0.0])

    a, b = calc_coeff.linear_coeff(x[0], y[0], x[1], y[1])

    def wrapper(x):
        diff = numpy.abs(x - peak_slope) / numpy.abs(peak_slope)
        return max(0, calc_coeff.linear(diff, a, b))

    return wrapper


def pear_corr(cr):
    # handle the case where a nan is returned
    if numpy.isnan(cr):
        return 0.0
    if cr < 0.0:
        return 0.0
    else:
        return cr

    # so far in looking cr has never been negative and the scores mostly just sit in the 0.8 to 0.99999 range
    # I am not even sure if cr could be negative with chromatography (indicating inverse relationship between simulation and experiment)
    # return (1.0+cr)/2.0


def cut_zero(times, values, min_value, max_value):
    "cut the raw times and values as close to min_value and max_value as possible and set the rest to zero without smoothing"
    data_zero = numpy.zeros(len(times))

    offset = numpy.array([-1, 0, 1])

    # find the starting point for the min_index and max_index, the real value could be off by 1 in either direction
    peak_max_index = numpy.argmax(values)

    max_index = numpy.argmax(values[:peak_max_index] >= max_value)

    if not max_index:
        # if max_index is zero the whole array is below the value we are searching for so just returnt the whole array
        return values, None, None, None, None

    min_index = numpy.argmax(values[:max_index] >= min_value)

    check_max_index = max_index + offset
    check_max_value = values[check_max_index]
    check_max_diff = numpy.abs(check_max_value - max_value)

    check_min_index = min_index + offset
    check_min_value = values[check_min_index]
    check_min_diff = numpy.abs(check_min_value - min_value)

    min_index = min_index + offset[numpy.argmin(check_min_diff)]
    max_index = max_index + offset[numpy.argmin(check_max_diff)]

    min_time = times[min_index]
    min_value = values[min_index]

    max_time = times[max_index]
    max_value = values[max_index]

    data_zero[min_index : max_index + 1] = values[min_index : max_index + 1]

    return data_zero, min_time, min_value, max_time, max_value


def find_cuts(times, values, spline, spline_der):
    max_index = numpy.argmax(values)
    max_time = times[max_index]

    def goal(time):
        return -spline_der(time)

    result = scipy.optimize.minimize(goal, max_time, method="powell")

    max_time = float(result.x[0])
    max_value = float(spline(max_time))

    min_index = numpy.argmax(values >= 1e-3 * max_value)
    min_time = times[min_index]

    def goal(time):
        return abs(spline(time) - 1e-3 * max_value)

    result = scipy.optimize.minimize(goal, min_time, method="powell")

    min_time = float(result.x[0])
    min_value = float(spline(min_time))

    return min_time, min_value, max_time, max_value
