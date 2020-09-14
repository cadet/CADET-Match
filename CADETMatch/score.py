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


def root_poly(corr, len_times, dt):
    index = numpy.argmax(corr)

    dt_pre = (index - len_times + 1) * dt

    if index == 0:
        indexes = numpy.array([index, index+1, index+2])
        multiprocessing.get_logger().warn("min index encountered in root poly  index: %s  dt: %s", index, dt_pre)
    elif index == (len(corr)-1):
        indexes = numpy.array([index-2, index-1, index])
        multiprocessing.get_logger().warn("max index encountered in root poly  index: %s  dt: %s", index, dt_pre)
    else:
        indexes = numpy.array([index-1, index, index+1])
        
    x = (indexes - len_times + 1) * dt
    y = corr[indexes]
    
    poly, res = numpy.polynomial.Polynomial.fit(x,y,2, full=True)
    
    dt = poly.deriv().roots()[0]
    return dt

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

    dt = root_poly(corr, len(times), dt)
    
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

    dt = root_poly(corr, len(times), dt)
    
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

    # find the starting point for the min_index and max_index, the real value could be off by 1 in either direction
    peak_max_index = numpy.argmax(values)

    if values[peak_max_index] < max_value:
        # the whole array is below the max_value we are looking for
        return values, None, None, None, None

    max_index = numpy.argmin((values[:peak_max_index] - max_value)**2)
    min_index = numpy.argmin((values[:max_index] - min_value)**2)

    min_time_found = times[min_index]
    min_value_found = values[min_index]

    max_time_found = times[max_index]
    max_value_found = values[max_index]

    data_zero[min_index : max_index + 1] = values[min_index : max_index + 1]

    return data_zero, min_time_found, min_value_found, max_time_found, max_value_found


def find_cuts(times, values, spline, spline_der):
    max_index = numpy.argmax(values)
    max_time = times[max_index]

    def goal(time):
        return -float(spline_der(time))

    time_space = numpy.linspace(0, max_time, int(max_time)*10)
    goal_space = -(spline_der(time_space))

    idx = numpy.argmin(goal_space)

    indexes = numpy.array([idx-1, idx, idx+1])

    lb, guess, ub = time_space[indexes]

    result = scipy.optimize.minimize(goal, guess, method="powell", bounds=[(lb, ub),])

    max_time = float(result.x[0])
    max_value = float(spline(max_time))

    max_target_time = find_target(spline_der, 1, times, values)

    max_index = numpy.argmin((values - max_value)**2)

    min_time = find_target(spline, 1e-3 * max_value, times[:max_index], values[:max_index])
    min_value = float(spline(min_time))

    return min_time, min_value, max_time, max_value

def find_target(spline, target, times, values, rate=10):
    max_index = numpy.argmax(values)
    max_time = times[max_index]
    
    test_times = numpy.linspace(0, max_time, int(max_time)*rate)

    if not len(test_times):
        return None

    test_values = spline(test_times)
    
    error = (test_values - target)**2
    idx = numpy.argmin(error) 
    min_idx = 0
    max_idx = len(test_times) -1
    
    lb = test_times[max(idx-1, min_idx)]

    guess = test_times[idx]

    ub = test_times[min(idx+1, max_idx)]
    
    def goal(time):
        sse = float((spline(time) - target) ** 2)
        return sse

    result = scipy.optimize.minimize(goal, guess, method="powell", tol=1e-5, bounds=[(lb,ub),])

    found_time = float(result.x[0])
    found_value = spline(found_time)

    if result.success is False:
        multiprocessing.get_logger().info("target %s time %s value %s lb %s guess %s ub %s", target, found_time, 
                                          found_value, lb, guess, ub)
        return None    
    
    return found_time