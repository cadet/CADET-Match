import math
import multiprocessing
import sys

import numpy
import numpy.linalg
import SALib.sample.sobol_sequence
import scipy.interpolate
import scipy.optimize
import scipy.signal
import scipy.stats

import CADETMatch.calc_coeff as calc_coeff
import CADETMatch.util as util
import numba


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
        indexes = numpy.array([index, index + 1, index + 2])
        multiprocessing.get_logger().warn(
            "min index encountered in root poly  index: %s  dt: %s", index, dt_pre
        )
    elif index == (len(corr) - 1):
        indexes = numpy.array([index - 2, index - 1, index])
        multiprocessing.get_logger().warn(
            "max index encountered in root poly  index: %s  dt: %s", index, dt_pre
        )
    else:
        indexes = numpy.array([index - 1, index, index + 1])

    x = (indexes - len_times + 1) * dt
    y = corr[indexes]

    poly, res = numpy.polynomial.Polynomial.fit(x, y, 2, full=True)

    try:
        dt = poly.deriv().roots()[0]
    except IndexError:
        # this happens if all y values are the same in which case just take the center value
        multiprocessing.get_logger().warn("root poly index_error x %s  y %s", x, y)
        dt = x[indexes[1]]
    return dt


def pearson_spline(exp_time_values, sim_data_values, exp_data_values):
    # resample to a much smaller time step to get a more precise offset
    sim_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        exp_time_values, sim_data_values, ext=1
    )
    return pearson_spline_fun(exp_time_values, exp_data_values, sim_spline)

@numba.njit(fastmath=True)
def pearsonr_mat(x, Y, times):
    r = numpy.zeros(Y.shape[0])
    xm = x - x.mean()
    
    r_x_den = 0.0
    for i in range(x.shape[0]):
        r_x_den += xm[i]*xm[i]
    r_x_den = numpy.sqrt(r_x_den)

    for i in range(Y.shape[0]):
        ymean = 0.0
        for j in range(Y.shape[1]):
            ymean += Y[i,j]
        ym = Y[i] - ymean/Y.shape[1]

        r_num = 0.0
        r_y_den = 0.0
        for j in range(Y.shape[1]):
            r_num += xm[j]*ym[j]
            r_y_den += ym[j] * ym[j]
        r_y_den = numpy.sqrt(r_y_den)

        if r_y_den == 0.0:
            r[i] = -1.0
        else:
            min_fun = numpy.zeros(x.shape[0])
            for j in range(x.shape[0]):
                min_fun[j] = min(x[j], Y[i,j])

            area = numpy.trapz(min_fun, times)

            r[i] = min(max(r_num/(r_x_den*r_y_den), -1.0), 1.0) * area
    return r

def eval_offsets(offsets, sim_spline, exp_time_values, exp_data_values):
    rol_mat = numpy.zeros([len(offsets), len(exp_data_values)])

    for idx,offset in enumerate(offsets):
        rol_mat[idx,:] = sim_spline(exp_time_values - offset)

    scores = pearsonr_mat(exp_data_values, rol_mat, exp_time_values)
    return scores

def pearson_offset(offset, times, sim_data, exp_data):
    sim_spline = scipy.interpolate.InterpolatedUnivariateSpline(times, sim_data, ext=1)
    sim_data_offset = sim_spline(times - offset)
    try:
        pear = scipy.stats.pearsonr(exp_data, sim_data_offset)[0]
    except ValueError:
        multiprocessing.get_logger().warn(
            "Pearson correlation failed to do NaN or InF in array  exp_array: [%s]   sim_array: [%s]",
            list(exp_data),
            list(sim_data_offset),
        )
        pear = 0
    score_local = pear_corr(pear)
    return score_local


def pearson_spline_fun(
    exp_time_values, exp_data_values, sim_spline, size=100, nest=10, bounds=2, tol=1e-4
):
    for i in range(nest + 1):
        if i == 0:
            lb = -exp_time_values[-1]
            ub = exp_time_values[-1]
            local_size = min(1000 + 1, int((ub - lb) * 2 + 1))
        else:
            idx_max = numpy.argmax(pearson)

            try:
                lb = offsets[idx_max - bounds]
            except IndexError:
                lb = offsets[0]

            try:
                ub = offsets[idx_max + bounds]
            except IndexError:
                ub = offsets[-1]
            local_size = size

        if ub - lb < tol:
            break

        offsets = numpy.linspace(lb, ub, local_size)

        pearson = eval_offsets(offsets, sim_spline, exp_time_values, exp_data_values)

        idx_max = numpy.argmax(pearson)

        expand_lb = max(bounds - idx_max, 0)
        expand_ub = max(bounds - (len(pearson) - 1 - idx_max), 0)

        if expand_lb or expand_ub:
            # need to expand boundaries to handle our new edges
            # if boundaries do have to be expanded make sure to expand by double the amount required since it is only done once
            expand_lb = expand_lb * 2
            expand_ub = expand_ub * 2
            dt = offsets[1] - offsets[0]
            if expand_lb:
                local_offsets = numpy.linspace(
                    offsets[0] - expand_lb * dt, offsets[0] - dt, expand_lb
                )
                local_pearson = eval_offsets(
                    local_offsets, sim_spline, exp_time_values, exp_data_values
                )

                offsets = numpy.concatenate([local_offsets, offsets])
                pearson = numpy.concatenate([local_pearson, pearson])

            if expand_ub:
                local_offsets = numpy.linspace(
                    offsets[-1] + dt, offsets[-1] + expand_ub * dt, expand_ub
                )

                local_pearson = eval_offsets(
                    local_offsets, sim_spline, exp_time_values, exp_data_values
                )

                offsets = numpy.concatenate([offsets, local_offsets])
                pearson = numpy.concatenate([pearson, local_pearson])

    idx = numpy.argmax(pearson)

    dt, time_found, goal_found = util.find_opt_poly(offsets, pearson, idx)

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
        multiprocessing.get_logger().warn(
            "peak height less than tolerance %s %s", tolerance, peak_height
        )

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
        multiprocessing.get_logger().warn(
            "peak height less than tolerance %s %s", tolerance, peak_height
        )

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


def cut_zero(times, values, min_value, max_value):
    "cut the raw times and values as close to min_value and max_value as possible and set the rest to zero without smoothing"
    data_zero = numpy.zeros(len(times))

    # find the starting point for the min_index and max_index, the real value could be off by 1 in either direction
    peak_max_index = numpy.argmax(values)

    if values[peak_max_index] < max_value:
        # the whole array is below the max_value we are looking for
        return values, None, None, None, None

    max_index = numpy.argmin((values[:peak_max_index] - max_value) ** 2)
    min_index = numpy.argmin((values[:max_index] - min_value) ** 2)

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

    time_space = numpy.linspace(0, max_time, int(max_time) * 10)
    goal_space = -(spline_der(time_space))

    idx = numpy.argmin(goal_space)

    guess, time_found, goal_found = util.find_opt_poly(time_space, goal_space, idx)

    lb = time_found[0]
    ub = time_found[-1]

    result = scipy.optimize.minimize(
        goal,
        guess,
        method="powell",
        bounds=[
            (lb, ub),
        ],
    )

    max_time = float(result.x[0])
    max_value = float(spline(max_time))

    max_target_time = find_target(spline_der, 1, times, values)

    max_index = numpy.argmin((values - max_value) ** 2)

    min_time = find_target(
        spline, 1e-3 * max_value, times[:max_index], values[:max_index]
    )
    min_value = float(spline(min_time))

    return min_time, min_value, max_time, max_value


def find_target(spline, target, times, values, rate=10):
    max_index = numpy.argmax(values)
    max_time = times[max_index]

    test_times = numpy.linspace(0, max_time, int(max_time) * rate)

    if not len(test_times):
        return None

    test_values = spline(test_times)

    error = (test_values - target) ** 2
    idx = numpy.argmin(error)

    guess, time_found, goal_found = util.find_opt_poly(test_times, error, idx)

    lb = time_found[0]
    ub = time_found[-1]

    def goal(time):
        sse = float((spline(time) - target) ** 2)
        return sse

    result = scipy.optimize.minimize(
        goal,
        guess,
        method="powell",
        tol=1e-5,
        bounds=[
            (lb, ub),
        ],
    )

    found_time = float(result.x[0])
    found_value = spline(found_time)

    if result.success is False:
        multiprocessing.get_logger().info(
            "target %s time %s value %s lb %s guess %s ub %s",
            target,
            found_time,
            found_value,
            lb,
            guess,
            ub,
        )
        return None

    return found_time
