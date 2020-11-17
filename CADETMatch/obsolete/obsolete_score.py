def time_function_decay_exp(CV_time, peak_time, diff_input=False):
    x_exp = numpy.array([0, 1.0 * CV_time])
    y_exp = numpy.array([1, 0.5])

    a, b = calc_coeff.exponential_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])

    # a, b = calc_coeff.linear_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])

    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        value = max(0.0, calc_coeff.exponential(diff, a, b))
        # value = max(0.0, calc_coeff.linear(diff, a, b))

        return value

    return wrapper


def time_function(CV_time, peak_time, diff_input=False):
    x_lin = numpy.array([0, 4 * CV_time])
    # y_lin = numpy.array([1, 0.0])

    y_lin = numpy.array([1, 0.05])

    a, b = calc_coeff.exponential_coeff(x_lin[0], y_lin[0], x_lin[1], y_lin[1])
    # a, b = calc_coeff.linear_coeff(x_lin[0], y_lin[0], x_lin[1], y_lin[1])

    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        # if diff < CV_time/2.0:
        # value = max(0.0, calc_coeff.linear(diff, a, b))
        value = max(0.0, calc_coeff.exponential(diff, a, b))

        return value

    return wrapper


def time_function_decay(CV_time, peak_time, diff_input=False):
    x_exp = numpy.array([0, 1.0 * CV_time])
    y_exp = numpy.array([1, 0.5])

    # a, b = calc_coeff.exponential_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])

    a, b = calc_coeff.linear_coeff(x_exp[0], y_exp[0], x_exp[1], y_exp[1])

    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        # value = max(0.0, calc_coeff.exponential(diff, a, b))
        value = max(0.0, calc_coeff.linear(diff, a, b))

        return value

    return wrapper


def cross_correlate(exp_time_values, sim_data_values, exp_data_values):
    corr = scipy.signal.correlate(exp_data_values, sim_data_values) / (
        numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values)
    )

    # need +1 due to how correlate works
    index = numpy.argmax(corr) + 1

    roll_index = index - len(exp_time_values)

    score = corr[index]

    sim_time_values = roll(exp_time_values, shift=int(numpy.ceil(roll_index)))

    diff_time = numpy.abs(
        exp_time_values[int(len(exp_time_values) / 2)]
        - sim_time_values[int(len(exp_time_values) / 2)]
    )

    return score, diff_time


def pearson(exp_time_values, sim_data_values, exp_data_values):
    corr = scipy.signal.correlate(exp_data_values, sim_data_values) / (
        numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values)
    )

    # need +1 due to how correlate works
    index = numpy.argmax(corr) + 1

    roll_index = index - len(exp_time_values)

    try:
        score = corr[index]
    except IndexError:
        score = 0.0
        multiprocessing.get_logger().warn(
            "Index error in pearson score at index %s out of %s entries",
            index,
            len(corr),
        )

    endTime = exp_time_values[-1]

    sim_time_values = roll(exp_time_values, shift=int(numpy.ceil(roll_index)))

    diff_time = numpy.abs(
        exp_time_values[int(len(exp_time_values) / 2)]
        - sim_time_values[int(len(exp_time_values) / 2)]
    )

    sim_data_values_copy = roll(sim_data_values, shift=int(numpy.ceil(roll_index)))

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

    return score, diff_time


def roll(x, shift):
    if shift > 0:
        temp = numpy.pad(x, (shift, 0), mode="constant")
        return temp[:-shift]
    elif shift < 0:
        temp = numpy.pad(x, (0, numpy.abs(shift)), mode="constant")
        return temp[numpy.abs(shift) :]
    else:
        return x
