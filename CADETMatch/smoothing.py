import multiprocessing
import warnings

import numpy
import numpy.linalg
import scipy
import scipy.signal
from cadet import H5

import CADETMatch.util as util

butter_order = 3


def get_p(x, y):
    x = numpy.array(x)
    y = numpy.array(y)

    sort_idx = numpy.argsort(x)

    x = x[sort_idx]
    y = y[sort_idx]

    y_min = y - min(y)
    x_min = x - min(x)

    p3 = numpy.array([x_min, y_min]).T
    factor = numpy.max(p3, 0)
    p3 = p3 / factor
    p1 = p3[0, :]
    p2 = p3[-1, :]

    return x, min(x), y, min(y), p1, p2, p3, factor


def refine_butter(times, values, x, y, fs, start):
    x, x_min, y, y_min, p1, p2, p3, factor = get_p(x, y)

    def goal(crit_fs):
        crit_fs = 10.0 ** crit_fs[0]
        try:
            sos = scipy.signal.butter(
                butter_order,
                crit_fs,
                btype="lowpass",
                analog=False,
                fs=fs,
                output="sos",
            )
        except ValueError:
            return 1e6
        low_passed = scipy.signal.sosfiltfilt(sos, values)

        sse = numpy.sum((low_passed - values) ** 2)

        pT = numpy.array([crit_fs - x_min, numpy.log(sse) - y_min]).T / factor

        d = numpy.cross(p2 - p1, p1 - pT) / numpy.linalg.norm(p2 - p1)

        return -d

    lb = numpy.log10(x[0])
    ub = numpy.log10(x[-1])

    crit_fs_sample = numpy.linspace(lb, ub, 50)
    errors = numpy.array([goal([i]) for i in crit_fs_sample])

    root, fs_x, fs_y = util.find_opt_poly(crit_fs_sample, errors, numpy.argmin(errors))

    result = scipy.optimize.minimize(
        goal,
        root,
        method="powell",
        bounds=[
            (fs_x[0], fs_x[-1]),
        ],
    )
    crit_fs = 10 ** result.x[0]

    return crit_fs


def refine_smooth(times, values, x, y, start, name):
    x, x_min, y, y_min, p1, p2, p3, factor = get_p(x, y)
    if name is None:
        name = "unknown"

    def goal(s):
        s = 10 ** s[0]
        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            try:
                spline = scipy.interpolate.UnivariateSpline(
                    times, values, s=s, k=5, ext=3
                )
            except Warning:
                multiprocessing.get_logger().info("caught a warning for %s %s", name, s)
                return 1e6

        pT = numpy.array([s - x_min, len(spline.get_knots()) - y_min]).T / factor
        d = numpy.cross(p2 - p1, p1 - pT) / numpy.linalg.norm(p2 - p1)

        return -d

    lb = numpy.log10(x[0])
    ub = numpy.log10(x[-1])

    smoothing_sample = numpy.linspace(lb, ub, 50)
    errors = numpy.array([goal([i]) for i in smoothing_sample])

    if numpy.any(numpy.isnan(errors)):
        #all points are equal distant to our min point, changing smoothness does not change number of knots
        #this means we should use the lowest smoothness possible to have the least error possible with no impact on knots
        s = 10 ** lb
    else:
        root, fs_x, fs_y = util.find_opt_poly(
            smoothing_sample, errors, numpy.argmin(errors)
        )

        result = scipy.optimize.minimize(
            goal,
            root,
            method="powell",
            bounds=[
                (fs_x[0], fs_x[-1]),
            ],
        )
        s = 10 ** result.x[0]

    spline = scipy.interpolate.UnivariateSpline(times, values, s=s, k=5, ext=3)

    return s, len(spline.get_knots())


def find_L(x, y):
    # find the largest value greater than 0, otherwise return none to just turn off butter filter
    x, x_min, y, y_min, p1, p2, p3, factor = get_p(x, y)

    d = numpy.cross(p2 - p1, p1 - p3) / numpy.linalg.norm(p2 - p1)

    max_idx = numpy.argmax(d)
    max_d = d[max_idx]
    l_x = x[max_idx]
    l_y = y[max_idx]

    if max_d <= 0:
        return None, None

    return l_x, l_y


def find_butter(times, values):
    filters = []
    sse = []

    fs = 1.0 / (times[1] - times[0])

    ub = fs / 2.0
    ub_l = numpy.log10(ub)

    for i in numpy.logspace(-6, ub_l, 50):
        try:
            sos = scipy.signal.butter(
                butter_order, i, btype="lowpass", analog=False, fs=fs, output="sos"
            )
            low_passed = scipy.signal.sosfiltfilt(sos, values)

            filters.append(i)
            sse.append(numpy.sum((low_passed - values) ** 2))
        except ValueError:
            continue

    L_x, L_y = find_L(filters, numpy.log(sse))

    if L_x is not None:
        L_x = refine_butter(times, values, filters, numpy.log(sse), fs, L_x)

    return L_x


def smoothing_filter_butter(times, values, crit_fs):
    if crit_fs is None:
        return values
    fs = 1.0 / (times[1] - times[0])

    sos = scipy.signal.butter(
        butter_order, crit_fs, btype="lowpass", analog=False, fs=fs, output="sos"
    )
    low_passed = scipy.signal.sosfiltfilt(sos, values)
    return low_passed


def load_data(name, cache):
    crit_fs = None
    crit_fs_der = None
    s = None
    s_knots = 0

    # quick abort
    if name is None or cache is None:
        return s, crit_fs, crit_fs_der

    factor_file = cache.settings["resultsDirMisc"] / "find_smoothing_factor.h5"

    data = H5()
    data.filename = factor_file.as_posix()

    if factor_file.exists():
        data.load(lock=True)

    if name in data.root:
        s = float(data.root[name].s)

        crit_fs = data.root[name].crit_fs
        if crit_fs == -1.0:
            crit_fs = None

        crit_fs_der = data.root[name].crit_fs_der
        if crit_fs_der == -1.0:
            crit_fs_der = None

        s_knots = int(data.root[name].s_knots)
    else:
        return s, crit_fs, crit_fs_der

    crit_fs_message = "critical frequency disable"
    if crit_fs is not None:
        crit_fs_message = "critical frequency %.3e" % crit_fs

    crit_fs_der_message = "critical frequency der disable"
    if crit_fs_der is not None:
        crit_fs_der_message = "critical frequency der %.3e" % crit_fs_der

    multiprocessing.get_logger().info(
        "smoothing_factor %s  %.3e  %s  %s knots %d",
        name,
        s,
        crit_fs_message,
        crit_fs_der_message,
        s_knots,
    )

    return s, crit_fs, crit_fs_der


def find_smoothing_factors(times, values, name, cache):
    times, values = resample(times, values)
    min = 1e-2

    s, crit_fs, crit_fs_der = load_data(name, cache)

    if s is not None:
        return s, crit_fs, crit_fs_der

    # normalize the data
    values = values * 1.0 / max(values)

    crit_fs = find_butter(times, values)

    if crit_fs is None:
        multiprocessing.get_logger().info(
            "%s butter filter disabled, no viable L point found", name
        )

    values_filter = smoothing_filter_butter(times, values, crit_fs)

    spline = scipy.interpolate.UnivariateSpline(times, values_filter, s=min, k=5, ext=3)
    knots = []
    all_s = []

    knots.append(len(spline.get_knots()))
    all_s.append(min)

    # This limits to 1e-14 max smoothness which is way beyond anything normal
    for i in range(1, 200):
        s = min / (1.1 ** i)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            try:
                spline = scipy.interpolate.UnivariateSpline(
                    times, values_filter, s=s, k=5, ext=3
                )
                knots.append(len(spline.get_knots()))
                all_s.append(s)

                if len(spline.get_knots()) > 600:
                    break

            except Warning:
                multiprocessing.get_logger().info("caught a warning for %s %s", name, s)
                break

    knots = numpy.array(knots)
    all_s = numpy.array(all_s)

    s, s_knots = find_L(all_s, knots)

    if s is not None:
        s, s_knots = refine_smooth(times, values_filter, all_s, knots, s, name)

    spline, factor = create_spline(times, values, crit_fs, s)

    # run a quick butter pass to remove high frequency noise in the derivative (needed for some experimental data)
    values_filter = spline.derivative()(times) / factor
    crit_fs_der = find_butter(times, values_filter)

    record_smoothing(s, s_knots, crit_fs, crit_fs_der, knots, all_s, name, cache)

    return s, crit_fs, crit_fs_der


def record_smoothing(
    s, s_knots, crit_fs, crit_fs_der, knots, all_s, name=None, cache=None
):
    if name is None or cache is None:
        return
    factor_file = cache.settings["resultsDirMisc"] / "find_smoothing_factor.h5"

    data = H5()
    data.filename = factor_file.as_posix()

    if factor_file.exists():
        data.load(lock=True)

    if name not in data.root:
        data.root[name].knots = knots
        data.root[name].all_s = all_s
        data.root[name].s = float(s)
        data.root[name].s_knots = int(s_knots)
        if crit_fs is None:
            data.root[name].crit_fs = -1.0
        else:
            data.root[name].crit_fs = float(crit_fs)
        if crit_fs_der is None:
            data.root[name].crit_fs_der = -1.0
        else:
            data.root[name].crit_fs_der = float(crit_fs)
        data.save(lock=True)

    crit_fs_message = "critical frequency disable"
    if crit_fs is not None:
        crit_fs_message = "critical frequency %.3e" % crit_fs

    crit_fs_der_message = "critical frequency der disable"
    if crit_fs_der is not None:
        crit_fs_der_message = "critical frequency der %.3e" % crit_fs_der

    multiprocessing.get_logger().info(
        "smoothing_factor %s  %.3e  %s  %s knots %d",
        name,
        s,
        crit_fs_message,
        crit_fs_der_message,
        s_knots,
    )


def create_spline(times, values, crit_fs, s):
    times, values = resample(times, values)
    factor = 1.0 / max(values)
    values = values * factor
    values_filter = smoothing_filter_butter(times, values, crit_fs)

    return (
        scipy.interpolate.UnivariateSpline(times, values_filter, s=s, k=5, ext=3),
        factor,
    )


def smooth_data(times, values, crit_fs, s):
    spline, factor = create_spline(times, values, crit_fs, s)

    return spline(times) / factor


def smooth_data_derivative(times, values, crit_fs, s, crit_fs_der, smooth=True):
    times_resample, values_resample = resample(times, values)
    spline, factor = create_spline(times_resample, values_resample, crit_fs, s)

    if smooth:
        values_filter_der = spline.derivative()(times_resample) / factor
        factor_der = numpy.max(values_filter_der)
        values_filter_der = (
            butter(times_resample, values_filter_der / factor_der, crit_fs_der)
            * factor_der
        )
        spline_der = scipy.interpolate.InterpolatedUnivariateSpline(
            times_resample, values_filter_der, k=5, ext=3
        )
        values_filter_der = spline_der(times)
    else:
        values_filter_der = spline.derivative()(times) / factor
    return values_filter_der


def full_smooth(times, values, crit_fs, s, crit_fs_der, smooth=True):
    # return smooth data derivative of data
    times_resample, values_resample = resample(times, values)
    spline, factor = create_spline(times_resample, values_resample, crit_fs, s)

    values_filter = spline(times) / factor

    # run a quick butter pass to remove high frequency noise in the derivative (needed for some experimental data)

    if smooth:
        values_filter_der = spline.derivative()(times_resample) / factor
        factor_der = numpy.max(values_filter_der)
        values_filter_der = (
            butter(times_resample, values_filter_der / factor_der, crit_fs_der)
            * factor_der
        )
        spline_der = scipy.interpolate.InterpolatedUnivariateSpline(
            times_resample, values_filter_der, k=5, ext=3
        )
        values_filter_der = spline_der(times)
    else:
        values_filter_der = spline.derivative()(times) / factor
    return values_filter, values_filter_der


def butter(times, values, crit_fs_der):
    factor = 1.0 / max(values)
    values = values * factor

    values_filter = smoothing_filter_butter(times, values, crit_fs_der) / factor

    return values_filter


def resample(times, values):
    diff_times = times[1:] - times[:-1]
    max_time = numpy.max(diff_times)
    min_time = numpy.min(diff_times)
    per = (max_time - min_time) / min_time

    if per > 0.01:
        # time step is not consistent, resample the time steps to a uniform grid based on the smallest time step size seen
        times_resample = numpy.arange(times[0], times[-1], min_time)
        times_resample[-1] = times[-1]
        diff_times = times_resample[1:] - times_resample[:-1]
        max_time = numpy.max(diff_times)
        min_time = numpy.min(diff_times)
        per = (max_time - min_time) / min_time

        spline_resample = scipy.interpolate.InterpolatedUnivariateSpline(
            times, values, k=5, ext=3
        )
        values_resample = spline_resample(times_resample)

        return times_resample, values_resample
    else:
        return times, values
