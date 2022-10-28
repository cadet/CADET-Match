import multiprocessing
import warnings

import numpy
import numpy.linalg
import scipy
import scipy.signal
from cadet import H5

import CADETMatch.util as util

from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_reference_directions
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize

butter_order = 3

class TargetProblem(ElementwiseProblem):

    def __init__(self, lb, ub, sse_target, func, values, fs):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=lb, xu=ub)
        self.sse_target = sse_target
        self.func = func
        self.values = values
        self.fs = fs

    def _evaluate(self, crit_fs, out, *args, **kwargs):
        crit_fs = 10**crit_fs
        try:
            sos = self.func(crit_fs, self.fs)
            low_passed = scipy.signal.sosfiltfilt(sos, self.values)
            sse = numpy.sum((low_passed - self.values) ** 2)

            error = (sse - self.sse_target)**2
        except ValueError:
            error = numpy.inf
        out["F"] = error

class MaxDistance(ElementwiseProblem):

    def __init__(self, lb, ub, func, fs, values, x_min, y_min, p1, p2, factor):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=lb, xu=ub)
        self.func = func
        self.fs = fs
        self.values = values
        self.x_min = x_min
        self.y_min = y_min
        self.p1 = p1
        self.p2 = p2
        self.factor = factor

    def _evaluate(self, crit_fs, out, *args, **kwargs):
        crit_fs = 10.0 ** crit_fs[0]
        try:
            sos = self.func(crit_fs, self.fs)
        except ValueError:
            out['F'] = 1e6
            return

        try:
            low_passed = scipy.signal.sosfiltfilt(sos, self.values)
        except numpy.linalg.LinAlgError:
            out['F'] = 1e6
            return

        sse = numpy.sum((low_passed - self.values) ** 2)

        pT = numpy.array([crit_fs - self.x_min, numpy.log(sse) - self.y_min]).T / self.factor

        d = numpy.cross(self.p2 - self.p1, self.p1 - pT) / numpy.linalg.norm(self.p2 - self.p1)
        out["F"] = -d

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

def signal_bessel(crit_fs, fs):
    return scipy.signal.bessel(butter_order, crit_fs, btype="lowpass", analog=False, fs=fs, output="sos", norm="delay")

def signal_butter(crit_fs, fs):
    return scipy.signal.butter(butter_order, crit_fs, btype="lowpass", analog=False, fs=fs, output="sos")

def refine_signal(func, times, values, x, y, fs, start):
    x, x_min, y, y_min, p1, p2, p3, factor = get_p(x, y)

    lb = numpy.log10(x[0])
    ub = numpy.log10(x[-1])

    problem = MaxDistance(lb, ub, func, fs, values, x_min, y_min, p1, p2, factor)

    algorithm = PatternSearch(n_sample_points=50, eps=1e-13)

    res = minimize(problem,
               algorithm,
               verbose=False,
               seed=1)

    crit_fs = 10**res.X[0]

    return crit_fs


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


def find_signal(func, times, values, sse_target):
    filters = []
    sse = []

    fs = 1.0 / (times[1] - times[0])

    ub = fs / 2.0

    for i in numpy.logspace(-8, numpy.log10(ub), 50):
        try:
            sos = func(i, fs)
            low_passed = scipy.signal.sosfiltfilt(sos, values)

            filters.append(i)
            sse.append(numpy.sum((low_passed - values) ** 2))
        except (ValueError, numpy.linalg.LinAlgError):
            continue

    crit_fs_max = find_max_signal(func, times, values, sse_target, filters, sse)

    L_x, L_y = find_L(filters, numpy.log(sse))

    if L_x is not None:
        L_x = refine_signal(func, times, values, filters, numpy.log(sse), fs, L_x)

    if L_x  is not None and crit_fs_max < L_x:
        L_x = crit_fs_max

    return L_x

def find_max_signal(func, times, values, sse_target, filters, sse):
    fs = 1.0 / (times[1] - times[0])

    filters = numpy.log10(filters)
    problem = TargetProblem(filters[0], filters[-1], sse_target, func, values, fs)

    algorithm = PatternSearch(n_sample_points=50, eps=1e-13)

    res = minimize(problem,
               algorithm,
               verbose=False,
               seed=1)

    crit_fs = 10**res.X[0]

    return crit_fs


def smoothing_filter_signal(func, times, values, crit_fs):
    if crit_fs is None:
        return values
    fs = 1.0 / (times[1] - times[0])
    sos = func(crit_fs, fs)
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


def find_smoothing_factors(times, values, name, cache, rmse_target=1e-4):
    times, values = resample(times, values)
    min = 1e-3

    sse_target = (rmse_target**2.0)*len(values)

    s, crit_fs, crit_fs_der = load_data(name, cache)

    if s is not None:
        return s, crit_fs, crit_fs_der

    # normalize the data
    values = values * 1.0 / max(values)

    crit_fs = find_signal(signal_bessel, times, values, sse_target)

    if crit_fs is None:
        multiprocessing.get_logger().info(
            "%s butter filter disabled, no viable L point found", name
        )

    values_filter = smoothing_filter_signal(signal_bessel, times, values, crit_fs)

    s = sse_target

    spline, factor = create_spline(times, values, crit_fs, s)

    # run a quick butter pass to remove high frequency noise in the derivative (needed for some experimental data)
    values_filter = spline.derivative()(times) / factor
    factor = 1.0/numpy.max(values_filter)
    values_filter = values_filter * factor
    crit_fs_der = find_signal(signal_bessel, times, values_filter, sse_target)

    s_knots = 0
    knots = spline.get_knots()
    all_s = [s, s]

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
    factor = 1.0 / numpy.max(values)
    values = values * factor
    values_filter = smoothing_filter_signal(signal_bessel, times, values, crit_fs)

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

        factor_der = 1.0/numpy.max(values_filter_der)
        values_filter_der = values_filter_der * factor_der
        values_filter_der = butter(times_resample, values_filter_der, crit_fs_der)
        values_filter_der = values_filter_der / factor_der
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

        factor_der = 1.0/numpy.max(values_filter_der)
        values_filter_der = values_filter_der * factor_der
        values_filter_der = butter(times_resample, values_filter_der, crit_fs_der)
        values_filter_der = values_filter_der / factor_der
        spline_der = scipy.interpolate.InterpolatedUnivariateSpline(
            times_resample, values_filter_der, k=5, ext=3
        )
        values_filter_der = spline_der(times)
    else:
        values_filter_der = spline.derivative()(times) / factor
    return values_filter, values_filter_der


def butter(times, values, crit_fs_der):
    max_value = numpy.max(values)
    values = values / max_value

    values_filter = smoothing_filter_signal(signal_bessel, times, values, crit_fs_der) * max_value

    return values_filter


def resample(times, values, max_samples=5000):
    if len(times) > max_samples:
        times_resample = numpy.linspace(times[0], times[-1], max_samples)
        spline_resample = scipy.interpolate.InterpolatedUnivariateSpline(times, values, k=5, ext=3)
        values_resample = spline_resample(times_resample)

        return times_resample, values_resample
    else:
        diff_times = times[1:] - times[:-1]
        max_time = numpy.max(diff_times)
        min_time = numpy.min(diff_times)
        per = (max_time - min_time) / min_time

        if per > 0.01:
            # time step is not consistent, resample the time steps to a uniform grid based on the smallest time step size seen
            #but not more samples than max_samples
            times_resample = numpy.arange(times[0], times[-1], min_time)

            if len(times_resample) > max_samples:
               times_resample = numpy.linspace(times[0], times[-1], max_samples)

            times_resample[-1] = times[-1]
            spline_resample = scipy.interpolate.InterpolatedUnivariateSpline(times, values, k=5, ext=3)
            values_resample = spline_resample(times_resample)

            return times_resample, values_resample
        else:
            return times, values
