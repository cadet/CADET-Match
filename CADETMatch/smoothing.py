import numpy
import numpy.linalg
import scipy
import scipy.signal
import CADETMatch.util as util
import warnings

import multiprocessing

from cadet import H5
import CADETMatch.ga_simple as ga_simple

butter_order = 3

def refine_butter(times, values, x, y, fs, start):
    x = numpy.array(x)
    y = numpy.array(y)

    y_min = y - min(y)

    p3 = numpy.array([x, y]).T
    p1 = p3[0,:]
    p2 = p3[-1,:]
    
    def goal(crit_fs):
        crit_fs = 10.0**crit_fs[0]
        try:
            sos = scipy.signal.butter(butter_order, crit_fs, btype='lowpass', analog=False, fs=fs, output="sos")
        except ValueError:
            return 1e6
        low_passed = scipy.signal.sosfiltfilt(sos, values)
        
        sse = numpy.sum( (low_passed - values)**2 )
        
        pT = numpy.array([crit_fs, numpy.log(sse)]).T

        d = numpy.cross(p2-p1,p1-pT)/numpy.linalg.norm(p2-p1)
        
        return d
    
    start = numpy.log10(start)
    lb = numpy.log10(x[-1])
    ub = numpy.log10(x[0])
    diff = min(start - lb, ub-start)
    
    lb = start - 0.2 * diff
    ub = start + 0.2 * diff

    result_evo = scipy.optimize.differential_evolution(goal, ((lb, ub),), polish=False)
    
    crit_fs = 10**result_evo.x[0]
    
    return crit_fs

def refine_smooth(times, values, x, y, start):
    x = numpy.array(x)
    y = numpy.array(y)

    y_min = y - min(y)

    p3 = numpy.array([x, y_min]).T
    p1 = p3[0,:]
    p2 = p3[-1,:]
    
    def goal(s):
        s = 10**s[0]
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:
                spline = scipy.interpolate.UnivariateSpline(times, values, s=s, k=5, ext=3)
            except Warning:
                multiprocessing.get_logger().info("caught a warning for %s %s", name, s)
                return 1e6
        
        pT = numpy.array([s, len(spline.get_knots())]).T

        d = numpy.cross(p2-p1,p1-pT)/numpy.linalg.norm(p2-p1)
        
        return d
    
    start = numpy.log10(start)
    lb = numpy.log10(x[-1])
    ub = numpy.log10(x[0])
    diff = min(start - lb, ub-start)
    
    lb = start - 0.2 * diff
    ub = start + 0.2 * diff

    result_evo = scipy.optimize.differential_evolution(goal, ((lb, ub),), polish=False)
    
    s = 10**result_evo.x[0]
    
    spline = scipy.interpolate.UnivariateSpline(times, values, s=s, k=5, ext=3)
    
    return s, len(spline.get_knots())

def find_butter(times, values):
    filters = []
    sse = []
    
    fs = 1.0/(times[1] - times[0])

    for i in numpy.logspace(2, -4, 50):
        try:
            sos = scipy.signal.butter(butter_order, i, btype='lowpass', analog=False, fs=fs, output="sos")
        except ValueError:
            continue
        low_passed = scipy.signal.sosfiltfilt(sos, values)

        filters.append(i)
        sse.append(  numpy.sum( (low_passed - values)**2 ) )
        
    L_x, L_y = util.find_Left_L(filters, numpy.log(sse))

    if L_x is not None:
        L_x = refine_butter(times, values, filters, numpy.log(sse), fs, L_x)
  
    return L_x

def smoothing_filter_butter(times, values, crit_fs):
    if crit_fs is None:
        return values
    fs = 1.0/(times[1] - times[0])

    sos = scipy.signal.butter(butter_order, crit_fs, btype='lowpass', analog=False, fs=fs, output="sos")
    low_passed = scipy.signal.sosfiltfilt(sos, values)
    return low_passed

def load_data(name, cache):
    crit_fs = None
    s = None
    s_knots = 0

    #quick abort
    if name is None or cache is None:
        return s, crit_fs

    factor_file = cache.settings['resultsDirMisc'] / "find_smoothing_factor.h5"

    data = H5()
    data.filename = factor_file.as_posix()

    if factor_file.exists():
        data.load()

    if name in data.root:
        s = float(data.root[name].s)

        crit_fs = data.root[name].crit_fs
        if crit_fs == -1.0:
            crit_fs = None

        s_knots = int(data.root[name].s_knots)
    else:
        return s, crit_fs

    if crit_fs is None:
        multiprocessing.get_logger().info("loaded smoothing_factor %s  %.3e  critical frequency disable", name, s)
    else:
        multiprocessing.get_logger().info("loaded smoothing_factor %s  %.3e  critical frequency %.3e  knots %d", name, s, crit_fs, s_knots)

    return s, crit_fs


def find_smoothing_factors(times, values, name, cache):
    min = 1e-2

    s, crit_fs = load_data(name, cache)

    if s is not None:
        return s, crit_fs

    #normalize the data
    values = values * 1.0/max(values)
    
    crit_fs = find_butter(times, values)

    if crit_fs is None:
        multiprocessing.get_logger().info("%s butter filter disabled, no viable L point found", name)
    
    values_filter = smoothing_filter_butter(times, values, crit_fs)    

    spline = scipy.interpolate.UnivariateSpline(times, values_filter, s=min, k=5, ext=3)
    knots = []
    all_s = []

    knots.append(len(spline.get_knots()))
    all_s.append(min)

    #This limits to 1e-14 max smoothness which is way beyond anything normal
    for i in range(1,200):  
        s = min/(1.1**i)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:
                spline = scipy.interpolate.UnivariateSpline(times, values_filter, s=s, k=5, ext=3)
                knots.append(len(spline.get_knots()))
                all_s.append(s)

                if len(spline.get_knots()) > 600:
                    break

            except Warning:
                multiprocessing.get_logger().info("caught a warning for %s %s", name, s)
                break
    
    knots = numpy.array(knots)
    all_s = numpy.array(all_s)
    
    s, s_knots = util.find_Left_L(all_s,knots)

    s, s_knots = refine_smooth(times, values_filter, all_s, knots, s)

    record_smoothing(s, s_knots, crit_fs, knots, all_s, name, cache)
    
    return s, crit_fs

def record_smoothing(s, s_knots, crit_fs, knots, all_s, name=None, cache=None):
    if name is None or cache is None:
        return
    factor_file = cache.settings['resultsDirMisc'] / "find_smoothing_factor.h5"

    data = H5()
    data.filename = factor_file.as_posix()

    if factor_file.exists():
        data.load()

    if name not in data.root:
        data.root[name].knots = knots
        data.root[name].all_s = all_s
        data.root[name].s = float(s)
        data.root[name].s_knots = int(s_knots)
        if crit_fs is None:
            data.root[name].crit_fs = -1.0
        else:
            data.root[name].crit_fs = float(crit_fs)
        data.save()

    if crit_fs is None:
        multiprocessing.get_logger().info("smoothing_factor %s  %.3e  critical frequency disable", name, s)
    else:
        multiprocessing.get_logger().info("smoothing_factor %s  %.3e  critical frequency %.3e  knots %d", name, s, crit_fs, s_knots)

def create_spline(times, values, crit_fs, s):
    factor = 1.0/max(values)
    values = values * factor
    values_filter = smoothing_filter_butter(times, values, crit_fs)

    return scipy.interpolate.UnivariateSpline(times, values_filter, s=s, k=5, ext=3), factor

def smooth_data(times, values, crit_fs, s):
    spline, factor = create_spline(times, values, crit_fs, s)
    
    return spline(times) / factor

def smooth_data_derivative(times, values, crit_fs, s, smooth=True):
    spline, factor = create_spline(times, values, crit_fs, s)
    
    #run a quick butter pass to remove high frequency noise in the derivative (needed for some experimental data)
    values_filter = spline.derivative()(times) / factor
    if smooth:
        values_filter = butter(times, values_filter)
    return values_filter

def butter(times, values):
    factor = 1.0/max(values)
    values = values * factor

    crit_fs = find_butter(times, values)
    
    values_filter = smoothing_filter_butter(times, values, crit_fs) / factor
    
    return values_filter
