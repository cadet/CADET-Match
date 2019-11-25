import numpy
import numpy.linalg
import scipy
import scipy.signal
import CADETMatch.util as util

import scoop

from cadet import H5

def find_butter(times, values):
    filters = []
    sse = []
    
    fs = 1.0/(times[1] - times[0])

    for i in numpy.logspace(-1, -4, 100):
        b, a = scipy.signal.butter(3, i, btype='lowpass', analog=False, fs=fs)
        low_passed = scipy.signal.filtfilt(b, a, values)

        filters.append(i)
        sse.append(  numpy.sum( (low_passed - values)**2 ) )
        
    L_x, L_y = util.find_L(filters, numpy.log(sse))
    
    return L_x

def smoothing_filter_butter(times, values, crit_fs):
    fs = 1.0/(times[1] - times[0])

    b, a = scipy.signal.butter(3, crit_fs, btype='lowpass', analog=False, fs=fs)
    low_passed = scipy.signal.filtfilt(b, a, values)
    return low_passed

def find_smoothing_factors(times, values, name, cache):
    min = 1e-2

    #normalize the data
    values = values * 1.0/max(values)
    
    crit_fs = find_butter(times, values)
    
    values_filter = smoothing_filter_butter(times, values, crit_fs)    

    spline = scipy.interpolate.UnivariateSpline(times, values_filter, s=min, k=5, ext=3)
    knots = []
    all_s = []

    knots.append(len(spline.get_knots()))
    all_s.append(min)

    while knots[-1] < (knots[0]*10):
        s = all_s[-1]/1.5
        spline = scipy.interpolate.UnivariateSpline(times, values_filter, s=s, k=5, ext=3)
        knots.append(len(spline.get_knots()))
        all_s.append(s)
    
    knots = numpy.array(knots)
    all_s = numpy.array(all_s)
    
    s, s_knots = util.find_L(all_s,knots)

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
        data.root[name].crit_fs = float(crit_fs)
        data.save()

    scoop.logger.info("smoothing_factor %s  %.3e  critical frequency %.3e  knots %d", name, s, crit_fs, s_knots)

def create_spline(times, values, crit_fs, s):
    factor = 1.0/max(values)
    values = values * factor
    values_filter = smoothing_filter_butter(times, values, crit_fs)

    return scipy.interpolate.UnivariateSpline(times, values_filter, s=s, k=5, ext=3), factor

def smooth_data(times, values, crit_fs, s):
    spline, factor = create_spline(times, values, crit_fs, s)
    
    return spline(times) / factor

def smooth_data_derivative(times, values, crit_fs, s):
    spline, factor = create_spline(times, values, crit_fs, s)
    
    return spline.derivative()(times) / factor
