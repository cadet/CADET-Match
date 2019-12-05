import numpy
import numpy.linalg
import scipy
import scipy.signal
import CADETMatch.util as util
import warnings

import scoop

from cadet import H5

def refine_butter(times, values, x, y, fs, start):
    x = numpy.array(x)
    y = numpy.array(y)

    p3 = numpy.array([x, y]).T
    p1 = p3[0,:]
    p2 = p3[-1,:]
    
    def goal(crit_fs):
        crit_fs = 10.0**crit_fs[0]
        try:
            b, a = scipy.signal.butter(3, crit_fs, btype='lowpass', analog=False, fs=fs)
        except ValueError:
            return 1e6
        low_passed = scipy.signal.filtfilt(b, a, values)
        
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
    
    result_evo = scipy.optimize.differential_evolution(goal, ((lb, ub),) )
    
    crit_fs = 10**result_evo.x[0]
    
    return crit_fs

def refine_smooth(times, values, x, y, start):
    x = numpy.array(x)
    y = numpy.array(y)

    p3 = numpy.array([x, y]).T
    p1 = p3[0,:]
    p2 = p3[-1,:]
    
    def goal(s):
        s = 10**s[0]
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:
                spline = scipy.interpolate.UnivariateSpline(times, values, s=s, k=3, ext=3)
            except Warning:
                print("caught a warning for %s %s", name, s)
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
    
    result_evo = scipy.optimize.differential_evolution(goal, ((lb, ub),) )
    
    s = 10**result_evo.x[0]
    
    spline = scipy.interpolate.UnivariateSpline(times, values, s=s, k=5, ext=3)
    
    return s, len(spline.get_knots())

def find_butter(times, values):
    filters = []
    sse = []
    
    fs = 1.0/(times[1] - times[0])

    for i in numpy.logspace(-1, -4, 100):
        try:
            b, a = scipy.signal.butter(3, i, btype='lowpass', analog=False, fs=fs)
        except ValueError:
            continue
        low_passed = scipy.signal.filtfilt(b, a, values)

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

    b, a = scipy.signal.butter(3, crit_fs, btype='lowpass', analog=False, fs=fs)
    low_passed = scipy.signal.filtfilt(b, a, values)
    return low_passed

def find_smoothing_factors(times, values, name, cache):
    min = 1e-2

    #normalize the data
    values = values * 1.0/max(values)
    
    crit_fs = find_butter(times, values)

    if crit_fs is None:
        scoop.logger.info("%s butter filter disabled, no viable L point found", name)
    
    values_filter = smoothing_filter_butter(times, values, crit_fs)    

    spline = scipy.interpolate.UnivariateSpline(times, values_filter, s=min, k=5, ext=3)
    knots = []
    all_s = []

    knots.append(len(spline.get_knots()))
    all_s.append(min)

    last = 10

    progress = True
    
    while knots[-1] < (knots[0]*10) and progress:
        s = all_s[-1]/1.5
        with warnings.catch_warnings():
            warnings.filterwarnings('error')

            try:
                spline = scipy.interpolate.UnivariateSpline(times, values_filter, s=s, k=5, ext=3)
                knots.append(len(spline.get_knots()))
                all_s.append(s)
            except Warning:
                scoop.logger.info("caught a warning for %s %s", name, s)
                pass

        if len(knots) < (last+1):
            progress = True
        else:
            lower = numpy.array(knots[-last:-1])
            upper = numpy.array(knots[-last+1:])
            progress = numpy.any(lower < upper)
    
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
        scoop.logger.info("smoothing_factor %s  %.3e  critical frequency disable", name, s)
    else:
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
