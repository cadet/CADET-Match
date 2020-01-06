import CADETMatch.util as util
import CADETMatch.score as score
import scipy.stats
import numpy
import numpy.linalg
from addict import Dict
import CADETMatch.smoothing as smoothing
import multiprocessing

name = "DextranShape"
settings = Dict()
settings.adaptive = True
settings.badScore = 0
settings.meta_mask = True
settings.count = 2
settings.failure = [0.0] * settings.count, 1e6, 1, numpy.array([0.0]), numpy.array([0.0]), numpy.array([1e6]), [1.0] * settings.count

def run(sim_data, feature):
    "special score designed for dextran. This looks at only the front side of the peak up to the maximum slope and pins a value at the elbow in addition to the top"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)

    exp_time_zero = feature['exp_time_zero']
    exp_data_zero = feature['exp_data_zero']
    
    sim_data_zero = cut_front(sim_time_values, sim_data_values, exp_time_zero, 
                                             feature['min_value_front'], feature['max_value_front'],
                                             feature['smoothing_factor'], feature['critical_frequency'])
        
    pearson, diff_time = score.pearson_spline(exp_time_zero, exp_data_zero, sim_data_zero)

    exp_data_zero_sse = feature['exp_data_zero_sse']
    sim_data_zero_sse = scipy.interpolate.InterpolatedUnivariateSpline(exp_time_zero, sim_data_zero, ext=1)(sim_time_values)

    temp = [pearson,
            feature['offsetTimeFunction'](numpy.abs(diff_time)),
            ]

    data = (temp, util.sse(sim_data_zero_sse, exp_data_zero_sse), len(sim_data_zero_sse), 
            sim_time_values, sim_data_zero_sse, exp_data_zero_sse, [1.0 - i for i in temp])

    return data

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    temp = {}
    #change the stop point to be where the max positive slope is along the searched interval
    name = '%s_%s' % (sim.root.experiment_name,   feature['name'])
    exp_time_zero, exp_data_zero, min_time, min_value, max_time, max_value, s, crit_fs = cut_front_find(selectedTimes, selectedValues, name, cache)

    multiprocessing.get_logger().info("Dextran %s  start: %s   stop: %s  max value: %s", name, 
                                      min_time, max_time, max_value)

    exp_data_zero_sse = scipy.interpolate.InterpolatedUnivariateSpline(exp_time_zero, exp_data_zero, ext=1)(selectedTimes)

    temp['min_time'] = feature['start']
    temp['max_time'] = feature['stop']
    
    temp['min_time_front'] = min_time
    temp['min_value_front'] = min_value
    temp['max_time_front'] = max_time
    temp['max_value_front'] = max_value

    temp['exp_time_zero'] = exp_time_zero
    temp['exp_data_zero'] = exp_data_zero
    temp['exp_data_zero_sse'] = exp_data_zero_sse
    temp['offsetTimeFunction'] = score.time_function_decay_cv(CV_time, selectedTimes, max_time)
    temp['peak_max'] = max_value
    temp['smoothing_factor'] = s
    temp['critical_frequency'] = crit_fs
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_Shape" % name, 
            "%s_Time" % name,
            ]
    return temp

def cut_front_find(times, values, name, cache):
    s, crit_fs = smoothing.find_smoothing_factors(times, values, name, cache)
    values_der = smoothing.smooth_data_derivative(times, values, crit_fs, s)

    smooth_value = smoothing.smooth_data(times, values, crit_fs, s)
    
    spline_der = scipy.interpolate.InterpolatedUnivariateSpline(times, values_der, ext=1)
    spline = scipy.interpolate.InterpolatedUnivariateSpline(times, smooth_value, ext=1)
    
    max_index = numpy.argmax(values)
    max_time = times[max_index]
    
    def goal(time):
        return -spline_der(time)
    
    result = scipy.optimize.minimize(goal, max_time, method='powell')
    
    max_time = float(result.x)
    max_value = spline(float(result.x))

    min_index = numpy.argmax(smooth_value >= 1e-2*max_value)
    min_time = times[min_index]
    
    def goal(time):
        return abs(spline(time)-1e-2*max_value)
    
    result = scipy.optimize.minimize(goal, min_time, method='powell')
    
    min_time = float(result.x)
    min_value = spline(float(result.x))
    
    #resample to 100 points/second
    needed_points = int( (times[-1] - times[0]) * 100)
    
    new_times = numpy.linspace(times[0], times[-1], needed_points)
    new_values = spline(new_times)
    
    max_index = numpy.argmax(new_values >= max_value)
    min_index = numpy.argmax(new_values >= min_value)

    data_zero = numpy.zeros(needed_points)
    
    data_zero[min_index:max_index+1] = new_values[min_index:max_index+1]
    
    return new_times, data_zero, min_time, min_value, max_time, max_value, s, crit_fs

def cut_front(times, values, new_times, min_value, max_value, s, crit_fs):
    smooth_value = smoothing.smooth_data(times, values, crit_fs, s)

    spline = scipy.interpolate.InterpolatedUnivariateSpline(times, smooth_value, ext=1)
    
    max_index = numpy.argmax(values >= max_value)
    max_time = times[max_index]
    
    def goal(time):
        return abs(spline(time)-max_value)
    
    result = scipy.optimize.minimize(goal, max_time, method='powell')
    
    max_time = float(result.x)
    max_value = spline(float(result.x))

    min_index = numpy.argmax(values >= min_value)
    min_time = times[min_index]
    
    def goal(time):
        return abs(spline(time)-min_value)
    
    result = scipy.optimize.minimize(goal, min_time, method='powell')
    
    min_time = float(result.x)
    min_value = spline(float(result.x))

    new_values = spline(new_times)
    
    max_index = numpy.argmax(new_values >= max_value)
    min_index = numpy.argmax(new_values >= min_value)

    data_zero = numpy.zeros(len(new_times))    
    data_zero[min_index:max_index+1] = new_values[min_index:max_index+1]
    
    return data_zero




