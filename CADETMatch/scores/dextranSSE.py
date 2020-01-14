import CADETMatch.util as util
import CADETMatch.score as score
import scipy.stats
import numpy
import numpy.linalg
from addict import Dict
import sys
import CADETMatch.smoothing as smoothing
import multiprocessing

name = "DextranSSE"
settings = Dict()
settings.adaptive = False
settings.badScore = -sys.float_info.max
settings.meta_mask = True
settings.count = 1
settings.failure = [0.0] * settings.count, 1e6, 1, numpy.array([0.0]), numpy.array([0.0]), numpy.array([1e6]), [1.0] * settings.count

def run(sim_data, feature):
    "special score designed for dextran. This looks at only the front side of the peak up to the maximum slope and pins a value at the elbow in addition to the top"
    exp_time_values = feature['time']
    max_value = feature['max_value']

    selected = feature['selected']
        
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)

    if max(sim_data_values) < max_value: #the system has no point higher than the value we are looking for
        #remove hard failure
        max_value = max(sim_data_values)

    exp_time_values = exp_time_values[selected]
    exp_data_zero = feature['exp_data_zero']

    sim_data_values_smooth = smoothing.smooth_data(sim_time_values, sim_data_values, 
                                                    feature['critical_frequency'], feature['smoothing_factor'])
    sim_data_zero = cut_zero(sim_time_values, sim_data_values_smooth, 1e-3*max_value, max_value)

    sse = util.sse(sim_data_zero, exp_data_zero)

    data = [-sse,], sse, len(sim_data_zero), sim_time_values, sim_data_zero, exp_data_zero, [sse,]

    return data

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    temp = {}
    #change the stop point to be where the max positive slope is along the searched interval
    name = '%s_%s' % (sim.root.experiment_name,   feature['name'])
    s, crit_fs = smoothing.find_smoothing_factors(selectedTimes, selectedValues, name, cache)
    values = smoothing.smooth_data_derivative(selectedTimes, selectedValues, crit_fs, s)
    
    smooth_value = smoothing.smooth_data(selectedTimes, selectedValues, crit_fs, s)
    
    max_index = numpy.argmax(values)
    max_time = selectedTimes[max_index]
    max_value = smooth_value[max_index]

    min_index = numpy.argmax(smooth_value >= 1e-3*max_value)
    min_time = selectedTimes[min_index]
    min_value = smooth_value[min_index]    

    exp_data_zero = cut_zero(selectedTimes, smooth_value, 1e-3*max_value, max_value)

    multiprocessing.get_logger().info("Dextran %s  start: %s   stop: %s  max value: %s", name, 
                                      min_time, max_time, max_value)

    temp['min_time'] = feature['start']
    temp['max_time'] = feature['stop']
    temp['max_value'] = max_value
    temp['exp_data_zero'] = exp_data_zero
    temp['offsetTimeFunction'] = score.time_function_decay_cv(CV_time, selectedTimes, max_time)
    temp['peak_max'] = max_value
    temp['smoothing_factor'] = s
    temp['critical_frequency'] = crit_fs
    temp['smooth_value'] = smooth_value
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_SSE" % name,]
    return temp

def cut_zero(times, values, min_value, max_value):
    "cut the raw times and values as close to min_value and max_value as possible and set the rest to zero without smoothing"
    data_zero = numpy.zeros(len(times))
    
    offset = numpy.array([-1, 0, 1])
    
    #find the starting point for the min_index and max_index, the real value could be off by 1 in either direction
    peak_max_index = numpy.argmax(values)
    
    max_index = numpy.argmax(values[:peak_max_index] >= max_value)

    if not max_index:
        #if max_index is zero the whole array is below the value we are searching for so just returnt the whole array
        return values

    min_index = numpy.argmax(values[:max_index] >= min_value)
        
    check_max_index = max_index + offset
    check_max_value = values[check_max_index]
    check_max_diff = numpy.abs(check_max_value - max_value)    
    
    check_min_index = min_index + offset
    check_min_value = values[check_min_index]
    check_min_diff = numpy.abs(check_min_value - min_value)
    
    min_index = min_index + offset[numpy.argmin(check_min_diff)]
    max_index = max_index + offset[numpy.argmin(check_max_diff)]
    
    data_zero[min_index:max_index+1] = values[min_index:max_index+1]
    
    return data_zero





