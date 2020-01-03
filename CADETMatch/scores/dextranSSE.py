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

    diff = feature['value'] - sim_data_values

    if max(sim_data_values) < max_value: #the system has no point higher than the value we are looking for
        #remove hard failure
        max_value = max(sim_data_values)

    exp_time_values = exp_time_values[selected]
    exp_data_zero = feature['exp_data_zero']

    min_index = numpy.argmax(sim_data_values >= 1e-3*max_value)
    max_index = numpy.argmax(sim_data_values >= max_value)

    sim_data_zero = numpy.zeros(len(sim_data_values))
    sim_data_zero[min_index:max_index+1] = sim_data_values[min_index:max_index+1]

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

    exp_data_zero = numpy.zeros(len(smooth_value))
    exp_data_zero[min_index:max_index+1] = smooth_value[min_index:max_index+1]

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







