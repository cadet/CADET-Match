import util
import score
import numpy

name = "width"
adaptive = True
badScore = 0

def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = feature['value'][selected]
    exp_time_values = feature['time'][selected]

    sim_width_25 = find_width(sim_time_values, sim_data_values, 0.25)
    sim_width_50 = find_width(sim_time_values, sim_data_values, 0.50)
    sim_width_75 = find_width(sim_time_values, sim_data_values, 0.75)
    
    temp = [feature['width_25'](sim_width_25), 
            feature['width_50'](sim_width_50), 
            feature['width_75'](sim_width_75),]
    return temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), sim_data_values - exp_data_values, [1.0 - i for i in temp]

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    temp['width_25'] = score.value_function(find_width(selectedTimes, selectedValues, 0.25), abstol)
    temp['width_50'] = score.value_function(find_width(selectedTimes, selectedValues, 0.50), abstol)
    temp['width_75'] = score.value_function(find_width(selectedTimes, selectedValues, 0.75), abstol)
    return temp

def find_width(times, values, percent):
    idx_max = numpy.argmax(values)

    max_value = values[idx_max]
    max_time = times[idx_max]
    
    idx_upper = numpy.argmax(values[idx_max:] <= (max_value * percent))
    
    idx_lower = numpy.argmin(values[:idx_max] <= (max_value * percent))
    
    diff_time = times[idx_max + idx_upper] - times[idx_lower]
    
    return diff_time

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_width_25" % name, "%s_width_50" % name, "%s_width_75" % name]
    return temp


