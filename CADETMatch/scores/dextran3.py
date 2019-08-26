import util
import score
import scipy.stats
import numpy
import numpy.linalg
from addict import Dict
from cadet import H5

name = "Dextran3"
settings = Dict()
settings.adaptive = True
settings.badScore = 0
settings.meta_mask = True
settings.count = 4
settings.failure = [0.0] * settings.count, 1e6, 1, numpy.array([0.0]), numpy.array([0.0]), numpy.array([1e6]), [1.0] * settings.count

def run(sim_data, feature):
    "special score designed for dextran. This looks at only the front side of the peak up to the maximum slope and pins a value at the elbow in addition to the top"
    exp_time_values = feature['time']
    max_value = feature['max_value']

    selected = feature['selected']
        
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)

    diff = feature['value'] - sim_data_values

    sse = numpy.sum(diff)
    norm = numpy.linalg.norm(diff)

    if max(sim_data_values) < max_value: #the system has no point higher than the value we are looking for
        #remove hard failure
        max_value = max(sim_data_values)

    exp_time_values = exp_time_values[selected]
    exp_data_zero = feature['exp_data_zero']

    min_index = numpy.argmax(sim_data_values >= 1e-3*max_value)
    max_index = numpy.argmax(sim_data_values >= max_value)

    sim_data_zero = numpy.zeros(len(sim_data_values))
    sim_data_zero[min_index:max_index+1] = sim_data_values[min_index:max_index+1]

    pearson, diff_time = score.pearson_spline(exp_time_values, sim_data_zero, exp_data_zero)

    p = poly_fit(exp_time_values, sim_data_zero)
    if p is None:
        return settings.failure

    temp = [feature['p0'](p[0]),
            feature['p1'](p[1]),
            feature['p2'](p[2]),
            feature['offsetTimeFunction'](numpy.abs(diff_time)),
            ]

    data = (temp, util.sse(sim_data_zero, exp_data_zero), len(sim_data_zero), 
            sim_time_values, sim_data_zero, exp_data_zero, [1.0 - i for i in temp])

    return data

def poly_fit(times, values):
    max_value = numpy.max(values)
    max_index = numpy.argmax(values >= 0.8*max_value)
    max_time = times[max_index]

    min_index = numpy.argmax(values >= 0.1*max_value)
    min_time = times[min_index]
    min_value = values[min_index]
    
    temp_values = values[min_index:max_index+1]
    temp_times = times[min_index:max_index+1]
    
    p, residuals, rank, singular_values, rcond = numpy.polyfit(temp_times, temp_values, 3, full=True)
    if rank < 4:
        return None
    return p

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    #change the stop point to be where the max positive slope is along the searched interval
    exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, selectedValues, s=util.smoothing_factor(selectedValues)).derivative(1)

    values = exp_spline(selectedTimes)
    
    max_index = numpy.argmax(values)
    max_time = selectedTimes[max_index]
    max_value = selectedValues[max_index]

    min_index = numpy.argmax(selectedValues >= 1e-3*max_value)
    min_time = selectedTimes[min_index]
    min_value = selectedValues[min_index]

    exp_data_zero = numpy.zeros(len(selectedValues))
    exp_data_zero[min_index:max_index+1] = selectedValues[min_index:max_index+1]

    exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, util.smoothing(selectedTimes, exp_data_zero), s=util.smoothing_factor(exp_data_zero)).derivative(1)

    [high, low] = util.find_peak(selectedTimes, exp_spline(selectedTimes))

    p = poly_fit(selectedTimes, exp_data_zero)
                
    temp['min_time'] = feature['start']
    temp['max_time'] = feature['stop']
    temp['max_value'] = max_value
    temp['exp_data_zero'] = exp_data_zero
    temp['offsetTimeFunction'] = score.time_function_decay_cv(CV_time, selectedTimes, max_time)
    temp['p0'] = score.slope_function(p[0])
    temp['p1'] = score.slope_function(p[1])
    temp['p2'] = score.slope_function(p[2])
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_P0" % name, 
            "%s_P1" % name,
            "%s_P2" % name,
            "%s_Time" % name,
            ]
    return temp





