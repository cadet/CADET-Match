import CADETMatch.util as util
import CADETMatch.score as score
import scipy.stats
import numpy
from addict import Dict

name = "dextran_original"
settings = Dict()
settings.adaptive = True
settings.badScore = 0
settings.meta_mask = True
settings.count = 3
settings.failure = [0.0] * settings.count, 1e6, 1, numpy.array([0.0]), numpy.array([0.0]), numpy.array([1e6]), [1.0] * settings.count

def run(sim_data, feature):
    "special score designed for dextran. This looks at only the front side of the peak up to the maximum slope and pins a value at the elbow in addition to the top"
    selected = feature['origSelected']
    max_time = feature['max_time']

    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature, selected)

    exp_time_values = feature['time'][selected]
    exp_data_values = feature['value'][selected]

    sim_spline = util.create_spline(exp_time_values, sim_data_values).derivative(1)
    exp_spline = util.create_spline(exp_time_values, exp_data_values).derivative(1)

    expSelected = selected & (feature['time'] <= max_time)
    expTime = feature['time'][expSelected]
    expValues = feature['value'][expSelected]
    expDerivValues = exp_spline_derivative(expSelected)

    values = sim_spline_derivative(exp_time_values)
    #sim_max_index = numpy.argmax(values)
    #sim_max_time = exp_time_values[sim_max_index]

    simSelected = selected & (feature['time'] <= max_time)

    simTime, simValues = util.get_times_values(sim_data['simulation'], feature, simSelected)

    simDerivValues = sim_spline_derivative(simSelected)

    score_cross, diff_time = score.cross_correlate(expTime, simValues, expValues)
    scoreDeriv, diff_time_deriv = score.cross_correlate(expTime, simDerivValues, expDerivValues)

    if score_cross < 0:
        score_cross = 0

    if scoreDeriv < 0:
        scoreDeriv = 0

    temp = [score_cross, scoreDeriv, feature['maxTimeFunction'](diff_time)]

    return (temp, util.sse(sim_data_values, exp_data_values), len(sim_data_values), 
            sim_time_values, sim_data_values, exp_data_values, [1.0 - i for i in temp])

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    #change the stop point to be where the max positive slope is along the searched interval
    exp_spline = util.create_spline(selectedTimes, selectedValues).derivative(1)
    
    values = exp_spline(selectedTimes)
    max_index = numpy.argmax(values)
    max_time = selectedTimes[max_index]
            
    temp['origSelected'] = temp['selected']
    temp['selected'] = temp['selected'] & (temp['time'] <= max_time)
    temp['max_time'] = max_time
    temp['maxTimeFunction'] = score.time_function_decay(CV_time/10.0, max_time, diff_input=True)
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_Front_Similarity" % name, "%s_Derivative_Similarity" % name, "%s_Time" % name]
    return temp
