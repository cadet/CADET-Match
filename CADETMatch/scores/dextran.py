import util
import score
import scipy.stats
import numpy

name = "dextran"
adaptive = True
badScore = 0

def run(sim_data, feature):
    "special score designed for dextran. This looks at only the front side of the peak up to the maximum slope and pins a value at the elbow in addition to the top"
    selected = feature['origSelected']
    max_time = feature['max_time']

    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature, selected)

    exp_time_values = feature['time'][selected]
    exp_data_values = feature['value'][selected]

    try:
        sim_spline_derivative = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=util.smoothing_factor(sim_data_values)).derivative(1)
        exp_spline_derivative = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=util.smoothing_factor(exp_data_values)).derivative(1)
    except:  #I know a bare exception is based but it looks like the exception is not exposed inside UnivariateSpline
        return [0.0, 0.0, 0.0], 1e6

    expSelected = selected & (feature['time'] <= max_time)
    expTime = feature['time'][expSelected]
    expValues = feature['value'][expSelected]
    expDerivValues = exp_spline_derivative(expSelected)

    values = sim_spline_derivative(exp_time_values)
    sim_max_index = numpy.argmax(values)
    sim_max_time = exp_time_values[sim_max_index]

    simSelected = selected & (feature['time'] <= sim_max_time)

    simTime, simValues = util.get_times_values(sim_data['simulation'], feature, simSelected)

    simDerivValues = sim_spline_derivative(simSelected)

    score, diff_time = cross_correlate(expTime, simValues, expValues)
    scoreDeriv, diff_time_deriv = cross_correlate(expTime, simDerivValues, expDerivValues)

    if score < 0:
        score = 0

    if scoreDeriv < 0:
        scoreDeriv = 0

    return [score, scoreDeriv, feature['maxTimeFunction'](diff_time)], util.sse(sim_data_values, exp_data_values)

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    #change the stop point to be where the max positive slope is along the searched interval
    exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, selectedValues, s=util.smoothing_factor(selectedValues), k=1).derivative(1)
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