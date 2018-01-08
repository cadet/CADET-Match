import util
import score
import scipy.stats
import numpy

name = "dextranHybrid"
adaptive = True
badScore = 0

def run(sim_data,  feature):
    "special score designed for dextran. This looks at only the front side of the peak up to the maximum slope and pins a value at the elbow in addition to the top"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_time_values = feature['time'][selected]
    exp_data_values = feature['value'][selected]

    score_corr, diff_time = score.cross_correlate(exp_time_values, sim_data_values, exp_data_values)

    try:
        sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=util.smoothing_factor(sim_data_values)).derivative(1)
        exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=util.smoothing_factor(exp_data_values)).derivative(1)
    except:  #I know a bare exception is bad but it looks like the exception is not exposed inside UnivariateSpline
        return [0.0, 0.0, 0.0], 1e6

    exp_der_data_values = exp_spline(exp_time_values)
    sim_der_data_values = sim_spline(exp_time_values)

    return [score.pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0]), 
            score.pear_corr(scipy.stats.pearsonr(sim_der_data_values, exp_der_data_values)[0]), 
            feature['offsetTimeFunction'](diff_time)], util.sse(sim_data_values, exp_data_values)

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
    temp['offsetTimeFunction'] = score.time_function_decay(CV_time/10.0, max_time, diff_input=True)
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_Front_Similarity" % name, "%s_Derivative_Similarity" % name, "%s_Time" % name]
    return temp


