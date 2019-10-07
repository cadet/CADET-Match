import CADETMatch.util as util
import CADETMatch.score as score
import numpy
import pandas
import scipy.optimize
from addict import Dict

name = "fractionationSlide"
settings = Dict()
settings.adaptive = True
settings.badScore = 0
settings.meta_mask = True
settings.count = None


def goal(offset, frac_exp, sim_data_time, sim_data_value, start, stop):
    sim_data_value = score.roll_spline(sim_data_time, sim_data_value, offset)
    frac_sim = util.fractionate(start, stop, sim_data_time, sim_data_value)
    return numpy.sum((frac_exp-frac_sim)**2)

def searchRange(times, start_frac, stop_frac, CV_time):
    collectionStart = min(start_frac)
    collectionStop = max(stop_frac)

    searchStart = collectionStart - CV_time
    searchStop = collectionStop + CV_time

    searchStart = max(searchStart, times[0])
    searchStop = min(searchStop, times[-1])

    searchIndexStart = numpy.argmax( times[times <= searchStart])
    searchIndexStop = numpy.argmax( times[times <= searchStop])
    return searchIndexStart, searchIndexStop

def run(sim_data, feature):
    simulation = sim_data['simulation']
    timeFunc = feature['timeFunc']
    components = feature['components']
    numComponents = len(components)
    samplesPerComponent = feature['samplesPerComponent']
    data = feature['data']
    CV_time = feature['CV_time']
    start = feature['start']
    stop = feature['stop']
    funcs = feature['funcs']

    time_center = (start + stop)/2.0

    times = simulation.root.output.solution.solution_times

    scores = []

    sim_values_sse = []
    exp_values_sse = []
   
    graph_sim = {}
    graph_exp = {}

    searchIndexStart, searchIndexStop = searchRange(times, start, stop, CV_time)

    for component, value_func in funcs:
        exp_values = numpy.array(data[str(component)])
        sim_value = simulation.root.output.solution[feature['unit']]["solution_outlet_comp_%03d" % component]

        bounds = find_bounds(times, sim_value)
        result = scipy.optimize.differential_evolution(goal, bounds = [bounds,], 
                                                       args = (exp_values, times, sim_value, start, stop))

        time_offset = abs(result.x[0])
        sim_data_value = score.roll_spline(times, sim_value, result.x[0])

        fracOffset = util.fractionate(start, stop, times, sim_data_value)

        value_score = value_func(max(fracOffset))
        pear = score.pear_corr(scipy.stats.pearsonr(exp_values, fracOffset)[0])
        time_score = timeFunc(time_offset)

        exp_values_sse.extend(exp_values)
        sim_values_sse.extend(fracOffset)

        scores.append(pear)
        scores.append(time_score)
        scores.append(value_score)

        graph_sim[component] = list(zip(time_center, fracOffset))
        graph_exp[component] = list(zip(time_center, exp_values))

    sim_data['graph_exp'] = graph_exp
    sim_data['graph_sim'] = graph_sim

    return (scores, util.sse(numpy.array(sim_values_sse), numpy.array(exp_values_sse)), len(sim_values_sse), 
        time_center, numpy.array(sim_values_sse), numpy.array(exp_values_sse), [1.0 - i for i in scores])

def find_bounds(times, values):
    "find the maximum amount left and right the system can be rolled based on 10% of peak max"
    peak_max = max(values)
    cutoff = 0.1 * peak_max

    data = (values > cutoff).nonzero()

    min_index = data[0][0]
    max_index = data[0][-1]

    return [-min_index, len(values) - max_index - 1]

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    data = pandas.read_csv(feature['fraction_csv'])
    rows, cols = data.shape

    headers = data.columns.values.tolist()

    start = numpy.array(data.iloc[:, 0])
    stop = numpy.array(data.iloc[:, 1])

    smallestTime = min(start - stop)
    abstolFraction = abstol * smallestTime

    funcs = []

    for idx, component in enumerate(headers[2:], 2):
        value = numpy.array(data.iloc[:, idx])
        funcs.append((int(component), score.value_function(max(value), abstolFraction)))

    settings.count = 3 * len(funcs)

    temp['data'] = data
    temp['start'] = start
    temp['stop'] = stop
    temp['timeFunc'] = score.time_function(CV_time, 0, diff_input = True)
    temp['components'] = [int(i) for i in headers[2:]]
    temp['samplesPerComponent'] = rows
    temp['CV_time'] = CV_time
    temp['funcs'] = funcs
    temp['unit'] = feature['unit_name']
    temp['peak_max'] = data.iloc[:,2:].max().min()
    return temp

def headers(experimentName, feature):
    data = pandas.read_csv(feature['fraction_csv'])
    rows, cols = data.shape

    data_headers = data.columns.values.tolist()

    temp = []
    for component in data_headers[2:]:
        temp.append('%s_%s_Component_%s_Similarity' % (experimentName, feature['name'], component))
        temp.append('%s_%s_Component_%s_Time' % (experimentName, feature['name'], component))
        temp.append('%s_%s_Component_%s_Value' % (experimentName, feature['name'], component))
    return temp


