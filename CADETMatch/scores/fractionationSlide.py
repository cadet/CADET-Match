import util
import score
import numpy
import pandas
import scipy.optimize

name = "fractionationSlide"
adaptive = True
badScore = 0

def goal(offset, frac_exp, sim_data_time, sim_data_value, start, stop, flow):
    sim_data_value = numpy.roll(sim_data_value, int(offset))
    frac_sim = util.fractionate(start, stop, sim_data_time, sim_data_value) * flow
    return numpy.sum((frac_exp-frac_sim)**2)

def searchRange(times, start_frac, stop_frac, CV_time):
    collectionStart = min(start_frac)
    collectionStop = max(stop_frac)

    searchStart = collectionStart - CV_time
    searchStop = collectionStop + CV_time

    searchStart = max(searchStart, times[0])
    searchStop = min(searchStop, times[-1])

    #print(CV_time, collectionStart, collectionStop, searchStart, searchStop)
    searchIndexStart = numpy.argmax( times[times <= searchStart])
    searchIndexStop = numpy.argmax( times[times <= searchStop])
    return searchIndexStart, searchIndexStop

def rollRange(times, sim_value, searchIndexStart, searchIndexStop):
    peakMaxIndex = numpy.argmax(sim_value)
    peakMaxTime = times[peakMaxIndex]
    peakMaxValue = sim_value[peakMaxIndex]

    above1 = sim_value < 0.1 * peakMaxValue
    beforePeak = times < peakMaxTime

    search = sim_value[beforePeak & above1]
    searchMax = numpy.argmax(search)

    rollLeft = searchIndexStart - searchMax
    rollRight = searchIndexStop - searchMax
    return rollLeft, rollRight, searchMax

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
    flow = simulation.root.input.model.connections.switch_000.connections[9]

    scores = []

    sim_values_sse = []
    exp_values_sse = []
   
    graph_sim = {}
    graph_exp = {}

    searchIndexStart, searchIndexStop = searchRange(times, start, stop, CV_time)
    #print("searchIndexStart\t", searchIndexStart, "\tsearchIndexStop\t", searchIndexStop)

    for component, value_func in funcs:
        exp_values = numpy.array(data[str(component)])
        sim_value = simulation.root.output.solution.unit_001["solution_outlet_comp_%03d" % component]

        rollLeft, rollRight, searchMax = rollRange(times, sim_value, searchIndexStart, searchIndexStop)
        #print("rollLeft\t", rollLeft, "\trollRight\t", rollRight, "\tsearchMax\t", searchMax)

        result = scipy.optimize.differential_evolution(goal, bounds = [(0,3001),], 
                                                       args = (exp_values, times, sim_value, start, stop, flow))
        #print(result)

        try:
            time_offset = numpy.abs(times[searchMax] - times[searchMax + int(result.x[0])])
        except IndexError:
            #This covers the case where no working time offset can be found so give the max penalty for this position
            time_offset = times[-1]
        #print("time_offset\t", time_offset)
        sim_data_value = numpy.roll(sim_value, int(result.x[0]))
        fracOffset = util.fractionate(start, stop, times, sim_data_value) * flow

        #score_corr, diff_time = score.cross_correlate(time_center, fracOffset, exp_values)

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

    return scores, util.sse(numpy.array(sim_values_sse), numpy.array(exp_values_sse))

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    data = pandas.read_csv(feature['csv'])
    rows, cols = data.shape

    headers = data.columns.values.tolist()

    start = numpy.array(data.iloc[:, 0])
    stop = numpy.array(data.iloc[:, 1])

    flow = sim.root.input.model.connections.switch_000.connections[9]
    smallestTime = min(start - stop)
    abstolFraction = flow * abstol * smallestTime

    funcs = []

    for idx, component in enumerate(headers[2:], 2):
        value = numpy.array(data.iloc[:, idx])
        funcs.append((int(component), score.value_function(max(value), abstolFraction)))

    temp['data'] = data
    temp['start'] = start
    temp['stop'] = stop
    temp['timeFunc'] = score.time_function(CV_time, 0, diff_input = True)
    temp['components'] = [int(i) for i in headers[2:]]
    temp['samplesPerComponent'] = rows
    temp['CV_time'] = CV_time
    temp['funcs'] = funcs
    return temp

def headers(experimentName, feature):
    data = pandas.read_csv(feature['csv'])
    rows, cols = data.shape

    data_headers = data.columns.values.tolist()

    temp = []
    for component in data_headers[2:]:
        temp.append('%s_%s_Component_%s_Similarity' % (experimentName, feature['name'], component))
        temp.append('%s_%s_Component_%s_Time' % (experimentName, feature['name'], component))
        temp.append('%s_%s_Component_%s_Value' % (experimentName, feature['name'], component))
    return temp


