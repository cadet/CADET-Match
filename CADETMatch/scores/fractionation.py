import util
import score
import numpy
import pandas

name = "fractionation"
adaptive = True
badScore = 0

def run(sim_data, feature):
    "similarity, value, start stop"
    simulation = sim_data['simulation']
    funcs = feature['funcs']

    times = simulation.root.output.solution.solution_times

    scores = []
    sim_values = []
    exp_values = []
   
    graph_sim = {}
    graph_exp = {}
    for (start, stop, component, exp_value, func) in funcs:
        selected = (times >= start) & (times <= stop)

        local_times = times[selected]
        local_values = simulation.root.output.solution.unit_001["solution_outlet_comp_%03d" % component][selected]

        sim_value = numpy.trapz(local_values, local_times)

        exp_values.append(exp_value)
        sim_values.append(sim_value)
        scores.append(func(sim_value))

        if component not in graph_sim:
            graph_sim[component] = []
            graph_exp[component] = []

        time_center = (start + stop)/2.0
        graph_sim[component].append( (time_center, sim_value) )
        graph_exp[component].append( (time_center, exp_value) )


    #sort lists
    for key, value in graph_sim.items():
        value.sort()
    for key, value in graph_exp.items():
        value.sort()

    sim_data['graph_exp'] = graph_exp
    sim_data['graph_sim'] = graph_sim
    return scores, util.sse(numpy.array(sim_values), numpy.array(exp_values)), len(sim_values)

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    data = pandas.read_csv(feature['csv'])
    rows, cols = data.shape

    smallestTime = min(data['Stop'] - data['Start'])
    abstolFraction = abstol * smallestTime

    headers = data.columns.values.tolist()

    funcs = []

    for sample in range(rows):
        for component in headers[2:]:
            start = data['Start'][sample]
            stop = data['Stop'][sample]
            value = data[component][sample]
            func = score.value_function(value, abstolFraction)

            funcs.append( (start, stop, int(component), value, func) )
    temp['funcs'] = funcs
    return temp

def headers(experimentName, feature):
    data = pandas.read_csv(feature['csv'])
    rows, cols = data.shape
    #remove first two columns since those are the start and stop times
    cols = cols - 2

    total = rows * cols
    data_headers = data.columns.values.tolist()

    temp = []
    for sample in range(rows):
        for component in data_headers[2:]:
            temp.append('%s_%s_Sample_%s_Component_%s' % (experimentName, feature['name'], sample, component))
    return temp
