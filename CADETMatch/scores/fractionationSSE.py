import util
import score
import numpy
import pandas
from addict import Dict
import sys

"""DO NOT USE THIS SCORE. This score only exists for the purpose of a paper to confirm that
this is not an appropriate method to solve fractionation problems.
"""

name = "fractionationSSE"
settings = Dict()
settings.adaptive = False
settings.badScore = -sys.float_info.max
settings.meta_mask = True
settings.count = 1
settings.failure = [0.0] * settings.count, 1e6, 1, [], [1.0] * settings.count

def run(sim_data, feature):
    "similarity, value, start stop"
    simulation = sim_data['simulation']
    funcs = feature['funcs']

    times = simulation.root.output.solution.solution_times

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

    sse = util.sse(numpy.array(sim_values), numpy.array(exp_values))

    return [-sse,], sse, len(sim_values), numpy.array(sim_values) - numpy.array(exp_values), [sse,]

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    data = pandas.read_csv(feature['csv'])
    rows, cols = data.shape

    smallestTime = min(data['Stop'] - data['Start'])
    abstolFraction = abstol * smallestTime

    headers = data.columns.values.tolist()
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_SSE" % name]
    return temp

