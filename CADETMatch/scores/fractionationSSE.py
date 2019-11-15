import CADETMatch.util as util
import CADETMatch.score as score
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
settings.failure = [0.0] * settings.count, 1e6, 1, numpy.array([0.0]), numpy.array([0.0]), numpy.array([1e6]), [1.0] * settings.count

def run(sim_data, feature):
    "similarity, value, start stop"
    simulation = sim_data['simulation']
    start = feature['start']
    stop = feature['stop']
    comps = feature['comps']
    data = feature['data']

    time_center = (start + stop)/2.0

    times = simulation.root.output.solution.solution_times

    sim_values_sse = []
    exp_values_sse = []
   
    graph_sim = {}
    graph_exp = {}

    for component  in comps:
        exp_values = numpy.array(data[str(component)])
        sim_value = simulation.root.output.solution[feature['unit']]["solution_outlet_comp_%03d" % component]

        fractions = util.fractionate(start, stop, times, sim_value)

        exp_values_sse.extend(exp_values)
        sim_values_sse.extend(fractions)

        graph_sim[component] = list(zip(time_center, fractions))
        graph_exp[component] = list(zip(time_center, exp_values))

    #sort lists
    for key, value in graph_sim.items():
        value.sort()
    for key, value in graph_exp.items():
        value.sort()

    sim_data['graph_exp'] = graph_exp
    sim_data['graph_sim'] = graph_sim

    sse = util.sse(numpy.array(sim_values_sse), numpy.array(exp_values_sse))

    return [-sse,], sse, len(sim_values_sse), time_center, numpy.array(sim_values_sse), numpy.array(exp_values_sse), [sse,]

def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    temp = {}
    data = pandas.read_csv(feature['fraction_csv'])

    temp['comps'] = [int(i) for i in data.columns.values.tolist()[2:]]

    temp['data'] = data
    temp['start'] = numpy.array(data.iloc[:, 0])
    temp['stop'] = numpy.array(data.iloc[:, 1])
    temp['unit'] = feature['unit_name']

    smallestTime = min(data['Stop'] - data['Start'])
    abstolFraction = abstol * smallestTime

    headers = data.columns.values.tolist()
    temp['peak_max'] = data.iloc[:,2:].max().min()
    return temp

def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature['name'])
    temp = ["%s_SSE" % name]
    return temp

