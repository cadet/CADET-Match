import peakdetect
import random
import math
import numpy

from deap import tools
import scipy.signal
from scipy.spatial.distance import cdist
import operator
import functools
from collections import Sequence

from addict import Dict

saltIsotherms = {b'STERIC_MASS_ACTION', b'SELF_ASSOCIATION', b'MULTISTATE_STERIC_MASS_ACTION', 
                 b'SIMPLE_MULTISTATE_STERIC_MASS_ACTION', b'BI_STERIC_MASS_ACTION'}

def smoothing_factor(y):
    return max(y)/1000000.0

def find_extreme(seq):
    try:
        return max(seq, key=lambda x: abs(x[1]))
    except ValueError:
        return [0,0]

def get_times_values(simulation, target, selected = None):

    times = simulation.root.output.solution.solution_times

    isotherm = target['isotherm']

    if isinstance(isotherm, list):
        values = numpy.sum([simulation[i] for i in isotherm],0)
    else:
        values = simulation[isotherm]
    
    if selected is None:
        selected = target['selected']

    return times[selected], values[selected]

def sse(data1, data2):
    return numpy.sum( (data1 - data2)**2 )

def find_peak(times, data):
    "Return tuples of (times,data) for the peak we need"
    [highs, lows] = peakdetect.peakdetect(data, times, 1)

    return find_extreme(highs), find_extreme(lows)

def find_breakthrough(times, data):
    "return tupe of time,value for the start breakthrough and end breakthrough"
    selected = data > 0.999 * max(data)
    selected_times = times[selected]
    return (selected_times[0], max(data)), (selected_times[-1], max(data))

def generateIndividual(icls, size, imin, imax):
    while 1:
        ind = icls(random.uniform(imin[idx], imax[idx]) for idx in range(size))
        if feasible(ind):
            return ind

def initIndividual(icls, content):
    return icls(content)

def feasible(individual):
    "evaluate if this individual is feasible"

    return True

print_log = 0

def log(*args):
    if print_log:
        print(args)

def averageFitness(offspring):
    total = 0.0
    number = 0.0
    bestMin = 0.0

    for i in offspring:
        total += sum(i.fitness.values)
        number += len(i.fitness.values)
        bestMin = max(bestMin, min(i.fitness.values))
    return total/number, bestMin

def smoothing(times, values):
    #temporarily get rid of smoothing for debugging
    #return values
    #filter length must be odd, set to 10% of the feature size and then make it odd if necesary
    filter_length = int(.1 * len(values))
    if filter_length % 2 == 0:
        filter_length += 1
    return scipy.signal.savgol_filter(values, filter_length, 3)
    #return scipy.signal.hilbert(values)

def graph_simulation(simulation, graph):
    ncomp = simulation.root.input.model.unit_001.ncomp
    isotherm = simulation.root.input.model.unit_001.adsorption_model

    hasSalt = isotherm in saltIsotherms

    solution_times = simulation.root.output.solution.solution_times

    comps = []

    hasColumn = isinstance(simulation.root.output.solution.unit_001.solution_outlet_comp_000, Dict)

    if hasColumn:
        for i in range(ncomp):
            comps.append(simulation.root.output.solution.unit_001['solution_column_outlet_comp_%03d' % i])
    else:
        for i in range(ncomp):
            comps.append(simulation.root.output.solution.unit_001['solution_outlet_comp_%03d' % i])

    if hasSalt:
        graph.set_title("Output")
        graph.plot(solution_times, comps[0], 'b-', label="Salt")
        graph.set_xlabel('time (s)')
        
        # Make the y-axis label, ticks and tick labels match the line color.
        graph.set_ylabel('mMol Salt', color='b')
        graph.tick_params('y', colors='b')

        colors = ['r', 'g', 'c', 'm', 'y', 'k']
        axis2 = graph.twinx()
        for idx, comp in enumerate(comps[1:]):
            axis2.plot(solution_times, comp, '%s-' % colors[idx], label="P%s" % idx)
        axis2.set_ylabel('mMol Protein', color='r')
        axis2.tick_params('y', colors='r')


        lines, labels = graph.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis2.legend(lines + lines2, labels + labels2, loc=0)
    else:
        graph.set_title("Output")
        
        colors = ['r', 'g', 'c', 'm', 'y', 'k']
        for idx, comp in enumerate(comps):
            graph.plot(solution_times, comp, '%s-' % colors[idx], label="P%s" % idx)
        graph.set_ylabel('mMol Protein', color='r')
        graph.tick_params('y', colors='r')
        graph.set_xlabel('time (s)')

        lines, labels = graph.get_legend_handles_labels()
        graph.legend(lines, labels, loc=0)


def mutPolynomialBoundedAdaptive(individual, eta, low, up, indpb):
    """Adaptive eta for mutPolynomialBounded"""
    scores = individual.fitness.values
    prod = functools.reduce(operator.mul, scores, 1)**(1.0/len(scores))
    eta = eta + prod * 100
    return tools.mutPolynomialBounded(individual, eta, low, up, indpb)

def plotExperiments(save_name_base, settings, target, results, directory, file_pattern):
    for experiment in settings['experiments']:
        experimentName = experiment['name']
        
        dst = Path(directory, file_pattern % (save_name_base, experimentName))

        numPlots = len(experiment['features']) + 1  #1 additional plot added as an overview for the simulation

        exp_time = target[experimentName]['time']
        exp_value = target[experimentName]['value']

        fig = plt.figure(figsize=[10, numPlots*10])

        util.graph_simulation(results[experimentName]['simulation'], fig.add_subplot(numPlots, 1, 1))

        for idx,feature in enumerate(experiment['features']):
            graph = fig.add_subplot(numPlots, 1, idx+1+1) #additional +1 added due to the overview plot
            
            featureName = feature['name']
            featureType = feature['type']

            feat = target[experimentName][featureName]

            selected = feat['selected']
            exp_time = feat['time'][selected]
            exp_value = feat['value'][selected]

            sim_time, sim_value = util.get_times_values(results[experimentName]['simulation'],target[experimentName][featureName])

            if featureType in ('similarity', 'similarityDecay', 'similarityHybrid', 'similarityHybridDecay','curve', 'breakthrough', 'dextran', 'dextranHybrid', 'similarityCross', 'similarityCrossDecay', 'breakthroughCross'):
                graph.plot(sim_time, sim_value, 'r--', label='Simulation')
                graph.plot(exp_time, exp_value, 'g:', label='Experiment')
            elif featureType in ('derivative_similarity', 'derivative_similarity_hybrid', 'derivative_similarity_cross', 'derivative_similarity_cross_alt'):
                try:
                    sim_spline = scipy.interpolate.UnivariateSpline(sim_time, util.smoothing(sim_time, sim_value), s=util.smoothing_factor(sim_value)).derivative(1)
                    exp_spline = scipy.interpolate.UnivariateSpline(exp_time, util.smoothing(exp_time, exp_value), s=util.smoothing_factor(exp_value)).derivative(1)

                    graph.plot(sim_time, sim_spline(sim_time), 'r--', label='Simulation')
                    graph.plot(exp_time, exp_spline(exp_time), 'g:', label='Experiment')
                except:
                    pass
            elif featureType in ('fractionation', 'fractionationCombine'):
                graph_exp = results[experimentName]['graph_exp']
                graph_sim = results[experimentName]['graph_sim']

                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

                for idx,(key,value) in enumerate(graph_sim.items()):
                    (time, values) = zip(*value)
                    graph.plot(time, values, '%s--' % colors[idx], label='Simulation Comp: %s' % key)

                for idx,(key,value) in enumerate(graph_exp.items()):
                    (time, values) = zip(*value)
                    graph.plot(time, values, '%s:' % colors[idx], label='Experiment Comp: %s' % key)
            graph.legend()

        plt.savefig(bytes(dst), dpi=100)
        plt.close()

def saveExperiments(save_name_base, settings,target, results, directory, file_pattern):
    for experiment in settings['experiments']:
        experimentName = experiment['name']

        dst = Path(directory, file_pattern % (save_name_base, experimentName))

        if dst.is_file():  #File already exists don't try to write over it
            return False
        else:
            simulation = results[experimentName]['simulation']
            simulation.filename = bytes(dst)

            for (header, score) in zip(experiment['headers'], results[experimentName]['scores']):
                simulation.root.score[header] = score
            simulation.save()
    return True