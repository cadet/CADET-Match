import peakdetect
import random
import math
import numpy

from deap import tools
import scipy.signal
from scipy.spatial.distance import cdist

def smoothing_factor(y):
    return max(y)/1000000.0

def find_extreme(seq):
    try:
        return max(seq, key=lambda x: abs(x[1]))
    except ValueError:
        return [0,0]

def get_times_values(simulation, target):

    times = simulation.root.output.solution.solution_times

    isotherm = target['isotherm']
    
    if isinstance(isotherm, list):
        values = numpy.sum([simulation[i] for i in isotherm],0)
    else:
        values = simulation[isotherm]

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
    return values
    #filter length must be odd, set to 10% of the feature size and then make it odd if necesary
    filter_length = int(.1 * len(values))
    if filter_length % 2 == 0:
        filter_length += 1
    return scipy.signal.savgol_filter(values, filter_length, 3)
    #return scipy.signal.hilbert(values)

def set_value(node, nameH5, dtype, value):
    "merge the values from parms into node of the hdf5 file"
    data = numpy.array(value, dtype=dtype)
    if node.get(nameH5, None) is not None:
        del node[nameH5]
    node.create_dataset(nameH5, data=data, maxshape=tuple(None for i in range(data.ndim)), fillvalue=[0])

def set_value_enum(node, nameH5, value):
    "merge the values from parms into node of the hdf5 file"
    if isinstance(value, list):
        dtype = 'S' + str(len(value[0])+1)
    else:
        dtype = 'S' + str(len(value)+1)
    data = numpy.array(value, dtype=dtype)
    if node.get(nameH5, None) is not None:
        del node[nameH5]
    node.create_dataset(nameH5, data=data)

def set_int(node, nameH5, value):
    "set one or more integers in the hdf5 file"
    data = numpy.array(value, dtype="i4")
    if node.get(nameH5, None) is not None:
        del node[nameH5]
    node.create_dataset(nameH5, data=data, maxshape=tuple(None for i in range(data.ndim)), fillvalue=[0])

def set_double(node, nameH5, value):
    "set one or more doubles in the hdf5 file"
    data = numpy.array(value, dtype="f8")
    if node.get(nameH5, None) is not None:
        del node[nameH5]
    node.create_dataset(nameH5, data=data, maxshape=tuple(None for i in range(data.ndim)), fillvalue=[0])

def set_string(node, nameH5, value):
    "set a string value in the hdf5 file"
    if isinstance(value, list):
        dtype = 'S' + str(len(value[0])+1)
    else:
        dtype = 'S' + str(len(value)+1)
    data = numpy.array(value, dtype=dtype)
    if node.get(nameH5, None) is not None:
        del node[nameH5]
    node.create_dataset(nameH5, data=data)