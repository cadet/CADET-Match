import peakdetect
import random
import math

from deap import tools
import scipy.signal


def find_extreme(seq):
    try:
        return max(seq, key=lambda x: abs(x[1]))
    except ValueError:
        return [0,0]

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
    #filter length must be odd, set to 10% of the feature size and then make it odd if necesary
    filter_length = int(.1 * len(values))
    if filter_length % 2 == 0:
        filter_length += 1
    return scipy.signal.savgol_filter(values, filter_length, 3)
    #return scipy.signal.hilbert(values)
    return values
