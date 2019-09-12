import util
import checkpoint_algorithms
import random
import nsga3_selection
from deap import tools
import pareto
import array
import numpy

import SALib.sample.sobol_sequence
from collections import Sequence

name = "NSGA3_strategy"

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()
    parameters = len(cache.MIN_VALUE)

    populationSize=parameters * cache.settings['population']
    CXPB = cache.settings['crossoverRate']

    totalGenerations = parameters * cache.settings['generations']

    pop = cache.toolbox.population(n=populationSize)

    if "seeds" in cache.settings:
        seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
        pop.extend(seed_pop)

    return checkpoint_algorithms.eaMuPlusLambda(pop, cache.toolbox,
                              mu=populationSize, 
                              lambda_=populationSize, 
                              cxpb=CXPB, 
                              mutpb=cache.settings['mutationRate'],
                              ngen=totalGenerations,
                              settings=cache.settings,
                              tools=tools,
                              cache=cache)

def generateIndividualStrategy(icls, scls, size, imin, imax, smin, smax, cache):
    ind = icls(numpy.random.uniform(imin, imax))
    ind.strategy = scls(numpy.random.uniform(smin, smax))
    ind.mean = array.array('d', [0.0]*len(imin))
    ind.confidence = array.array('d', [0.0]*len(imin))
    return ind

def minmax(x, lb, ub):
    return min(max(x,lb),ub)

def cxESBlend(ind1, ind2, alpha, imins, imaxs, smins, smaxs):

    ind1, ind2 = tools.cxESBlend(ind1, ind2, alpha)

    for i, (imin, imax, smin, smax) in enumerate(zip(imins, imaxs, smins, smaxs)):
        # Blend the values
        ind1[i] = minmax( ind1[i], imin, imax )
        ind2[i] = minmax( ind2[i], imin, imax )
        # Blend the strategies
        ind1.strategy[i] = minmax( ind1.strategy[i], smin, smax )
        ind2.strategy[i] = minmax( ind1.strategy[i], smin, smax )

    return ind1, ind2

def mutESLogNormal(individual, c, indpb, imins, imaxs, smins, smaxs):

    c = 0.7 + numpy.min(individual.fitness.values) * 0.3

    individual = tools.mutESLogNormal(individual, c, indpb)

    for i, (imin, imax, smin, smax) in enumerate(zip(imins, imaxs, smins, smaxs)):
        # Blend the values
        individual[0][i] = minmax( individual[0][i], imin, imax )
        # Blend the strategies
        individual[0].strategy[i] = minmax( individual[0].strategy[i], smin, smax )

    return individual

def sobolGenerator(icls, scls, cache, smin, smax, n):
    if n > 0:
        populationDimension = len(cache.MIN_VALUE)
        populationSize = n
        sobol = SALib.sample.sobol_sequence.sample(populationSize, populationDimension)
        data = numpy.apply_along_axis(list, 1, sobol)
        data = list(map(icls, data))

        for i in data:
            i.strategy = scls(numpy.random.uniform(smin, smax))
            i.mean = array.array('d', [0.0]*len(smin))
            i.confidence = array.array('d', [0.0]*len(smin))
        return data
    else:
        return []

def setupDEAP(cache, fitness, grad_fitness, grad_search, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None, mean=None, confidence=None)
    creator.create("Strategy", array.array, typecode="d")

    creator.create("FitnessMaxMeta", base.Fitness, weights=[1.0, 1.0, 1.0, -1.0])
    creator.create("IndividualMeta", array.array, typecode="d", fitness=creator.FitnessMaxMeta, strategy=None)
    cache.toolbox.register("individualMeta", util.initIndividual, creator.IndividualMeta, cache)

    MIN_STRAT = [80.0] * len(cache.MIN_VALUE)
    MAX_STRAT = [200.0] * len(cache.MIN_VALUE)

    MAX_STRAT_START = [120.0] * len(cache.MIN_VALUE)


    cache.toolbox.register("individual", generateIndividualStrategy, creator.Individual, creator.Strategy,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE, MIN_STRAT, MAX_STRAT, cache)
        
    if cache.sobolGeneration:
        cache.toolbox.register("population", sobolGenerator, creator.Individual, creator.Strategy, cache, MIN_STRAT, MAX_STRAT_START)
    else:
        cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)
    cache.toolbox.register("randomPopulation", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual, cache)

    #cache.toolbox.register("mate", cxESBlend, alpha=0.1, imins=cache.MIN_VALUE, imaxs=cache.MAX_VALUE, smins=MIN_STRAT, smaxs=MAX_STRAT)


    cache.toolbox.register("mate", cxSimulatedBinaryBounded, low=cache.MIN_VALUE, up=cache.MAX_VALUE, slow=MIN_STRAT, sup=MAX_STRAT)

    #if cache.adaptive:
    #    cache.toolbox.register("mutate", util.mutationBoundedAdaptive, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
    #    cache.toolbox.register("force_mutate", util.mutationBoundedAdaptive, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
    #else:
    cache.toolbox.register("mutate", mutESLogNormal, c=0.7, indpb=1.0/len(cache.MIN_VALUE), imins=cache.MIN_VALUE, imaxs=cache.MAX_VALUE, smins=MIN_STRAT, smaxs=MAX_STRAT)
    cache.toolbox.register("force_mutate", mutESLogNormal, c=0.7, indpb=1.0/len(cache.MIN_VALUE), imins=cache.MIN_VALUE, imaxs=cache.MAX_VALUE, smins=MIN_STRAT, smaxs=MAX_STRAT)

    cache.toolbox.register("select", nsga3_selection.sel_nsga_iii)
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)
    cache.toolbox.register("evaluate_grad", grad_fitness, json_path=cache.json_path)
    cache.toolbox.register('grad_search', grad_search)

    cache.toolbox.register('map', map_function)



def cxSimulatedBinaryBounded(ind1, ind2, low, up, slow, sup):
    """Executes a simulated binary crossover that modify in-place the input
    individuals. The simulated binary crossover expects :term:`sequence`
    individuals of floating point numbers.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param eta: Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different.
    :param low: A value or a :term:`python:sequence` of values that is the lower
                bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that is the upper
               bound of the search space.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    .. note::
       This implementation is similar to the one implemented in the
       original NSGA-II C code presented by Deb.
    """
    size = min(len(ind1), len(ind2))
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of the shorter individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of the shorter individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= 0.5:
            # This epsilon should probably be changed for 0 since
            # floating point arithmetic in Python is safer
            if abs(ind1[i] - ind2[i]) > 1e-14:

                if ind1[i] < ind2[i]:
                    x1 = ind1[i]
                    x2 = ind2[i]
                    eta1 = ind1.strategy[i]
                    eta2 = ind2.strategy[i]
                else:
                    x1 = ind2[i]
                    x2 = ind1[i]
                    eta1 = ind2.strategy[i]
                    eta2 = ind1.strategy[i]

                #x1 = min(ind1[i], ind2[i])
                #x2 = max(ind1[i], ind2[i])
                rand = random.random()

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta**-(eta1 + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha)**(1.0 / (eta1 + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta1 + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta**-(eta2 + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha)**(1.0 / (eta2 + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta2 + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)

                if random.random() <= 0.5:
                    ind1[i] = c2
                    ind2[i] = c1
                    ind1.strategy[i] = eta2
                    ind2.strategy[i] = eta1

                    mean, confidence = util.confidence_eta(eta2, xl, xu)
                    ind1.mean[i] = mean
                    ind1.confidence[i] = confidence

                    mean, confidence = util.confidence_eta(eta1, xl, xu)
                    ind2.mean[i] = mean
                    ind2.confidence[i] = confidence
                else:
                    ind1[i] = c1
                    ind2[i] = c2
                    ind1.strategy[i] = eta1
                    ind2.strategy[i] = eta2

                    mean, confidence = util.confidence_eta(eta1, xl, xu)
                    ind1.mean[i] = mean
                    ind1.confidence[i] = confidence

                    mean, confidence = util.confidence_eta(eta2, xl, xu)
                    ind2.mean[i] = mean
                    ind2.confidence[i] = confidence
        else:
            eta1 = ind1.strategy[i]
            eta2 = ind2.strategy[i]

            mean, confidence = util.confidence_eta(eta1, xl, xu)
            ind1.mean[i] = mean
            ind1.confidence[i] = confidence

            mean, confidence = util.confidence_eta(eta2, xl, xu)
            ind2.mean[i] = mean
            ind2.confidence[i] = confidence

    return ind1, ind2
