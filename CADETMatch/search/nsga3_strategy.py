import util
import checkpoint_algorithms
import random
import nsga3_selection

from deap import algorithms
from deap import tools
import functools
import pareto
import array
import numpy

import SALib.sample.sobol_sequence

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

    hof = pareto.ParetoFront(similar=util.similar)
    meta_hof = pareto.ParetoFront(similar=util.similar)

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
    if cache.roundParameters is not None:
        ind = icls(util.RoundToSigFigs(numpy.random.uniform(imin, imax), cache.roundParameters))
    else:
        ind = icls(numpy.random.uniform(imin, imax))
    ind.strategy = scls(numpy.random.uniform(smin, smax))
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
        return data
    else:
        return []

def setupDEAP(cache, fitness, grad_fitness, grad_search, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    creator.create("FitnessMaxMeta", base.Fitness, weights=[1.0] * 4)
    creator.create("IndividualMeta", array.array, typecode="d", fitness=creator.FitnessMaxMeta, strategy=None)
    cache.toolbox.register("individualMeta", util.initIndividual, creator.IndividualMeta, cache)

    MIN_STRAT = [-1.0] * len(cache.MIN_VALUE)
    MAX_STRAT = [1.0] * len(cache.MIN_VALUE)


    cache.toolbox.register("individual", generateIndividualStrategy, creator.Individual, creator.Strategy,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE, MIN_STRAT, MAX_STRAT, cache)
        
    if cache.sobolGeneration:
        cache.toolbox.register("population", sobolGenerator, creator.Individual, creator.Strategy, cache, MIN_STRAT, MAX_STRAT)
    else:
        cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)
    cache.toolbox.register("randomPopulation", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual, cache)

    cache.toolbox.register("mate", cxESBlend, alpha=0.1, imins=cache.MIN_VALUE, imaxs=cache.MAX_VALUE, smins=MIN_STRAT, smaxs=MAX_STRAT)

    #if cache.adaptive:
    #    cache.toolbox.register("mutate", util.mutationBoundedAdaptive, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
    #    cache.toolbox.register("force_mutate", util.mutationBoundedAdaptive, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
    #else:
    cache.toolbox.register("mutate", mutESLogNormal, c=1.0, indpb=1.0/len(cache.MIN_VALUE), imins=cache.MIN_VALUE, imaxs=cache.MAX_VALUE, smins=MIN_STRAT, smaxs=MAX_STRAT)
    cache.toolbox.register("force_mutate", mutESLogNormal, c=1.0, indpb=1.0/len(cache.MIN_VALUE), imins=cache.MIN_VALUE, imaxs=cache.MAX_VALUE, smins=MIN_STRAT, smaxs=MAX_STRAT)

    cache.toolbox.register("select", nsga3_selection.sel_nsga_iii)
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)
    cache.toolbox.register("evaluate_grad", grad_fitness, json_path=cache.json_path)
    cache.toolbox.register('grad_search', grad_search)

    cache.toolbox.register('map', map_function)

