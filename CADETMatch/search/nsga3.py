import CADETMatch.util as util
import CADETMatch.checkpoint_algorithms as checkpoint_algorithms
import random

import CADETMatch.pareto as pareto
import array

import scipy.special
from deap import tools
import numpy

import multiprocessing

name = "NSGA3"

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()
    parameters = len(cache.MIN_VALUE)

    populationSize=parameters * cache.settings.get('population', 100)
    CXPB = cache.settings.get('crossoverRate', 1.0)
    MUTPB = cache.settings.get('mutationRate', 1.0)

    totalGenerations = parameters * cache.settings.get('generations', 1000)

    pop = cache.toolbox.population(n=populationSize)

    if "seeds" in cache.settings:
        seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
        pop.extend(seed_pop)
    return checkpoint_algorithms.eaMuPlusLambda(pop, cache.toolbox,
                              mu=populationSize, 
                              lambda_=populationSize, 
                              cxpb=CXPB, 
                              mutpb=MUTPB,
                              ngen=totalGenerations,
                              settings=cache.settings,
                              tools=tools,
                              cache=cache,
                              varOr=False)

def setupDEAP(cache, fitness, grad_fitness, grad_search, map_function, creator, base, tools):
    "setup the DEAP variables"
    ref_points = generate_reference_points(cache.numGoals)
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None, mean=None, confidence=None)

    creator.create("FitnessMaxMeta", base.Fitness, weights=[1.0, 1.0, 1.0, -1.0])
    creator.create("IndividualMeta", array.array, typecode="d", fitness=creator.FitnessMaxMeta, strategy=None)
    cache.toolbox.register("individualMeta", util.initIndividual, creator.IndividualMeta, cache)

    cache.toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE, cache)
        
    if cache.sobolGeneration:
        cache.toolbox.register("population", util.sobolGenerator, creator.Individual, cache)
    else:
        cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)
    cache.toolbox.register("randomPopulation", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual, cache)

    cache.toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=30.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE)

    cache.toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
    cache.toolbox.register("force_mutate", tools.mutPolynomialBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))

    cache.toolbox.register("select", tools.selNSGA3WithMemory(ref_points))
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)
    cache.toolbox.register("evaluate_grad", grad_fitness, json_path=cache.json_path)
    cache.toolbox.register('grad_search', grad_search)

    cache.toolbox.register('map', map_function)


def num_ref_points(n,k):
    return scipy.special.comb(n+k-1,k, exact=True)

def find_max_p(ndim, max_size, max_p):
    for p in range(max_p, 0, -1):
        if num_ref_points(ndim, p) <= max_size:
            break
    return p

def find_ref_point_setup(ndim, max_size, inner):
    p = [find_max_p(ndim, max_size, 32),]
    if p[0] < inner:
        p.append(p[0]-1)
    return p

def generate_reference_points(ndim, max_size=10000, inner=4):
    P = find_ref_point_setup(ndim, max_size, inner)
    SCALES = [1.0, 0.5]
    ref_points = [tools.uniform_reference_points(ndim, p, s) for p, s in zip(P, SCALES)]
    ref_points = numpy.concatenate(ref_points, axis=0)
    _, uniques = numpy.unique(ref_points, axis=0, return_index=True)
    ref_points = ref_points[uniques]
    multiprocessing.get_logger().info("Reference points chosen P = %s  with shape %s", P, ref_points.shape)
    return ref_points