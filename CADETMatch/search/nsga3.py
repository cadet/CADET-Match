
import math
import util
import checkpoint_algorithms
import random
import nsga3_selection

from deap import algorithms

name = "NSGA3"

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()
    parameters = len(cache.MIN_VALUE)

    populationSize=parameters * cache.settings['population']
    CXPB=cache.settings['crossoverRate']

    totalGenerations = parameters * cache.settings['generations']

    pop = cache.toolbox.population(n=populationSize)

    if "seeds" in cache.settings:
        seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
        pop.extend(seed_pop)

    hof = tools.ParetoFront(similar=util.similar)

    return checkpoint_algorithms.eaMuPlusLambda(pop, cache.toolbox,
                              mu=populationSize, 
                              lambda_=populationSize, 
                              cxpb=CXPB, 
                              mutpb=cache.settings['mutationRate'],
                              ngen=totalGenerations,
                              settings=cache.settings,
                              tools=tools,
                              halloffame=hof,
                              cache=cache)

def setupDEAP(cache, fitness, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    cache.toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE)
    cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual)

    cache.toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE)

    if cache.adaptive:
        cache.toolbox.register("mutate", util.mutPolynomialBoundedAdaptive, eta=10.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
    else:
        cache.toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))

    cache.toolbox.register("select", nsga3_selection.sel_nsga_iii)
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)

    cache.toolbox.register('map', map_function)
