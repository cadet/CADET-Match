import math
import util
import evo
import checkpoint_algorithms
import random

from deap import algorithms

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(evo.MIN_VALUE)

    LAMBDA=parameters * cache.settings['population']
    MU = int(math.ceil(cache.settings['keep']*LAMBDA))
    if MU < 2:
        MU = 2

    pop = cache.toolbox.population(n=LAMBDA)

    if "seeds" in cache.settings:
        seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
        pop.extend(seed_pop)

    totalGenerations = parameters * cache.settings['generations']

    hof = tools.ParetoFront()

    #return checkpoint_algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
    #    cxpb=settings['crossoverRate'], mutpb=settings['mutationRate'], ngen=totalGenerations, settings=settings, halloffame=hof, tools=tools)

    return checkpoint_algorithms.eaMuPlusLambda(pop, cache.toolbox, mu=MU, lambda_=LAMBDA,
        cxpb=cache.settings['crossoverRate'], mutpb=cache.settings['mutationRate'], ngen=totalGenerations, settings=cache.settings, halloffame=hof, tools=tools)

def setupDEAP(cache, fitness, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    cache.toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE)
    cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual)

    cache.toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE)
    cache.toolbox.register("mutate", util.mutPolynomialBoundedAdaptive, eta=10.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))

    cache.toolbox.register("select", tools.selSPEA2)
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)

    cache.toolbox.register('map', map_function)