import math
import util
import evo
import checkpoint_algorithms
import random

from deap import algorithms

def run(settings, toolbox, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(evo.MIN_VALUE)

    LAMBDA=parameters * settings['population']
    MU = int(math.ceil(settings['keep']*LAMBDA))
    if MU < 2:
        MU = 2

    pop = toolbox.population(n=LAMBDA)

    totalGenerations = parameters * settings['generations']

    hof = tools.ParetoFront()

    return checkpoint_algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
        cxpb=settings['crossoverRate'], mutpb=settings['mutationRate'], ngen=totalGenerations, settings=settings, halloffame=hof, tools=tools)

def setupDEAP(numGoals, settings, target, MIN_VALUE, MAX_VALUE, fitness, map_function, creator, toolbox, base, tools):
    "setup the DEAP variables"

    creator.create("FitnessMax", base.Fitness, weights=[1.0] * numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(MIN_VALUE), MIN_VALUE, MAX_VALUE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("individual_guess", util.initIndividual, creator.Individual)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=MIN_VALUE, up=MAX_VALUE)
    toolbox.register("mutate", util.mutPolynomialBoundedAdaptive, eta=10.0, low=MIN_VALUE, up=MAX_VALUE, indpb=1.0/len(MIN_VALUE))

    toolbox.register("select", tools.selSPEA2)
    toolbox.register("evaluate", fitness)

    toolbox.register('map', map_function)
    return toolbox