
import math
import util
import evo
import checkpoint_algorithms
import random
import nsga3_selection

from deap import algorithms

def run(settings, toolbox, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(evo.MIN_VALUE)

    populationSize=parameters * settings['population']
    CXPB=settings['crossoverRate']

    totalGenerations = parameters * settings['generations']

    hof = tools.ParetoFront()

    checkpoint_algorithms.eaMuPlusLambda(toolbox,
                              mu=populationSize, 
                              lambda_=populationSize, 
                              cxpb=CXPB, 
                              mutpb=settings['mutationRate'],
                              ngen=totalGenerations,
                              settings=settings,
                              tools=tools,
                              halloffame=hof)

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

    toolbox.register("select", nsga3_selection.sel_nsga_iii)
    toolbox.register("evaluate", fitness)

    toolbox.register('map', map_function)
    return toolbox