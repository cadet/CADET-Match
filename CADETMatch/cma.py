import random
import math
import pickle
import util
import numpy
import array
from pathlib import Path

from deap import algorithms
from deap.benchmarks.tools import hypervolume
import deap.cma

def run(settings, toolbox, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(settings['parameters'])

    LAMBDA=parameters * settings['population']
    MU = int(math.ceil(settings['keep']*LAMBDA))
    if MU < 2:
        MU = 2

    population = toolbox.population(n=LAMBDA)

    NGEN = parameters * settings['generations']

    verbose = False

    # The MO-CMA-ES algorithm takes a full population as argument
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        print(min(fit))

    avg, bestMin = util.averageFitness(population)
    print('avg', avg, 'best', bestMin)

    strategy = deap.cma.StrategyMultiObjective(population, sigma=1.0, lambda_ = LAMBDA)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
   
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    for gen in range(NGEN):
        # Generate a new population
        population = toolbox.generate()

        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            print(min(fit))

        avg, bestMin = util.averageFitness(population)
        print('avg', avg, 'best', bestMin)
        
        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        avg, bestMin = util.averageFitness(population)
        print('avg', avg, 'best', bestMin)
        
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)
        
        #if verbose:
        #   print("Population hypervolume is %f" % hypervolume(strategy.parents, [0.0, 0.0]))

        if avg > settings['stopAverage'] or bestMin > settings['stopBest']:
            return        
    
def setupDEAP(numGoals, settings, target, MIN_VALUE, MAX_VALUE, fitness, map_function, creator, toolbox, base, tools):
    "setup the DEAP variables"

    creator.create("FitnessMax", base.Fitness, weights=[1.0] * numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(MIN_VALUE), MIN_VALUE, MAX_VALUE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("individual_guess", util.initIndividual, creator.Individual)

    toolbox.register("evaluate", fitness)
    toolbox.decorate("evaluate", tools.ClosestValidPenalty(valid, closest_feasible, 1.0e-6, distance))

    toolbox.register('map', map_function)
    return toolbox

def distance(feasible_ind, original_ind):
    """A distance function to the feasibility region."""
    return sum((f - o)**2 for f, o in zip(feasible_ind, original_ind))

def closest_feasible(individual):
    """A function returning a valid individual from an invalid one."""
    feasible_ind = numpy.array(individual)
    feasible_ind = numpy.maximum(MIN_VALUE, feasible_ind)
    feasible_ind = numpy.minimum(MAX_VALUE, feasible_ind)
    return feasible_ind

def valid(individual):
    """Determines if the individual is valid or not."""
    if any(numpy.array(individual) < MIN_VALUE) or any(numpy.array(individual) > MAX_VALUE):
        return False
    return True