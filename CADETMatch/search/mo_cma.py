import CADETMatch.util as util
import CADETMatch.checkpoint_algorithms as checkpoint_algorithms
import random

import CADETMatch.pareto as pareto
import array

import scipy.special
from deap import tools
from deap import cma
import numpy

import multiprocessing

name = "MO_CMA"

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()
    parameters = len(cache.MIN_VALUE)

    populationSize=parameters * cache.settings.get('population', 100)

    totalGenerations = parameters * cache.settings.get('generations', 1000)

    pop = cache.toolbox.population(n=populationSize)

    if "seeds" in cache.settings:
        seed_pop = [cache.toolbox.individual_guess([f(v) for f, v in zip(cache.settings['transform'], sublist)]) for sublist in cache.settings['seeds']]
        pop.extend(seed_pop)
    return cma(pop, cache.toolbox,
                    mu=populationSize, 
                    lambda_=populationSize, 
                    ngen=totalGenerations,
                    settings=cache.settings,
                    tools=tools,
                    cache=cache,
                    varOr=False)

def setupDEAP(cache, fitness, grad_fitness, grad_search, map_function, creator, base, tools):
    "setup the DEAP variables"
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

    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)
    cache.toolbox.register("evaluate_grad", grad_fitness, json_path=cache.json_path)
    cache.toolbox.register('grad_search', grad_search)

    cache.toolbox.register('map', map_function)

def distance(feasible_ind, original_ind):
    """A distance function to the feasibility region."""
    return sum((f - o)**2 for f, o in zip(feasible_ind, original_ind))

def closest_feasible(individual):
    """A function returning a valid individual from an invalid one."""
    feasible_ind = numpy.array(individual)
    feasible_ind = numpy.maximum(MIN_BOUND, feasible_ind)
    feasible_ind = numpy.minimum(MAX_BOUND, feasible_ind)
    return feasible_ind

def valid(individual):
    """Determines if the individual is valid or not."""
    if any(individual < MIN_BOUND) or any(individual > MAX_BOUND):
        return False
    return True

def close_valid(individual):
    """Determines if the individual is close to valid."""
    if any(individual < MIN_BOUND-EPS_BOUND) or any(individual > MAX_BOUND+EPS_BOUND):
        return False
    return True

def cma(pop, toolbox, mu, lambda_, ngen, settings, tools, cache):
    verbose = True

    # The MO-CMA-ES algorithm takes a full population as argument
    population = pop

    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    strategy = cma.StrategyMultiObjective(population, sigma=1.0, mu=mu, lambda_=lambda_)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
   
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    fitness_history = []

    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()

        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            fitness_history.append(fit)
        
        # Update the strategy with the evaluated individuals
        toolbox.update(population)
        
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    if verbose:
        print("Final population hypervolume is %f" % hypervolume(strategy.parents, [0.0] * cache.numGoals))

        # Note that we use a penalty to guide the search to feasible solutions,
        # but there is no guarantee that individuals are valid.
        # We expect the best individuals will be within bounds or very close.
        num_valid = 0
        for ind in strategy.parents:
            dist = distance(closest_feasible(ind), ind)
            if numpy.isclose(dist, 0.0, rtol=1.e-5, atol=1.e-5):
                num_valid += 1
        print("Number of valid individuals is %d/%d" % (num_valid, len(strategy.parents)))

        print("Final population:")
        print(numpy.asarray(strategy.parents))

    return strategy.parents
