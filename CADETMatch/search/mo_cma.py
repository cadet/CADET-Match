from pathlib import Path
import csv
import CADETMatch.util as util
import CADETMatch.checkpoint_algorithms as checkpoint_algorithms
import random

import CADETMatch.pareto as pareto
import array

import scipy.special
from deap import tools
from deap import cma
import numpy
from functools import wraps
from collections.abc import Sequence
from itertools import repeat

import multiprocessing
import CADETMatch.cache as cache
import time

name = "MO_CMA"

class ClosestValidPenalty(object):
    """This decorator returns penalized fitness for invalid individuals and the
    original fitness value for valid individuals. The penalized fitness is made
    of the fitness of the closest valid individual added with a weighted
    (optional) *distance* penalty. The distance function, if provided, shall
    return a value growing as the individual moves away the valid zone.
    :param feasibility: A function returning the validity status of any
                        individual.
    :param feasible: A function returning the closest feasible individual
                     from the current invalid individual.
    :param alpha: Multiplication factor on the distance between the valid and
                  invalid individual.
    :param distance: A function returning the distance between the individual
                     and a given valid point. The distance function can also return a sequence
                     of length equal to the number of objectives to affect multi-objective
                     fitnesses differently (optional, defaults to 0).
    :returns: A decorator for evaluation function.
    This function relies on the fitness weights to add correctly the distance.
    The fitness value of the ith objective is defined as
    .. math::
       f^\mathrm{penalty}_i(\mathbf{x}) = f_i(\operatorname{valid}(\mathbf{x})) - \\alpha w_i d_i(\operatorname{valid}(\mathbf{x}), \mathbf{x})
    where :math:`\mathbf{x}` is the individual,
    :math:`\operatorname{valid}(\mathbf{x})` is a function returning the closest
    valid individual to :math:`\mathbf{x}`, :math:`\\alpha` is the distance
    multiplicative factor and :math:`w_i` is the weight of the ith objective.
    """

    def __init__(self, feasibility, feasible, alpha, distance, weights):
        self.fbty_fct = feasibility
        self.fbl_fct = feasible
        self.alpha = alpha
        self.dist_fct = distance
        self.weights = weights

    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            if self.fbty_fct(individual, cache.cache):
                return func(individual, *args, **kwargs)

            f_ind = self.fbl_fct(individual, cache.cache)
            f_fbl = func(f_ind, *args, **kwargs)
            fit, csv_line, results, new_individual = f_fbl

            weights = tuple(1.0 if w >= 0 else -1.0 for w in self.weights)

            if len(weights) != len(fit):
                raise IndexError("Fitness weights and computed fitness are of different size.")

            dists = tuple(0 for w in self.weights)
            if self.dist_fct is not None:
                dists = self.dist_fct(f_ind, individual)
                if not isinstance(dists, Sequence):
                    dists = repeat(dists)

            # print("penalty ", tuple(  - w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists)))
            # print("returned", tuple(f - w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists)))
            a =  tuple(f - w * self.alpha * d for f, w, d in zip(fit, weights, dists))
            return a, csv_line, results, individual

        return wrapper

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
    return cma_search(pop, cache.toolbox,
                    mu=populationSize, 
                    lambda_=populationSize, 
                    ngen=totalGenerations,
                    settings=cache.settings,
                    tools=tools,
                    cache=cache,
                    creator=creator)

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

def closest_feasible(individual, cache):
    """A function returning a valid individual from an invalid one."""
    feasible_ind = numpy.array(individual)
    feasible_ind = numpy.maximum(cache.MIN_VALUE, feasible_ind)
    feasible_ind = numpy.minimum(cache.MAX_VALUE, feasible_ind)
    clone = cache.toolbox.clone(individual)
    clone[:] = array.array("d", feasible_ind)
    return clone

def valid(individual, cache):
    """Determines if the individual is valid or not."""
    individual = numpy.array(individual)
    lb = numpy.array(cache.MIN_VALUE)
    ub = numpy.array(cache.MAX_VALUE)
    if any(individual < lb) or any(individual > ub):
        return False
    return True

def cma_search(pop, toolbox, mu, lambda_, ngen, settings, tools, cache, creator):
    verbose = True
    checkpointFile = Path(settings['resultsDirMisc'], settings.get('checkpointFile', 'check'))
    sim_start = generation_start = time.time()


    path = Path(cache.settings['resultsDirBase'], cache.settings['csv'])
    result_data = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}, 'input_transform':[], 'input_transform_extended':[], 'strategy':[], 
                   'mean':[], 'confidence':[]}
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        halloffame = pareto.ParetoFront(similar=util.similar, similar_fit=util.similar_fit)
        meta_hof = pareto.ParetoFront(similar=util.similar, similar_fit=util.similar_fit_meta)
        grad_hof = pareto.ParetoFront(similar=util.similar, similar_fit=util.similar_fit)

        population = pop

        stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, population, writer, csvfile, halloffame, meta_hof, -1, result_data=result_data)
        
        avg, bestMin, bestProd = util.averageFitness(population, cache)
        util.writeProgress(cache, -1, population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, result_data)
        util.graph_process(cache, "First")
        util.graph_corner_process(cache, last=False)

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

            valid_population = []
            invalid_idx = []
            for idx, individual in enumerate(population):
                if valid(individual, cache):
                    valid_population.append(individual)
                else:
                    invalid_idx.append(idx)
                    valid_population.append(closest_feasible(individual, cache))

            # Evaluate the individuals
            stalled, stallWarn, progressWarn = util.eval_population(toolbox, cache, valid_population, writer, csvfile, halloffame, meta_hof, gen, result_data=result_data)

            avg, bestMin, bestProd = util.averageFitness(valid_population, cache)
            util.writeProgress(cache, gen, valid_population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, sim_start, generation_start, result_data)
            util.graph_process(cache, gen)
            util.graph_corner_process(cache, last=False)

            #copy results back to normal population but penalize invalid members
            for idx, individual in enumerate(valid_population):
                if idx in invalid_idx:
                    dist = distance(population[idx], valid_population[idx])
                    new_fit = tuple(f - 1 * dist for f in valid_population[idx].fitness.values)
                    population[idx].fitness.values = new_fit
                else:
                    population[idx].fitness.values = valid_population[idx].fitness.values

            # Update the strategy with the evaluated individuals
            toolbox.update(population)
        
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)

            if avg >= settings.get('stopAverage', 1.0) or bestMin >= settings.get('stopBest', 1.0) or stalled:
                util.finish(cache)
                util.graph_corner_process(cache, last=True)
                return halloffame

    util.finish(cache)
    util.graph_corner_process(cache, last=True)
    return halloffame
