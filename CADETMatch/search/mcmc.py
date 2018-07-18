import random
import pickle
import util
import numpy
from pathlib import Path
#import grad
import time
import csv

from deap import algorithms

import checkpoint_algorithms

import emcee
import SALib.sample.sobol_sequence
import corner

import matplotlib.mlab as mlab

import evo
import cache

import numpy as np
import pareto

name = "MCMC"

def log_prior(theta, cache):
    # Create flat distributions.
    individual = theta[:-1]
    theta = numpy.array(theta)
    lower_bound = numpy.array(cache.MIN_VALUE)
    upper_bound = numpy.array(cache.MAX_VALUE)
    if numpy.all(individual >= lower_bound) and numpy.all(individual <= upper_bound):
        return 0.0
    else:
        return -numpy.inf

def log_likelihood(theta, json_path):
    individual = theta[:-1]
    scores, csv_record, results = evo.fitness(individual, json_path)
    error = theta[-1]
    return -0.5 * (numpy.sum(numpy.log(2 * numpy.pi * error ** 2)) + float(csv_record[-1]) / error ** 2), scores, csv_record, results

def log_posterior(theta, json_path):
    if json_path != cache.cache.json_path:
        cache.cache.setup(json_path)

    lp = log_prior(theta, cache.cache)
    # only compute model if likelihood of the prior is not - infinity
    if not numpy.isfinite(lp):
        return -numpy.inf, None, None, None
    #try:
    ll, scores, csv_record, results = log_likelihood(theta, json_path)
    return lp + ll, scores, csv_record, results
    #except:
    #    # if model does not converge:
    #    return -numpy.inf, None, None, None

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(cache.MIN_VALUE) + 1
    populationSize = parameters * cache.settings['population']
    sobol = SALib.sample.sobol_sequence.sample(populationSize, parameters)

    selected = sobol[:,-1] == 0
    sobol[selected,-1] += 1e-4

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        sampler = emcee.EnsembleSampler(populationSize, parameters, log_posterior, args=[cache.json_path], pool=cache.toolbox)
        emcee.EnsembleSampler._get_lnprob = _get_lnprob

        training = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}}
        halloffame = pareto.ParetoFront(similar=util.similar)
        meta_hof = pareto.ParetoFront(similar=util.similar)
        grad_hof = pareto.ParetoFront(similar=util.similar)

        def local(results):
            return process(cache, halloffame, meta_hof, grad_hof, training, results, writer, csvfile)
        sampler.process = local

        state = sampler.run_mcmc(sobol, cache.settings['burnIn'])
        sampler.reset()
        sampler.run_mcmc(state[0], cache.settings['chainLength'])

    fig = corner.corner(sampler.flatchain)
    out_dir = cache.settings['resultsDirBase']
    fig.savefig(str(out_dir / "corner.png"), bbox_inches='tight')

    chain = sampler.flatchain
    smallest = numpy.argmin(chain[:,-1])
    return chain[smallest]

def setupDEAP(cache, fitness, grad_fitness, grad_search, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)

    cache.toolbox.register("individual", util.generateIndividual, creator.Individual,
        len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE, cache)

    if cache.sobolGeneration:
        cache.toolbox.register("population", util.sobolGenerator, creator.Individual, cache)
    else:
        cache.toolbox.register("population", tools.initRepeat, list, cache.toolbox.individual)
    cache.toolbox.register("randomPopulation", tools.initRepeat, list, cache.toolbox.individual)

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual, cache)

    cache.toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=5.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE)

    if cache.adaptive:
        cache.toolbox.register("mutate", util.mutationBoundedAdaptive, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
        cache.toolbox.register("force_mutate", util.mutationBoundedAdaptive, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
    else:
        cache.toolbox.register("mutate", tools.mutPolynomialBounded, eta=2.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))
        cache.toolbox.register("force_mutate", tools.mutPolynomialBounded, eta=2.0, low=cache.MIN_VALUE, up=cache.MAX_VALUE, indpb=1.0/len(cache.MIN_VALUE))

    cache.toolbox.register("select", tools.selNSGA2)
    cache.toolbox.register("evaluate", fitness, json_path=cache.json_path)
    cache.toolbox.register("evaluate_grad", grad_fitness, json_path=cache.json_path)
    cache.toolbox.register('grad_search', grad_search)

    cache.toolbox.register('map', map_function)

def process(cache, halloffame, meta_hof, grad_hof, training, results, writer, csv_file):
    if 'gen' not in process.__dict__:
        process.gen = 0

    if 'sim_start' not in process.__dict__:
        process.sim_start = time.time()

    if 'generation_start' not in process.__dict__:
        process.generation_start = time.time()

    csv_lines = []
    meta_csv_lines = []

    population = []
    fitnesses = []
    for sse, fit, csv_line, result in results:
        if result is not None:
            for obj in result.values():
                parameters = obj['cadetValuesKEQ']
                break
            fitnesses.append( (fit, csv_line, result) )

            ind = cache.toolbox.individual_guess(parameters)
            population.append(ind)

    stalled, stallWarn, progressWarn = util.process_population(cache.toolbox, cache, population, 
                                                          fitnesses, writer, csv_file, 
                                                          halloffame, meta_hof, process.gen, training)
    
    avg, bestMin, bestProd = util.averageFitness(population, cache)
    util.writeProgress(cache, process.gen, population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, 
                       process.sim_start, process.generation_start, training)
    util.graph_process(cache, process.gen)

    process.gen += 1
    process.generation_start = time.time()
    return [i[0] for i in results]

def _get_lnprob(self, pos=None):
    """
    Calculate the vector of log-probability for the walkers.

    :param pos: (optional)
        The position vector in parameter space where the probability
        should be calculated. This defaults to the current position
        unless a different one is provided.

    This method returns:

    * ``lnprob`` - A vector of log-probabilities with one entry for each
        walker in this sub-ensemble.

    * ``blob`` - The list of meta data returned by the ``lnpostfn`` at
        this position or ``None`` if nothing was returned.

    """
    if pos is None:
        p = self.pos
    else:
        p = pos

    # Check that the parameters are in physical ranges.
    if np.any(np.isinf(p)):
        raise ValueError("At least one parameter value was infinite.")
    if np.any(np.isnan(p)):
        raise ValueError("At least one parameter value was NaN.")

    # If the `pool` property of the sampler has been set (i.e. we want
    # to use `multiprocessing`), use the `pool`'s map method. Otherwise,
    # just use the built-in `map` function.
    if self.pool is not None:
        M = self.pool.map
    else:
        M = map

    # sort the tasks according to (user-defined) some runtime guess
    if self.runtime_sortingfn is not None:
        p, idx = self.runtime_sortingfn(p)

    # Run the log-probability calculations (optionally in parallel).
    results = list(M(self.lnprobfn, [p[i] for i in range(len(p))]))
    results = self.process(results)

    try:
        lnprob = np.array([float(l[0]) for l in results])
        blob = [l[1] for l in results]
    except (IndexError, TypeError):
        lnprob = np.array([float(l) for l in results])
        blob = None

    # sort it back according to the original order - get the same
    # chain irrespective of the runtime sorting fn
    if self.runtime_sortingfn is not None:
        orig_idx = np.argsort(idx)
        lnprob = lnprob[orig_idx]
        p = [p[i] for i in orig_idx]
        if blob is not None:
            blob = [blob[i] for i in orig_idx]

    # Check for lnprob returning NaN.
    if np.any(np.isnan(lnprob)):
        # Print some debugging stuff.
        print("NaN value of lnprob for parameters: ")
        for pars in p[np.isnan(lnprob)]:
            print(pars)

        # Finally raise exception.
        raise ValueError("lnprob returned NaN.")

    return lnprob, blob