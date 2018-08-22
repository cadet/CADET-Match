import random
import pickle
import util
import numpy
from pathlib import Path
import h5py
#import grad
import time
import csv

from deap import algorithms

import checkpoint_algorithms

import emcee
import SALib.sample.sobol_sequence

import matplotlib
matplotlib.use('Agg')
import corner
import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import evo
import cache

import numpy as np
import pareto
import modEnsemble

import pickle
import scoop

name = "MCMC"

class Container:
    def __init__(self, minVar, maxVar):
        self.set(minVar, maxVar)

    def set(self, minVar, maxVar):
        self.minVar = minVar
        self.maxVar = maxVar
        self.minLogVar = numpy.log(minVar)
        self.maxLogVar = numpy.log(maxVar)

container = Container(1e-10, 1e-1)

def log_prior(theta, cache):
    # Create flat distributions.
    individual = theta[:-1]
    error = theta[-1]
    lower_bound = numpy.array(cache.MIN_VALUE)
    upper_bound = numpy.array(cache.MAX_VALUE)
    if numpy.all(individual >= lower_bound) and numpy.all(individual <= upper_bound) and 0 <= error <= 1:
    #if numpy.all(individual >= lower_bound) and numpy.all(individual <= upper_bound):
        return 0.0
    else:
        return -numpy.inf

def log_likelihood(theta, json_path):
    individual = theta[:-1]
    scores, csv_record, results = evo.fitness(individual, json_path)
    #error = theta[-1]
    error = numpy.exp((container.maxLogVar - container.minLogVar) * theta[-1] + container.minLogVar)
    count = sum([i['error_count'] for i in results.values()])
    sse = sum([i['error'] for i in results.values()])

    if json_path != cache.cache.json_path:
        cache.cache.setup(json_path, False)

    if cache.cache.scoreMCMC == 'score':
        score = -0.5 * (len(scores) * np.log(2 * numpy.pi * error ** 2) + sum([1.0 - i for i in scores]) / (error ** 2) )

    if cache.cache.scoreMCMC == 'sse':
        #sse
        score = -0.5 * (count * np.log(2 * numpy.pi * error ** 2) + sse / (error ** 2) )

    if cache.cache.scoreMCMC == 'min':
        #min
        min_score = min(scores)
        score = -0.5 * (1.0 * np.log(2 * numpy.pi * error ** 2) + (1.0 - min_score) / (error ** 2) )

    if cache.cache.scoreMCMC == 'product':
        #prod
        prod_score = numpy.product(scores)
        score = -0.5 * (1.0 * np.log(2 * numpy.pi * error ** 2) + (1.0 - prod_score) / (error ** 2) )

    return score, scores, csv_record, results 

def log_posterior(theta, json_path):
    if json_path != cache.cache.json_path:
        cache.cache.setup(json_path)

    lp = log_prior(theta, cache.cache)
    # only compute model if likelihood of the prior is not - infinity
    if not numpy.isfinite(lp):
        return -numpy.inf, None, None, None, None
    try:
        ll, scores, csv_record, results = log_likelihood(theta, json_path)
        return lp + ll, theta, scores, csv_record, results
    except:
        # if model does not converge:
        return -numpy.inf, None, None, None, None

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()

    if cache.scoreMCMC != 'sse':
        container.set(1e-4, 1e4)

    parameters = len(cache.MIN_VALUE) + 1
    populationSize = parameters * cache.settings['population']

    #Population must be even
    populationSize = populationSize + populationSize % 2  

    sobol = SALib.sample.sobol_sequence.sample(populationSize, parameters)

    #correct the last column to be in our error range
    #sobol[:,-1] = (maxLogVar - minLogVar) * sobol[:,-1] + minLogVar

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        sampler = emcee.EnsembleSampler(populationSize, parameters, log_posterior, args=[cache.json_path], pool=cache.toolbox)
        emcee.EnsembleSampler._get_lnprob = _get_lnprob

        training = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}}
        halloffame = pareto.DummyFront(similar=util.similar)
        meta_hof = pareto.ParetoFront(similar=util.similar)
        grad_hof = pareto.DummyFront(similar=util.similar)

        def local(results):
            return process(cache, halloffame, meta_hof, grad_hof, training, results, writer, csvfile, sampler)
        sampler.process = local

        converge = np.random.rand(50)
        burn_seq = []
        chain_seq = []

        checkpointFile = Path(cache.settings['resultsDirMisc'], cache.settings['checkpointFile'])

        checkpoint = getCheckPoint(checkpointFile,cache)

        if checkpoint['state'] == 'burn_in':
            tol = 5e-4
            count = checkpoint['length_burn'] - checkpoint['idx_burn']
            start_time = time.time()
            for idx, (p, ln_prob, random_state) in enumerate(sampler.sample(checkpoint['p_burn'], checkpoint['ln_prob_burn'],
                                    checkpoint['rstate_burn'], iterations=count ), start=checkpoint['idx_burn']):
                scoop.logger.info("%s burn step time %.4f", idx, time.time() - start_time)
                accept = np.mean(sampler.acceptance_fraction)
                burn_seq.append(accept)
                converge[:-1] = converge[1:]
                converge[-1] = accept
                writeMCMC(cache, sampler, burn_seq, chain_seq, idx)
                scoop.logger.info('idx: %s std: %s mean: %s converge: %s', idx, np.std(converge), np.mean(converge), np.std(converge)/tol)

                checkpoint['p_burn'] = p
                checkpoint['ln_prob_burn'] = ln_prob
                checkpoint['rstate_burn'] = random_state
                checkpoint['idx_burn'] = idx+1

                with checkpointFile.open('wb')as cp_file:
                    pickle.dump(checkpoint, cp_file)

                if np.std(converge) < tol:
                    scoop.logger.info("burn in completed at iteration %s", idx)
                    checkpoint['state'] = 'chain'
                    checkpoint['p_chain'] = p
                    checkpoint['ln_prob_chain'] = None
                    checkpoint['rstate_chain'] = None
                    sampler.reset()
                    break
                scoop.logger.info("%s burn process time %.4f", idx, time.time() - start_time)
                start_time = time.time()
            
        if checkpoint['state'] == 'chain':
            checkInterval = 25
            mult = 500
            start_time = time.time()
            count = checkpoint['length_chain'] - checkpoint['idx_chain']
            for idx, (p, ln_prob, random_state) in enumerate(sampler.sample(checkpoint['p_chain'], checkpoint['ln_prob_chain'],
                                    checkpoint['rstate_chain'], iterations=count ), start=checkpoint['idx_chain']):
                scoop.logger.info("%s chain step time %.4f", idx, time.time() - start_time)
                accept = np.mean(sampler.acceptance_fraction)
                chain_seq.append(accept)
                writeMCMC(cache, sampler, burn_seq, chain_seq, idx)

                checkpoint['p_chain'] = p
                checkpoint['ln_prob_chain'] = ln_prob
                checkpoint['rstate_chain'] = random_state
                checkpoint['idx_chain'] = idx+1

                with checkpointFile.open('wb')as cp_file:
                    pickle.dump(checkpoint, cp_file)

                if idx % checkInterval == 0:  
                    tau = autocorr_new(sampler.chain[:, :idx, 0].T)
                    scoop.logger.info("Mean acceptance fraction: %s %0.3f tau: %s", idx, accept, tau)
                    if idx > (mult * tau):
                        scoop.logger.info("we have run long enough and can quit %s", idx)
                        break
                scoop.logger.info("%s chain process time %.4f", idx, time.time() - start_time)
                start_time = time.time()

    chain = sampler.chain
    chain = chain[:, :idx, :]
    chain_shape = chain.shape
    chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])
                
    plotTube(cache, chain)
    util.finish(cache)
    return numpy.mean(chain, 0)

def getCheckPoint(checkpointFile, cache):
    if checkpointFile.exists():
        with checkpointFile.open('rb') as cp_file:
            checkpoint = pickle.load(cp_file)
    else:
        parameters = len(cache.MIN_VALUE) + 1
        populationSize = parameters * cache.settings['population']

        #Population must be even
        populationSize = populationSize + populationSize % 2  

        checkpoint = {}
        checkpoint['state'] = 'burn_in'
        checkpoint['p_burn'] = SALib.sample.sobol_sequence.sample(populationSize, parameters)
        checkpoint['ln_prob_burn'] = None
        checkpoint['rstate_burn'] = None
        checkpoint['idx_burn'] = 0
        checkpoint['length_burn'] = cache.settings.get('burnIn', 10000)

        checkpoint['p_chain'] = None
        checkpoint['ln_prob_chain'] = None
        checkpoint['rstate_chain'] = None
        checkpoint['idx_chain'] = 0
        checkpoint['length_chain'] = cache.settings.get('chainLength', 10000)
    return checkpoint

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

def process(cache, halloffame, meta_hof, grad_hof, training, results, writer, csv_file, sampler):
    if 'gen' not in process.__dict__:
        process.gen = 0

    if 'sim_start' not in process.__dict__:
        process.sim_start = time.time()

    if 'generation_start' not in process.__dict__:
        process.generation_start = time.time()

    scoop.logger.info("Mean acceptance fraction: %0.3f", numpy.mean(sampler.acceptance_fraction))

    csv_lines = []
    meta_csv_lines = []

    population = []
    fitnesses = []
    for sse, theta, fit, csv_line, result in results:
        if result is not None:
            parameters = theta
            fitnesses.append( (fit, csv_line, result) )

            ind = cache.toolbox.individual_guess(parameters)
            population.append(ind)

    process_time = time.time()
    total_time = time.time()
    stalled, stallWarn, progressWarn = util.process_population(cache.toolbox, cache, population, 
                                                          fitnesses, writer, csv_file, 
                                                          halloffame, meta_hof, process.gen, training)
    process_time = time.time() - process_time
    
    avg, bestMin, bestProd = util.averageFitness(population, cache)

    write_time = time.time()
    util.writeProgress(cache, process.gen, population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, 
                       process.sim_start, process.generation_start, training)
    write_time = time.time() - write_time

    graph_time = time.time()
    util.graph_process(cache, process.gen)
    graph_time = time.time() - graph_time

    total_time = time.time() - total_time

    scoop.logger.info("total time %.4f \tprocessing time %.4f \twrite time %.4f \tgraph_time %.4f", total_time, process_time, write_time, graph_time)

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
    simulation_time = time.time()
    results = list(M(self.lnprobfn, [p[i] for i in range(len(p))]))
    simulation_time = time.time() - simulation_time
    scoop.logger.info("Simulation time mcmc %.4f for %s items", simulation_time, len(results))
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
        bad_pars = []
        for pars in p[np.isnan(lnprob)]:
            bad_pars.append(pars)
        scoop.logger.info("NaN value of lnprob for parameters: %s", bad_pars)

        # Finally raise exception.
        raise ValueError("lnprob returned NaN.")

    return lnprob, blob

#auto correlation support functions

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def writeMCMC(cache, sampler, burn_seq, chain_seq, idx):
    "write out the mcmc data so it can be plotted"
    miscDir = Path(cache.settings['resultsDirMisc'])
    mcmc_h5 = miscDir / "mcmc.h5"

    chain = sampler.chain
    chain = chain[:, :idx+1, :]
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])

    chain_transform = numpy.array(chain)
    for walker in range(chain_shape[0]):
        for position in range(chain_shape[1]):
            chain_transform[walker, position,:-1] = util.convert_individual(chain_transform[walker, position, :-1], cache)
    chain_transform[:,:,-1] = numpy.exp((container.maxLogVar - container.minLogVar) * chain_transform[:,:,-1] + container.minLogVar)

    flat_chain_transform = chain_transform.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])

    with h5py.File(mcmc_h5, 'w') as hf:
        #if we don't have a file yet then we have to be doing burn in so no point in checking
        
        if burn_seq:
            data = numpy.array(burn_seq).reshape(-1, 1)
            hf.create_dataset("burn_in_acceptance", data=data, compression="gzip")
        
        if chain_seq:
            data = numpy.array(chain_seq).reshape(-1, 1)   
            hf.create_dataset("mcmc_acceptance", data=data, compression="gzip")
                
        hf.create_dataset("full_chain", data=chain, compression="gzip")
        hf.create_dataset("full_chain_transform", data=chain_transform, compression="gzip")

        hf.create_dataset("flat_chain", data=flat_chain, compression="gzip")
        hf.create_dataset("flat_chain_transform", data=flat_chain_transform, compression="gzip")

def processChainForPlots(cache, chain):
    mcmc_selected, mcmc_selected_transformed, mcmc_selected_score, results, times = genRandomChoice(cache, chain)

    writeSelected(cache, mcmc_selected, mcmc_selected_transformed, mcmc_selected_score)

    results, combinations = processResultsForPlotting(results, times)

    return results, combinations

def processResultsForPlotting(results, times):
    for expName, units in list(results.items()):
        for unitName, unit in list(units.items()):
            for compIdx, compValue in list(unit.items()):
                data = numpy.array(compValue)
                results[expName][unitName][compIdx] = {}
                results[expName][unitName][compIdx]['data'] = data
                results[expName][unitName][compIdx]["mean"] = numpy.mean(data, 0)
                results[expName][unitName][compIdx]["std"] = numpy.std(data, 0)
                results[expName][unitName][compIdx]["min"] = numpy.min(data, 0)
                results[expName][unitName][compIdx]["max"] = numpy.max(data, 0)
                results[expName][unitName][compIdx]['time'] = times[expName]

    combinations = {}
    for expName, units in results.items():
        for unitName, unit in list(units.items()):
            data = numpy.zeros(unit[0]['data'].shape)
            times = unit[0]['time']
            for compIdx, compValue in list(unit.items()):
                data = data + compValue['data']
            temp = {}
            temp['data'] = data
            temp['time'] = times
            temp["mean"] = numpy.mean(data, 0)
            temp["std"] = numpy.std(data, 0)
            temp["min"] = numpy.min(data, 0)
            temp["max"] = numpy.max(data, 0)
            comb_name = '%s_%s' % (expName, unitName)
            combinations[comb_name] = temp
    return results, combinations

def genRandomChoice(cache, chain, size=500):
    if len(chain) > size:
        indexes = np.random.choice(chain.shape[0], size, replace=False)
        chain = chain[indexes]

    individuals = []

    for idx in range(len(chain)):
        individuals.append(chain[idx,0:-1])

    fitnesses = cache.toolbox.map(cache.toolbox.evaluate, individuals)

    results = {}
    times = {}

    mcmc_selected = []
    mcmc_selected_transformed = []
    mcmc_selected_score = []

    for (fit, csv_line, result) in fitnesses:

        mcmc_selected_score.append(tuple(fit))
        for value in result.values():
            mcmc_selected_transformed.append(tuple(value['cadetValues']))
            mcmc_selected.append(tuple(value['individual']))
            break

        
        for key,value in result.items():
            sims = results.get(key, {})

            sim = value['simulation']

            outlets = util.findOutlets(sim)

            for outlet, ncomp in outlets:
                units = sims.get(outlet, {})

                for i in range(ncomp):
                    comps = units.get(i, [])
                    comps.append(sim.root.output.solution[outlet]["solution_inlet_comp_%03d" % i])
                    units[i] = comps

                sims[outlet] = units

                if key not in times:
                    times[key] = sim.root.output.solution.solution_times

            results[key] = sims

        util.cleanupProcess(result)

    mcmc_selected = numpy.array(mcmc_selected)
    mcmc_selected_transformed = numpy.array(mcmc_selected_transformed)
    mcmc_selected_score = numpy.array(mcmc_selected_score)

    #set the upperbound of find outliers to 100% since we don't need to get rid of good results only bad ones
    selected, bools = util.find_outliers(mcmc_selected, 10, 90)

    removeResultsOutliers(results, bools)
    
    return mcmc_selected[bools], mcmc_selected_transformed[bools], mcmc_selected_score[bools], results, times

def removeResultsOutliers(results, bools):
    for exp, units in results.items():
        for unitName, unit in units.items():
            for comp, data in unit.items():
                unit[comp] = np.array(data)[bools]

def writeSelected(cache, mcmc_selected, mcmc_selected_transformed, mcmc_selected_score):
    mcmc_h5 = Path(cache.settings['resultsDirMisc']) / "mcmc.h5"
    with h5py.File(mcmc_h5, 'a') as hf:
         hf.create_dataset("mcmc_selected", data=numpy.array(mcmc_selected), compression="gzip")
         hf.create_dataset("mcmc_selected_transformed", data=numpy.array(mcmc_selected_transformed), compression="gzip")
         hf.create_dataset("mcmc_selected_score", data=numpy.array(mcmc_selected_score), compression="gzip")

def plotTube(cache, chain):
    results, combinations = processChainForPlots(cache, chain)

    output_mcmc = cache.settings['resultsDirSpace'] / "mcmc"
    output_mcmc.mkdir(parents=True, exist_ok=True)

    for expName,value in combinations.items():
        plot_mcmc(output_mcmc, value, expName, "combine")


    for exp, units in results.items():
        for unitName, unit in units.items():
            for comp, data in unit.items():
                expName = '%s_%s' % (exp, unitName)
                plot_mcmc(output_mcmc, data, expName, comp)

def plot_mcmc(output_mcmc, value, expName, name):
    data = value['data']
    times = value['time']
    mean = value["mean"]
    std = value["std"]
    minValues = value["min"]
    maxValues = value["max"]

    plt.plot(times, mean)
    plt.fill_between(times, mean - std, mean + std,
                color='green', alpha=0.2)
    plt.fill_between(times, minValues, maxValues,
                color='red', alpha=0.2)
    plt.savefig(str(output_mcmc / ("%s_%s.png" % (expName, name) ) ), bbox_inches='tight')
    plt.close()

    plt.plot(times, data.transpose())
    plt.savefig(str(output_mcmc / ("%s_%s_lines.png" % (expName, name) ) ), bbox_inches='tight')
    plt.close()

