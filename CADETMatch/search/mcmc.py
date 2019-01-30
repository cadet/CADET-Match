import random
import pickle
import util
import numpy
from pathlib import Path
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
#import grad
import time
import csv

import emcee
import SALib.sample.sobol_sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import evo
import cache

import pareto

import scoop
import pandas
import array

import kde_generator

name = "MCMC"

log2 = numpy.log(2)

def log_likelihood(theta, json_path, kde_scores, kde_bw):
    if json_path != cache.cache.json_path:
        cache.cache.setup(json_path, False)
        cache.cache.roundScores = None
        cache.cache.roundParameters = None

    if 'kde' not in log_likelihood.__dict__:
        kde, pca, scaler = kde_generator.getKDE(cache.cache, kde_scores, kde_bw)
        log_likelihood.kde = kde
        log_likelihood.pca = pca
        log_likelihood.scaler = scaler
    
    individual = theta

    scores, csv_record, results = evo.fitness(individual, json_path)

    scores_shape = numpy.array(scores).reshape(1, -1)

    score_scaler = log_likelihood.scaler.transform(scores_shape)

    score = log_likelihood.kde.score_samples(score_scaler) + log2  #*2 is from mirroring and we need to double the probability to get back to the normalized distribution

    return score, scores, csv_record, results 

def log_posterior(theta, json_path, kde_scores, kde_bw):
    if json_path != cache.cache.json_path:
        cache.cache.setup(json_path)

    #try:
    ll, scores, csv_record, results = log_likelihood(theta, json_path, kde_scores, kde_bw)
    return ll, theta, scores, csv_record, results
    #except:
    #    # if model does not converge:
    #    return -numpy.inf, None, None, None, None

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()

    cache.roundScores = None
    cache.roundParameters = None

    parameters = len(cache.MIN_VALUE)
    
    populationSize = parameters * cache.settings['MCMCpopulation']

    #Population must be even
    populationSize = populationSize + populationSize % 2  

    sobol = SALib.sample.sobol_sequence.sample(populationSize, parameters)
    
    kde_scores, kde_bw = kde_generator.generate_data(cache)

    kde, pca, scaler = kde_generator.getKDE(cache, kde_scores, kde_bw)

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        sampler = emcee.EnsembleSampler(populationSize, parameters, log_posterior, args=[cache.json_path, kde_scores, kde_bw], pool=cache.toolbox, a=2.0)
        emcee.EnsembleSampler._get_lnprob = _get_lnprob
        emcee.EnsembleSampler._propose_stretch = _propose_stretch


        result_data = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}, 'input_transform':[], 'input_transform_extended':[], 'strategy':[], 
                   'mean':[], 'confidence':[], 'mcmc_score':[]}
        halloffame = pareto.DummyFront(similar=util.similar)
        meta_hof = pareto.ParetoFront(similar=util.similar)
        grad_hof = pareto.DummyFront(similar=util.similar)

        def local(results):
            return process(cache, halloffame, meta_hof, grad_hof, result_data, results, writer, csvfile, sampler)
        sampler.process = local

        converge = numpy.random.rand(50)
        burn_seq = []
        chain_seq = []

        checkpointFile = Path(cache.settings['resultsDirMisc'], cache.settings['checkpointFile'])

        checkpoint = getCheckPoint(checkpointFile,cache)

        if checkpoint['state'] == 'burn_in':
            tol = 5e-4
            convergence_check_interval = 50
            count = checkpoint['length_burn'] - checkpoint['idx_burn']
            for idx, (p, ln_prob, random_state) in enumerate(sampler.sample(checkpoint['p_burn'], checkpoint['ln_prob_burn'],
                                    checkpoint['rstate_burn'], iterations=count ), start=checkpoint['idx_burn']):
                accept = numpy.mean(sampler.acceptance_fraction)
                burn_seq.append(accept)
                converge[:-1] = converge[1:]
                converge[-1] = accept
                writeMCMC(cache, sampler, burn_seq, chain_seq, idx, parameters)
                scoop.logger.info('idx: %s std: %s mean: %s converge: %s', idx, numpy.std(converge), numpy.mean(converge), numpy.std(converge)/tol)

                checkpoint['p_burn'] = p
                checkpoint['ln_prob_burn'] = ln_prob
                checkpoint['rstate_burn'] = random_state
                checkpoint['idx_burn'] = idx+1

                with checkpointFile.open('wb')as cp_file:
                    pickle.dump(checkpoint, cp_file)

                if numpy.std(converge) < tol:
                    scoop.logger.info("burn in completed at iteration %s", idx)
                    checkpoint['state'] = 'chain'
                    checkpoint['p_chain'] = p
                    checkpoint['ln_prob_chain'] = None
                    checkpoint['rstate_chain'] = None
                    sampler.reset()
                    break
            
        if checkpoint['state'] == 'chain':
            checkInterval = 25
            mult = cache.MCMCTauMult
            count = checkpoint['length_chain'] - checkpoint['idx_chain']
            for idx, (p, ln_prob, random_state) in enumerate(sampler.sample(checkpoint['p_chain'], checkpoint['ln_prob_chain'],
                                    checkpoint['rstate_chain'], iterations=count ), start=checkpoint['idx_chain']):
                accept = numpy.mean(sampler.acceptance_fraction)
                chain_seq.append(accept)
                writeMCMC(cache, sampler, burn_seq, chain_seq, idx, parameters)

                checkpoint['p_chain'] = p
                checkpoint['ln_prob_chain'] = ln_prob
                checkpoint['rstate_chain'] = random_state
                checkpoint['idx_chain'] = idx+1

                with checkpointFile.open('wb')as cp_file:
                    pickle.dump(checkpoint, cp_file)

                if idx % checkInterval == 0 and idx >= 200:  
                    tau = autocorr_new(sampler.chain[:, :idx, 0].T)
                    scoop.logger.info("Mean acceptance fraction: %s %0.3f tau: %s", idx, accept, tau)
                    if idx > (mult * tau):
                        scoop.logger.info("we have run long enough and can quit %s", idx)
                        break

    chain = sampler.chain
    chain = chain[:, :idx, :]
    chain_shape = chain.shape
    chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])
                
    plotTube(cache, chain, kde, pca, scaler)
    util.finish(cache)
    return numpy.mean(chain, 0)

def getCheckPoint(checkpointFile, cache):
    if checkpointFile.exists():
        with checkpointFile.open('rb') as cp_file:
            checkpoint = pickle.load(cp_file)
    else:
        parameters = len(cache.MIN_VALUE)
        
        populationSize = parameters * cache.settings['MCMCpopulation']

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
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None, mean=None, confidence=None)

    creator.create("FitnessMaxMeta", base.Fitness, weights=[1.0] * 4)
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

def process(cache, halloffame, meta_hof, grad_hof, result_data, results, writer, csv_file, sampler):
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
            result_data['mcmc_score'].append(sse)

    stalled, stallWarn, progressWarn = util.process_population(cache.toolbox, cache, population, 
                                                          fitnesses, writer, csv_file, 
                                                          halloffame, meta_hof, process.gen, result_data)
    
    avg, bestMin, bestProd = util.averageFitness(population, cache)

    util.writeProgress(cache, process.gen, population, halloffame, meta_hof, grad_hof, avg, bestMin, bestProd, 
                       process.sim_start, process.generation_start, result_data)

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
    if numpy.any(numpy.isinf(p)):
        raise ValueError("At least one parameter value was infinite.")
    if numpy.any(numpy.isnan(p)):
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
        lnprob = numpy.array([float(l[0]) for l in results])
        blob = [l[1] for l in results]
    except (IndexError, TypeError):
        lnprob = numpy.array([float(l) for l in results])
        blob = None

    # sort it back according to the original order - get the same
    # chain irrespective of the runtime sorting fn
    if self.runtime_sortingfn is not None:
        orig_idx = numpy.argsort(idx)
        lnprob = lnprob[orig_idx]
        p = [p[i] for i in orig_idx]
        if blob is not None:
            blob = [blob[i] for i in orig_idx]

    # Check for lnprob returning NaN.
    if numpy.any(numpy.isnan(lnprob)):
        # Print some debugging stuff.
        bad_pars = []
        for pars in p[numpy.isnan(lnprob)]:
            bad_pars.append(pars)
        scoop.logger.info("NaN value of lnprob for parameters: %s", bad_pars)

        # Finally raise exception.
        raise ValueError("lnprob returned NaN.")

    return lnprob, blob

#auto correlation support functions

def autocorr_new(y, c=5.0):
    f = numpy.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0*numpy.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = numpy.arange(len(taus)) < c * taus
    if numpy.any(m):
        return numpy.argmin(m)
    return len(taus) - 1

def autocorr_func_1d(x, norm=True):
    x = numpy.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = numpy.fft.fft(x - numpy.mean(x), n=2*n)
    acf = numpy.fft.ifft(f * numpy.conjugate(f))[:len(x)].real
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

def writeMCMC(cache, sampler, burn_seq, chain_seq, idx, parameters):
    "write out the mcmc data so it can be plotted"
    mcmcDir = Path(cache.settings['resultsDirMCMC'])
    mcmc_h5 = mcmcDir / "mcmc.h5"

    chain = sampler.chain
    chain = chain[:, :idx+1, :]
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])

    chain_transform = numpy.array(chain)
    for walker in range(chain_shape[0]):
        for position in range(chain_shape[1]):
            chain_transform[walker, position,:] = util.convert_individual(chain_transform[walker, position, :], cache)[0]

    flat_chain_transform = chain_transform.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])

    flat_interval = interval(flat_chain, cache)
    flat_interval_transform = interval(flat_chain_transform, cache)

    flat_interval.to_csv(mcmcDir / "percentile.csv")
    flat_interval_transform.to_csv(mcmcDir / "percentile_transform.csv")

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

def processChainForPlots(cache, chain, kde, pca, scaler):
    mcmc_selected, mcmc_selected_transformed, mcmc_selected_score, results, times, mcmc_score = genRandomChoice(cache, chain, kde, pca, scaler)

    writeSelected(cache, mcmc_selected, mcmc_selected_transformed, mcmc_selected_score, mcmc_score)

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

def genRandomChoice(cache, chain, kde, pca, scaler):
    size = 5000
    if len(chain) > size:
        indexes = numpy.random.choice(chain.shape[0], size, replace=False)
        chain = chain[indexes]

    individuals = []

    for idx in range(len(chain)):
        individuals.append(chain[idx,:])

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

    mcmc_selected = numpy.array(mcmc_selected)
    mcmc_selected_transformed = numpy.array(mcmc_selected_transformed)
    mcmc_selected_score = numpy.array(mcmc_selected_score)

    mcmc_score = kde.score_samples(scaler.transform(mcmc_selected_score)) + log2

    #set the upperbound of find outliers to 100% since we don't need to get rid of good results only bad ones
    
    #selected, bools = util.find_outliers(mcmc_selected, 10, 90)

    #removeResultsOutliers(results, bools)
    
    #return mcmc_selected[bools], mcmc_selected_transformed[bools], mcmc_selected_score[bools], results, times

    return mcmc_selected, mcmc_selected_transformed, mcmc_selected_score, results, times, mcmc_score

def removeResultsOutliers(results, bools):
    for exp, units in results.items():
        for unitName, unit in units.items():
            for comp, data in unit.items():
                unit[comp] = numpy.array(data)[bools]

def writeSelected(cache, mcmc_selected, mcmc_selected_transformed, mcmc_selected_score, mcmc_score):
    mcmc_h5 = Path(cache.settings['resultsDirMCMC']) / "mcmc.h5"
    with h5py.File(mcmc_h5, 'a') as hf:
        hf.create_dataset("mcmc_selected", data=numpy.array(mcmc_selected), compression="gzip")
        hf.create_dataset("mcmc_selected_transformed", data=numpy.array(mcmc_selected_transformed), compression="gzip")
        hf.create_dataset("mcmc_selected_score", data=numpy.array(mcmc_selected_score), compression="gzip")
        hf.create_dataset("mcmc_score", data=numpy.array(mcmc_score), compression="gzip")

def plotTube(cache, chain, kde, pca, scaler):
    results, combinations = processChainForPlots(cache, chain, kde, pca, scaler)

    output_mcmc = cache.settings['resultsDirSpace'] / "mcmc"
    output_mcmc.mkdir(parents=True, exist_ok=True)

    mcmc_h5 = output_mcmc / "mcmc_plots.h5"
    with h5py.File(mcmc_h5, 'a') as hf:

        for expName,value in combinations.items():
            plot_mcmc(output_mcmc, value, expName, "combine")
            hf.create_dataset(expName, data=value['data'], compression="gzip")

        for exp, units in results.items():
            for unitName, unit in units.items():
                for comp, data in unit.items():
                    expName = '%s_%s' % (exp, unitName)
                    plot_mcmc(output_mcmc, data, expName, comp)
                    hf.create_dataset('%s_%s' % (expName, comp), data=data['data'], compression="gzip")

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

    row, col = data.shape
    alpha = row/1.3e6
    plt.plot(times, data.transpose(), 'g', alpha=alpha)
    plt.plot(times, mean, 'k')
    plt.savefig(str(output_mcmc / ("%s_%s_lines.png" % (expName, name) ) ), bbox_inches='tight')
    plt.close()

def interval(flat_chain, cache):
    #https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data

    mean = numpy.mean(flat_chain,0)

    percentile = numpy.percentile(flat_chain, [5, 10, 25, 50, 75, 90, 95], 0)

    data = numpy.vstack( (mean, percentile) ).transpose()

    data = util.roundParameter(data, cache)

    pd = pandas.DataFrame(data, columns = ['mean', '5', '10', '25', '50', '75', '90', '95'])
    pd.insert(0, 'name', cache.parameter_headers_actual)
    pd.set_index('name')
    return pd

def _propose_stretch(self, p0, p1, lnprob0):
    """
    Propose a new position for one sub-ensemble given the positions of
    another.

    :param p0:
        The positions from which to jump.

    :param p1:
        The positions of the other ensemble.

    :param lnprob0:
        The log-probabilities at ``p0``.

    This method returns:

    * ``q`` - The new proposed positions for the walkers in ``ensemble``.

    * ``newlnprob`` - The vector of log-probabilities at the positions
        given by ``q``.

    * ``accept`` - A vector of type ``bool`` indicating whether or not
        the proposed position for each walker should be accepted.

    * ``blob`` - The new meta data blobs or ``None`` if nothing was
        returned by ``lnprobfn``.

    """
    s = numpy.atleast_2d(p0)
    Ns = len(s)
    c = numpy.atleast_2d(p1)
    Nc = len(c)

    # Generate the vectors of random numbers that will produce the
    # proposal.
    zz = ((self.a - 1.) * self._random.rand(Ns) + 1) ** 2. / self.a
    rint = self._random.randint(Nc, size=(Ns,))

    # Calculate the proposed positions and the log-probability there.
    q = (c[rint] - zz[:, numpy.newaxis] * (c[rint] - s)) % 1
        
    newlnprob, blob = self._get_lnprob(q)

    # Decide whether or not the proposals should be accepted.
    lnpdiff = (self.dim - 1.) * numpy.log(zz) + newlnprob - lnprob0
    accept = (lnpdiff > numpy.log(self._random.rand(len(lnpdiff))))

    return q, newlnprob, accept, blob