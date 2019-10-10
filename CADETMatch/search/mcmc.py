import random
import pickle
import CADETMatch.util as util
import numpy
import scipy
from pathlib import Path
import time
import csv
import cadet

import emcee
import SALib.sample.sobol_sequence

import matplotlib
matplotlib.use('Agg')

from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import matplotlib.pyplot as plt
import subprocess
import sys

size = 20

plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title
plt.rc('figure', autolayout=True)

import CADETMatch.evo as evo
import CADETMatch.cache as cache

import CADETMatch.pareto as pareto

import scoop
import pandas
import array
import CADETMatch.mle as mle
import CADETMatch.autocorr as autocorr

import CADETMatch.kde_generator as kde_generator
from sklearn.neighbors.kde import KernelDensity

name = "MCMC"

from addict import Dict
import matplotlib.cm
cm_plot = matplotlib.cm.gist_rainbow

import joblib

def get_color(idx, max_colors, cmap):
    return cmap(1.*float(idx)/max_colors)

log2 = numpy.log(2)

saltIsotherms = {b'STERIC_MASS_ACTION', b'SELF_ASSOCIATION', b'MULTISTATE_STERIC_MASS_ACTION', 
                 b'SIMPLE_MULTISTATE_STERIC_MASS_ACTION', b'BI_STERIC_MASS_ACTION'}

def log_previous(cadetValues, kde_previous, kde_previous_scaler):
    #find the right values to use
    col = len(kde_previous_scaler.scale_)
    values = cadetValues[-col:]
    values_shape = numpy.array(values).reshape(1, -1)
    values_scaler = kde_previous_scaler.transform(values_shape)
    score = kde_previous.score_samples(values_scaler)
    return score

def log_likelihood(individual, json_path):
    if json_path != cache.cache.json_path:
        cache.cache.setup(json_path, False)

    kde_previous, kde_previous_scaler = kde_generator.getKDEPrevious(cache.cache)

    if 'kde' not in log_likelihood.__dict__:
        kde, kde_scaler = kde_generator.getKDE(cache.cache)
        log_likelihood.kde = kde
        log_likelihood.scaler = kde_scaler

    scores, csv_record, results = evo.fitness(individual, json_path)

    if results is None:
        return -numpy.inf, scores, csv_record, results

    if results is not None and kde_previous is not None:
        logPrevious = log_previous(individual, kde_previous, kde_previous_scaler)
    else:
        logPrevious = 0.0

    scores_shape = numpy.array(scores).reshape(1, -1)

    score_scaler = log_likelihood.scaler.transform(scores_shape)

    score_kde = log_likelihood.kde.score_samples(score_scaler)

    score = score_kde + log2 + logPrevious #*2 is from mirroring and we need to double the probability to get back to the normalized distribution

    return score, scores, csv_record, results 

def log_posterior(theta, json_path):
    if json_path != cache.cache.json_path:
        cache.cache.setup(json_path)

    #try:
    ll, scores, csv_record, results = log_likelihood(theta, json_path)
    if results is None:
        return -numpy.inf, None, None, None, None
    else:
        return ll, theta, scores, csv_record, results
    #except:
    #    # if model does not converge:
    #    return -numpy.inf, None, None, None, None

def addChain(*args):
    temp = [arg for arg in args if arg is not None]
    if len(temp) > 1:
        return numpy.concatenate( temp, axis=1)
    else:
        return numpy.array(temp[0])

def flatten(chain):
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])
    return flat_chain

def sampler_burn(cache, checkpoint, sampler, checkpointFile):
    burn_seq = checkpoint.get('burn_seq', [])
    chain_seq = checkpoint.get('chain_seq', [])
        
    train_chain = checkpoint.get('train_chain', None)
    run_chain = checkpoint.get('run_chain', None)

    train_chain_stat = checkpoint.get('train_chain_stat', None)
    run_chain_stat = checkpoint.get('run_chain_stat', None)

    converge = checkpoint.get('converge')    

    parameters = len(cache.MIN_VALUE)

    tol = 5e-4
    power = 0.0
    distance = 1.0
    distance_a = sampler.a
    stop_next = False
    finished = False

    generation = checkpoint['idx_burn']

    sampler.iterations = checkpoint['sampler_iterations']
    sampler.naccepted = checkpoint['sampler_naccepted']
    sampler.a = checkpoint['sampler_a']

    while not finished:
        p, ln_prob, random_state = next(sampler.sample(checkpoint['p_burn'], lnprob0=checkpoint['ln_prob_burn'], rstate0=checkpoint['rstate_burn'], iterations=1 ))

        accept = numpy.mean(sampler.acceptance_fraction)
        burn_seq.append(accept)
        converge[:-1] = converge[1:]
        converge[-1] = accept

        train_chain = addChain(train_chain, p[:, numpy.newaxis, :])

        train_chain_stat = addChain(train_chain_stat, numpy.percentile(flatten(train_chain), [5, 50, 95], 0)[:, numpy.newaxis, :])

        converge_real = converge[~numpy.isnan(converge)]
        scoop.logger.info('burn:  idx: %s accept: %.3g std: %.3g mean: %.3g converge: %.3g', generation, accept, 
                            numpy.std(converge_real), numpy.mean(converge_real), numpy.std(converge_real)/tol)

        generation += 1

        checkpoint['p_burn'] = p
        checkpoint['ln_prob_burn'] = ln_prob
        checkpoint['rstate_burn'] = random_state
        checkpoint['idx_burn'] = generation
        checkpoint['train_chain'] = train_chain
        checkpoint['burn_seq'] = burn_seq
        checkpoint['converge'] = converge
        checkpoint['sampler_iterations'] = sampler.iterations
        checkpoint['sampler_naccepted'] = sampler.naccepted
        checkpoint['train_chain_stat'] = train_chain_stat
        checkpoint['run_chain_stat'] = run_chain_stat
        checkpoint['sampler_a'] = sampler.a

        write_interval(cache.checkpointInterval, cache, checkpoint, checkpointFile, train_chain, run_chain, burn_seq, chain_seq, parameters, train_chain_stat, run_chain_stat, None)
        util.graph_corner_process(cache, last=False)

        if numpy.std(converge_real) < tol and len(converge) == len(converge_real):
            average_converge = numpy.mean(converge)
            if 0.16 < average_converge < 0.30:
                scoop.logger.info("burn in completed at iteration %s", generation)
                finished = True

            if stop_next is True:
                scoop.logger.info("burn in completed at iteration %s based on minimum distances", generation)
                finished = True

            if not finished:
                new_distance = numpy.abs(average_converge - 0.234)
                if new_distance < distance:
                    distance = new_distance
                    distance_a = sampler.a

                    scoop.logger.info("burn in acceptance is out of tolerance and alpha must be adjusted while burn in continues")
                    converge[:] = numpy.nan
                    prev_a = sampler.a
                    if average_converge > 0.3:
                        #a must be increased to decrease the acceptance rate (step size)
                        power += 1
                    else:
                        #a must be decreased to increase the acceptance rate (step size)
                        power -= 1
                    new_a = 1.0 + 2.0*power
                    sampler.a = new_a
                    sampler.reset()
                    scoop.logger.info('previous alpha: %s    new alpha: %s', prev_a, new_a)
                else:
                    sampler.a = distance_a
                    sampler.reset()
                    stop_next = True
 
    sampler.reset()

    checkpoint['sampler_iterations'] = sampler.iterations
    checkpoint['sampler_naccepted'] = sampler.naccepted
    checkpoint['state'] = 'chain'
    checkpoint['p_chain'] = p
    checkpoint['ln_prob_chain'] = None
    checkpoint['rstate_chain'] = None
    checkpoint['sampler_a'] = sampler.a

    write_interval(-1, cache, checkpoint, checkpointFile, train_chain, run_chain, burn_seq, chain_seq, parameters, train_chain_stat, run_chain_stat, None)
            

def sampler_run(cache, checkpoint, sampler, checkpointFile):
    burn_seq = checkpoint.get('burn_seq', [])
    chain_seq = checkpoint.get('chain_seq', [])
        
    train_chain = checkpoint.get('train_chain', None)
    run_chain = checkpoint.get('run_chain', None)

    train_chain_stat = checkpoint.get('train_chain_stat', None)
    run_chain_stat = checkpoint.get('run_chain_stat', None)

    checkInterval = 25

    parameters = len(cache.MIN_VALUE)

    finished = False

    generation = checkpoint['idx_chain']

    sampler.iterations = checkpoint['sampler_iterations']
    sampler.naccepted = checkpoint['sampler_naccepted']
    sampler.a = checkpoint['sampler_a']
    tau_percent = None

    while not finished:
        p, ln_prob, random_state = next(sampler.sample(checkpoint['p_chain'], lnprob0=checkpoint['ln_prob_chain'], rstate0=checkpoint['rstate_chain'], iterations=1 ))

        accept = numpy.mean(sampler.acceptance_fraction)
        chain_seq.append(accept)

        run_chain = addChain(run_chain, p[:, numpy.newaxis, :])

        run_chain_stat = addChain(run_chain_stat, numpy.percentile(flatten(run_chain), [5, 50, 95], 0)[:, numpy.newaxis, :])

        scoop.logger.info('run:  idx: %s accept: %.3f', generation, accept)
        
        generation += 1

        checkpoint['p_chain'] = p
        checkpoint['ln_prob_chain'] = ln_prob
        checkpoint['rstate_chain'] = random_state
        checkpoint['idx_chain'] = generation
        checkpoint['run_chain'] = run_chain
        checkpoint['chain_seq'] = chain_seq
        checkpoint['sampler_iterations'] = sampler.iterations
        checkpoint['sampler_naccepted'] = sampler.naccepted
        checkpoint['train_chain_stat'] = train_chain_stat
        checkpoint['run_chain_stat'] = run_chain_stat

        write_interval(cache.checkpointInterval, cache, checkpoint, checkpointFile, train_chain, run_chain, burn_seq, chain_seq, parameters, train_chain_stat, run_chain_stat, tau_percent)
        mle_process(last=False)
        util.graph_corner_process(cache, last=False)

        if generation % checkInterval == 0:
            try:
                tau = autocorr.integrated_time(numpy.swapaxes(run_chain, 0, 1), tol=cache.MCMCTauMult)
                scoop.logger.info("Mean acceptance fraction: %s %0.3f tau: %s with shape: %s", generation, accept, tau, run_chain.shape)
                if numpy.any(numpy.isnan(tau)):
                    scoop.logger.info("tau is NaN and clearly not complete %s", generation)
                else:
                    scoop.logger.info("we have run long enough and can quit %s", generation)
                    finished = True
            except autocorr.AutocorrError as err:
                scoop.logger.info(str(err))
                tau = err.tau
            scoop.logger.info("Mean acceptance fraction: %s %0.3f tau: %s", generation, accept, tau)
            tau = numpy.array(tau)
            tau_percent = generation / (tau * cache.MCMCTauMult)

    checkpoint['p_chain'] = p
    checkpoint['ln_prob_chain'] = ln_prob
    checkpoint['rstate_chain'] = random_state
    checkpoint['idx_chain'] = generation
    checkpoint['run_chain'] = run_chain
    checkpoint['chain_seq'] = chain_seq
    checkpoint['state'] = 'complete'

    write_interval(-1, cache, checkpoint, checkpointFile, train_chain, run_chain, burn_seq, chain_seq, parameters, train_chain_stat, run_chain_stat, tau_percent)

def write_interval(interval, cache, checkpoint, checkpointFile, train_chain, run_chain, burn_seq, chain_seq, parameters, train_chain_stat, run_chain_stat, tau_percent=None):
    "write the checkpoint and mcmc data at most every n seconds"
    if 'last_time' not in write_interval.__dict__:
        write_interval.last_time = time.time()

    if time.time() - write_interval.last_time > interval:
        with checkpointFile.open('wb') as cp_file:
            pickle.dump(checkpoint, cp_file)

        writeMCMC(cache, train_chain, run_chain, burn_seq, chain_seq, parameters, train_chain_stat, run_chain_stat, tau_percent)
        
        write_interval.last_time = time.time()

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()
    checkpointFile = Path(cache.settings['resultsDirMisc'], cache.settings.get('checkpointFile', 'check'))
    checkpoint = getCheckPoint(checkpointFile,cache)

    burn_seq = checkpoint.get('burn_seq', [])
    chain_seq = checkpoint.get('chain_seq', [])
        
    train_chain = checkpoint.get('train_chain', None)
    run_chain = checkpoint.get('run_chain', None)

    parameters = len(cache.MIN_VALUE)
    
    MCMCpopulationSet = cache.settings.get('MCMCpopulationSet', None)
    if MCMCpopulationSet is not None:
        populationSize = MCMCpopulationSet
    else:
        populationSize = parameters * cache.settings['MCMCpopulation']
               
    #Population must be even
    populationSize = populationSize + populationSize % 2  

    if checkpoint['state'] == 'start':
        scoop.logger.info("setting up kde")
        kde, kde_scaler = kde_generator.setupKDE(cache)
        checkpoint['state'] = 'burn_in'
    else:
        scoop.logger.info("loading kde")
        kde, kde_scaler = kde_generator.getKDE(cache)

    path = Path(cache.settings['resultsDirBase'], cache.settings['csv'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        sampler = emcee.EnsembleSampler(populationSize, parameters, log_posterior, args=[cache.json_path], pool=cache.toolbox, a=2.0)
        emcee.EnsembleSampler._get_lnprob = _get_lnprob
        emcee.EnsembleSampler._propose_stretch = _propose_stretch

        if 'sampler_a' not in checkpoint:
            checkpoint['sampler_a'] = sampler.a

        result_data = {'input':[], 'output':[], 'output_meta':[], 'results':{}, 'times':{}, 'input_transform':[], 'input_transform_extended':[], 'strategy':[], 
                   'mean':[], 'confidence':[], 'mcmc_score':[]}
        halloffame = pareto.DummyFront(similar=util.similar, similar_fit=util.similar_fit)
        meta_hof = pareto.ParetoFront(similar=util.similar, similar_fit=util.similar_fit_meta)
        grad_hof = pareto.DummyFront(similar=util.similar, similar_fit=util.similar_fit)

        def local(results):
            return process(cache, halloffame, meta_hof, grad_hof, result_data, results, writer, csvfile, sampler)
        sampler.process = local

        if checkpoint['state'] == 'burn_in':
            sampler_burn(cache, checkpoint, sampler, checkpointFile)

        if checkpoint['state'] == 'chain':
            run_chain = checkpoint.get('run_chain', None)
            if run_chain is not None:
                temp = run_chain[:, :checkpoint['idx_chain'], 0].T
                scoop.logger.info('complete shape %s', temp.shape)
 
                try:
                    tau = autocorr.integrated_time(numpy.swapaxes(run_chain[:, :checkpoint['idx_chain'], :], 0, 1), tol=cache.MCMCTauMult)
                    scoop.logger.info("we have previously run long enough and can quit %s", checkpoint['idx_chain'])
                    checkpoint['state'] = 'complete'
                except autocorr.AutocorrError as err:
                    scoop.logger.info(str(err))
                    tau = err.tau

                scoop.logger.info("Mean acceptance fraction: %s %0.3f tau: %s", checkpoint['idx_chain'], checkpoint['chain_seq'][-1], tau)
            
        if checkpoint['state'] == 'chain':
            sampler_run(cache, checkpoint, sampler, checkpointFile)

    chain = checkpoint['run_chain']
    chain_shape = chain.shape
    chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])

    if checkpoint['state'] == 'complete':
        tube_process(last=True)
        util.finish(cache)
        checkpoint['state'] = 'plot_finish'

        with checkpointFile.open('wb') as cp_file:
            pickle.dump(checkpoint, cp_file)

    if checkpoint['state'] == 'plot_finish':
        mle_process(last=True)
        util.graph_corner_process(cache, last=True)
    return numpy.mean(chain, 0)

def tube_process(last=False, interval=3600):
    cwd = str(Path(__file__).parent.parent)
    ret = subprocess.run([sys.executable, '-m', 'scoop', 'mcmc_plot_tube.py', str(cache.cache.json_path),], 
            stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)

def mle_process(last=False, interval=3600):
    if 'last_time' not in mle_process.__dict__:
        mle_process.last_time = time.time()

    if 'child' in mle_process.__dict__:
        if mle_process.child.poll() is None:  #This is false if the child has completed
            if last:
                mle_process.child.wait()
            else:
                return

    cwd = str(Path(__file__).parent.parent)

    if last:
        ret = subprocess.run([sys.executable, '-m', 'scoop', 'mle.py', str(cache.cache.json_path),], 
            stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
        mle_process.last_time = time.time()
    elif (time.time() - mle_process.last_time) > interval:
        #mle_process.child = subprocess.Popen([sys.executable, '-m', 'scoop', 'mle.py', str(cache.cache.json_path),], 
        #    stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
        subprocess.run([sys.executable, '-m', 'scoop', 'mle.py', str(cache.cache.json_path),], 
            stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
        mle_process.last_time = time.time()
        
def get_population(base, size, diff=0.05):
    new_population = base
    row, col = base.shape
    scoop.logger.info('%s', base)
    
    scoop.logger.info('row %s size %s', row, size)
    if row < size:
        #create new entries
        indexes = numpy.random.choice(new_population.shape[0], size - row, replace=True)
        temp = new_population[indexes,:]
        rand = numpy.random.uniform(1.0-diff, 1.0+diff, size=temp.shape)
        new_population = numpy.concatenate([new_population, temp * rand])
    if row > size:
        #randomly select entries to keep
        indexes = numpy.random.choice(new_population.shape[0], size, replace=False)
        scoop.logger.info('indexes: %s', indexes)
        new_population = new_population[indexes,:]
    return new_population

def getCheckPoint(checkpointFile, cache):
    if checkpointFile.exists():
        with checkpointFile.open('rb') as cp_file:
            checkpoint = pickle.load(cp_file)
    else:
        parameters = len(cache.MIN_VALUE)
        
        MCMCpopulationSet = cache.settings.get('MCMCpopulationSet', None)
        if MCMCpopulationSet is not None:
            populationSize = MCMCpopulationSet
        else:
            populationSize = parameters * cache.settings['MCMCpopulation']

        #Population must be even
        populationSize = populationSize + populationSize % 2  

        checkpoint = {}
        checkpoint['state'] = 'start'

        if cache.settings.get('PreviousResults', None) is not None:
            scoop.logger.info('running with previous best results')
            previousResultsFile = Path(cache.settings['PreviousResults'])
            results_h5 = cadet.H5()
            results_h5.filename = previousResultsFile.as_posix()
            results_h5.load()
            previousResults = results_h5.root.meta_population_transform

            row,col = previousResults.shape
            scoop.logger.info('row: %s col: %s  parameters: %s', row, col, parameters)
            if col < parameters:
                mcmc_h5 = Path(cache.settings.get('mcmc_h5', None))
                mcmcDir = mcmc_h5.parent
                mle_h5 = mcmcDir / "mle.h5"

                data = cadet.H5()
                data.filename = mle_h5.as_posix()
                data.load()
                scoop.logger.info('%s', list(data.root.keys()))
                stat_MLE = data.root.stat_MLE.reshape(1, -1)
                previousResults = numpy.hstack([previousResults, numpy.repeat(stat_MLE, row, 0)])
                scoop.logger.info('row: %s  col:%s   shape: %s', row, col, previousResults.shape)

            population = get_population(previousResults, populationSize, diff=0.1)
            checkpoint['p_burn'] = [util.convert_individual_inverse(i, cache) for i in population]
            scoop.logger.info('p_burn startup: %s', checkpoint['p_burn'])
        else:
            checkpoint['p_burn'] = SALib.sample.sobol_sequence.sample(populationSize, parameters)

        checkpoint['ln_prob_burn'] = None
        checkpoint['rstate_burn'] = None
        checkpoint['idx_burn'] = 0
        
        checkpoint['p_chain'] = None
        checkpoint['ln_prob_chain'] = None
        checkpoint['rstate_chain'] = None
        checkpoint['idx_chain'] = 0

        checkpoint['sampler_iterations'] = 0
        checkpoint['sampler_naccepted'] = numpy.zeros(populationSize)

        checkpoint['converge'] = numpy.ones(cache.settings.get('burnStable', 50)) * numpy.nan
    
    checkpoint['length_chain'] = cache.settings.get('chainLength', 50000)
    checkpoint['length_burn'] = cache.settings.get('burnIn', 50000)
    return checkpoint

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

    #scoop.logger.info("Mean acceptance fraction: %0.3f", numpy.mean(sampler.acceptance_fraction))

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
                       process.sim_start, process.generation_start, result_data, line_log=False)

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

def process_chain(chain, cache, idx):
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])

    flat_chain_transform = util.convert_population(flat_chain, cache)
    chain_transform = flat_chain_transform.reshape(chain_shape)

    return chain, flat_chain, chain_transform, flat_chain_transform

def writeMCMC(cache, train_chain, chain, burn_seq, chain_seq, parameters, train_chain_stat, run_chain_stat, tau_percent):
    "write out the mcmc data so it can be plotted"
    mcmcDir = Path(cache.settings['resultsDirMCMC'])
    mcmc_h5 = mcmcDir / "mcmc.h5"

    train_chain, train_chain_flat, train_chain_transform, train_chain_flat_transform = process_chain(train_chain, cache, len(burn_seq)-1)

    if chain is not None:
        chain, chain_flat, chain_transform, chain_flat_transform = process_chain(chain, cache, len(chain_seq)-1)
        interval_chain = chain_flat
        interval_chain_transform = chain_flat_transform
    else:
        interval_chain = train_chain_flat
        interval_chain_transform = train_chain_flat_transform

    flat_interval = interval(interval_chain, cache)
    flat_interval_transform = interval(interval_chain_transform, cache)

    flat_interval.to_csv(mcmcDir / "percentile.csv")
    flat_interval_transform.to_csv(mcmcDir / "percentile_transform.csv")

    h5 = cadet.H5()
    h5.filename = mcmc_h5.as_posix()

    if tau_percent is not None:
        h5.root.tau_percent = tau_percent.reshape(-1, 1)

    if train_chain_stat is not None:
        train_chain_stat, _, train_chain_stat_transform, _ = process_chain(train_chain_stat, cache, len(burn_seq)-1)

        h5.root.train_chain_stat = train_chain_stat
        h5.root.train_chain_stat_transform = train_chain_stat_transform

    if run_chain_stat is not None:
        run_chain_stat, _, run_chain_stat_transform, _ = process_chain(run_chain_stat, cache, len(burn_seq)-1)

        h5.root.run_chain_stat = run_chain_stat
        h5.root.run_chain_stat_transform = run_chain_stat_transform

    if burn_seq:
        h5.root.burn_in_acceptance = numpy.array(burn_seq).reshape(-1, 1)

    if chain_seq:
        h5.root.mcmc_acceptance = numpy.array(chain_seq).reshape(-1, 1)

    if chain is not None:
        h5.root.full_chain = chain
        h5.root.full_chain_transform = chain_transform
        h5.root.flat_chain = chain_flat
        h5.root.flat_chain_transform = chain_flat_transform

    h5.root.train_full_chain = train_chain
    h5.root.train_full_chain_transform = train_chain_transform
    h5.root.train_flat_chain = train_chain_flat
    h5.root.train_flat_chain_transform = train_chain_flat_transform

    mean = numpy.mean(interval_chain_transform,0)
    labels = [5, 10, 50, 90, 95]
    percentile = numpy.percentile(interval_chain_transform, labels, 0)

    h5.root.percentile['mean'] = mean
    for idx, label in enumerate(labels):
        h5.root.percentile['percentile_%s' % label] = percentile[idx,:]
        
    h5.save()

def interval(flat_chain, cache):
    mean = numpy.mean(flat_chain,0)

    percentile = numpy.percentile(flat_chain, [5, 10, 50, 90, 95], 0)

    data = numpy.vstack( (mean, percentile) ).transpose()

    pd = pandas.DataFrame(data, columns = ['mean', '5', '10', '50', '90', '95'])
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