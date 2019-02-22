import random
import pickle
import util
import numpy
import scipy
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

from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import matplotlib.pyplot as plt

import evo
import cache

import pareto

import scoop
import pandas
import array
import mle

import kde_generator
from sklearn.neighbors.kde import KernelDensity

name = "MCMC"

from addict import Dict
import matplotlib.cm
cm_plot = matplotlib.cm.gist_rainbow

def get_color(idx, max_colors, cmap):
    return cmap(1.*float(idx)/max_colors)

log2 = numpy.log(2)

saltIsotherms = {b'STERIC_MASS_ACTION', b'SELF_ASSOCIATION', b'MULTISTATE_STERIC_MASS_ACTION', 
                 b'SIMPLE_MULTISTATE_STERIC_MASS_ACTION', b'BI_STERIC_MASS_ACTION'}

def log_previous(cadetValues):
    if cache.cache.dataPreviousScaled is not None:
        #find the right values to use
        row, col = cache.cache.dataPreviousScaled.shape
        values = cadetValues[-col:]
        values_shape = numpy.array(values).reshape(1, -1)
        values_scaler = cache.cache.scalerPrevious.transform(values_shape)
        score = cache.cache.kdePrevious.score_samples(values_scaler)
        return score
    else:
        return 0.0

def setupPrevious(previous_bw):
    scores_temp = cache.cache.dataPreviousScaled
    kde = KernelDensity(kernel='gaussian', bandwidth=previous_bw, atol=kde_generator.bw_tol).fit(scores_temp)
    cache.cache.kdePrevious = kde

def log_likelihood(individual, json_path, kde_scores, kde_bw, previous_bw=None):
    if json_path != cache.cache.json_path:
        cache.cache.setup(json_path, False)
        cache.cache.roundScores = None
        cache.cache.roundParameters = None

    if previous_bw is not None and cache.cache.kdePrevious is None:
        setupPrevious(previous_bw)

    if 'kde' not in log_likelihood.__dict__:
        kde, pca, scaler = kde_generator.getKDE(cache.cache, kde_scores, kde_bw)
        log_likelihood.kde = kde
        log_likelihood.pca = pca
        log_likelihood.scaler = scaler

    scores, csv_record, results = evo.fitness(individual, json_path)

    if results is not None:
        logPrevious = log_previous(next(iter(results.values()))['cadetValues'])
    else:
        logPrevious = 0.0

    scores_shape = numpy.array(scores).reshape(1, -1)

    score_scaler = log_likelihood.scaler.transform(scores_shape)

    score = log_likelihood.kde.score_samples(score_scaler) + log2 + logPrevious #*2 is from mirroring and we need to double the probability to get back to the normalized distribution

    return score, scores, csv_record, results 

def log_posterior(theta, json_path, kde_scores, kde_bw, previous_bw=None):
    if json_path != cache.cache.json_path:
        cache.cache.setup(json_path)

    #try:
    ll, scores, csv_record, results = log_likelihood(theta, json_path, kde_scores, kde_bw, previous_bw)
    return ll, theta, scores, csv_record, results
    #except:
    #    # if model does not converge:
    #    return -numpy.inf, None, None, None, None

def previous_kde(cache):
    if cache.dataPreviousScaled is not None:
        bw, store = kde_generator.get_bandwidth(cache.dataPreviousScaled, cache)
        return bw

    return None

def addChain(*args):
    temp = [arg for arg in args if arg is not None]
    if len(temp) > 1:
        return numpy.concatenate( temp, axis=1)
    else:
        return numpy.array(temp[0])

def sampler_burn(cache, checkpoint, sampler, checkpointFile):
    burn_seq = checkpoint.get('burn_seq', [])
    chain_seq = checkpoint.get('chain_seq', [])
        
    train_chain = checkpoint.get('train_chain', None)
    run_chain = checkpoint.get('run_chain', None)

    if run_chain is not None:
        scoop.logger.info('sampler shape %s', run_chain.shape)

    converge = numpy.ones(cache.settings.get('burnStable', 50)) * numpy.nan

    parameters = len(cache.MIN_VALUE)

    tol = 5e-4
    count = checkpoint['length_burn'] - checkpoint['idx_burn']
    for idx, (p, ln_prob, random_state) in enumerate(sampler.sample(checkpoint['p_burn'], checkpoint['ln_prob_burn'],
                            checkpoint['rstate_burn'], iterations=count ), start=checkpoint['idx_burn']):
        accept = numpy.mean(sampler.acceptance_fraction)
        burn_seq.append(accept)
        converge[:-1] = converge[1:]
        converge[-1] = accept
        writeMCMC(cache, addChain(train_chain, sampler.chain), run_chain, burn_seq, chain_seq, parameters)

        converge_real = converge[~numpy.isnan(converge)]
        scoop.logger.info('burn:  idx: %s accept: %.3f std: %.3f mean: %.3f converge: %.3f', idx, accept, 
                            numpy.std(converge_real), numpy.mean(converge_real), numpy.std(converge_real)/tol)

        checkpoint['p_burn'] = p
        checkpoint['ln_prob_burn'] = ln_prob
        checkpoint['rstate_burn'] = random_state
        checkpoint['idx_burn'] = idx+1
        checkpoint['train_chain'] = addChain(train_chain, sampler.chain)
        checkpoint['burn_seq'] = burn_seq

        with checkpointFile.open('wb')as cp_file:
            pickle.dump(checkpoint, cp_file)

        if numpy.std(converge_real) < tol and len(converge) == len(converge_real):
            scoop.logger.info("burn in completed at iteration %s", idx)
            break

    checkpoint['state'] = 'chain'
    checkpoint['p_chain'] = p
    checkpoint['ln_prob_chain'] = None
    checkpoint['rstate_chain'] = None
    checkpoint['train_chain'] = addChain(train_chain, sampler.chain)
    checkpoint['burn_seq'] = burn_seq

    with checkpointFile.open('wb')as cp_file:
        pickle.dump(checkpoint, cp_file)

    train_chain = addChain(train_chain, sampler.chain)
    sampler.reset()

def sampler_run(cache, checkpoint, sampler, checkpointFile):
    burn_seq = checkpoint.get('burn_seq', [])
    chain_seq = checkpoint.get('chain_seq', [])
        
    train_chain = checkpoint.get('train_chain', None)
    run_chain = checkpoint.get('run_chain', None)

    if run_chain is not None:
        scoop.logger.info('sampler shape %s', run_chain.shape)

    checkInterval = 25
    mult = cache.MCMCTauMult
    count = checkpoint['length_chain'] - checkpoint['idx_chain']

    parameters = len(cache.MIN_VALUE)
                                 
    for idx, (p, ln_prob, random_state) in enumerate(sampler.sample(checkpoint['p_chain'], checkpoint['ln_prob_chain'],
                    checkpoint['rstate_chain'], iterations=count ), start=checkpoint['idx_chain']):
        accept = numpy.mean(sampler.acceptance_fraction)
        chain_seq.append(accept)
        writeMCMC(cache, train_chain, addChain(run_chain, sampler.chain), burn_seq, chain_seq, parameters)

        scoop.logger.info('run:  idx: %s accept: %.3f', idx, accept)
                
        checkpoint['p_chain'] = p
        checkpoint['ln_prob_chain'] = ln_prob
        checkpoint['rstate_chain'] = random_state
        checkpoint['idx_chain'] = idx+1
        checkpoint['run_chain'] = addChain(run_chain, sampler.chain)
        checkpoint['chain_seq'] = chain_seq

        scoop.logger.info('sampler idx %s shape %s', idx, checkpoint['run_chain'].shape)

        with checkpointFile.open('wb') as cp_file:
            pickle.dump(checkpoint, cp_file)

        if idx % checkInterval == 0 and idx >= 200:  
            tau = autocorr_new(addChain(run_chain, sampler.chain)[:, :idx, 0].T)
            scoop.logger.info("Mean acceptance fraction: %s %0.3f tau: %s", idx, accept, tau)
            if idx > (mult * tau):
                scoop.logger.info("we have run long enough and can quit %s", idx)
                break

    checkpoint['p_chain'] = p
    checkpoint['ln_prob_chain'] = ln_prob
    checkpoint['rstate_chain'] = random_state
    checkpoint['idx_chain'] = idx+1
    checkpoint['run_chain'] = addChain(run_chain, sampler.chain)
    checkpoint['chain_seq'] = chain_seq
    checkpoint['state'] = 'complete'

    scoop.logger.info('sampler idx %s shape %s', idx + 1, checkpoint['run_chain'].shape)

    with checkpointFile.open('wb')as cp_file:
        pickle.dump(checkpoint, cp_file)

def sampler_kde(checkpoint, cache, checkpointFile):
    if 'kde_scores' not in checkpoint:
        scoop.logger.info('Generating KDE for MCMC')
        kde_scores, kde_bw = kde_generator.generate_data(cache)
        checkpoint['kde_scores'] = kde_scores
        checkpoint['kde_bw'] = kde_bw
    else:
        scoop.logger.info('Reloading KDE from checkpoint for MCMC')
        kde_scores = checkpoint['kde_scores']
        kde_bw = checkpoint['kde_bw']

    with checkpointFile.open('wb')as cp_file:
        pickle.dump(checkpoint, cp_file)

    previous_bw = previous_kde(cache)

    scoop.logger.info("previous_bw: %s",  previous_bw)

    kde, pca, scaler = kde_generator.getKDE(cache, kde_scores, kde_bw)
    return kde, pca, scaler, kde_scores, kde_bw, previous_bw

def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()
    checkpointFile = Path(cache.settings['resultsDirMisc'], cache.settings['checkpointFile'])
    checkpoint = getCheckPoint(checkpointFile,cache)

    burn_seq = checkpoint.get('burn_seq', [])
    chain_seq = checkpoint.get('chain_seq', [])
        
    train_chain = checkpoint.get('train_chain', None)
    run_chain = checkpoint.get('run_chain', None)

    if run_chain is not None:
        scoop.logger.info('sampler shape %s', run_chain.shape)

    cache.roundScores = None
    cache.roundParameters = None

    parameters = len(cache.MIN_VALUE)
    
    MCMCpopulationSet = cache.settings.get('MCMCpopulationSet', None)
    if MCMCpopulationSet is not None:
        populationSize = MCMCpopulationSet
    else:
        populationSize = parameters * cache.settings['MCMCpopulation']
               
    #Population must be even
    populationSize = populationSize + populationSize % 2  

    sobol = SALib.sample.sobol_sequence.sample(populationSize, parameters)
    
    kde, pca, scaler, kde_scores, kde_bw, previous_bw = sampler_kde(checkpoint, cache, checkpointFile)

    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        sampler = emcee.EnsembleSampler(populationSize, parameters, log_posterior, args=[cache.json_path, kde_scores, kde_bw, previous_bw], pool=cache.toolbox, a=2.0)
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

        if checkpoint['state'] == 'burn_in':
            sampler_burn(cache, checkpoint, sampler, checkpointFile)

        if checkpoint['state'] == 'chain':
            run_chain = checkpoint.get('run_chain', None)
            if run_chain is not None:
                mult = cache.MCMCTauMult
                temp = run_chain[:, :checkpoint['idx_chain'], 0].T
                scoop.logger.info('complete shape %s', temp.shape)
                scoop.logger.info(run_chain[:, :checkpoint['idx_chain'], 0].T)
                tau = autocorr_new(run_chain[:, :checkpoint['idx_chain'], 0].T)
                scoop.logger.info("Mean acceptance fraction: %s %0.3f tau: %s", checkpoint['idx_chain'], checkpoint['chain_seq'][-1], tau)
                if checkpoint['idx_chain'] > (mult * tau):
                    scoop.logger.info("we have previously run long enough and can quit %s", checkpoint['idx_chain'])
                    checkpoint['state'] = 'complete'
            
        if checkpoint['state'] == 'chain':
            sampler_run(cache, checkpoint, sampler, checkpointFile)

    chain = checkpoint['run_chain']
    #chain = chain[:, :idx, :]
    chain_shape = chain.shape
    chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])
                
    plotTube(cache, chain, kde, pca, scaler)
    util.finish(cache)

    process_mle(chain, cache)
    return numpy.mean(chain, 0)

def process_mle(chain, cache):
    mle_x = mle.get_mle(chain)
    mle_ind = util.convert_individual(mle_x, cache)[0]
    scoop.logger.info("mle_x: %s", mle_x)
    scoop.logger.info("mle_ind: %s", mle_ind)

    temp = [mle_x,]

    scoop.logger.info('chain shape: %s', chain.shape)

    #run simulations for 5% 50% 95% and MLE vs experimental data
    percentile = numpy.percentile(chain, [5, 10, 50, 90, 95], 0)

    scoop.logger.info("percentile: %s", percentile)

    for row in percentile:
        temp.append(list(row))

    cadetValues = [util.convert_individual(i, cache)[0] for i in temp]

    scoop.logger.info('cadetValues: %s', cadetValues)

    fitnesses = list(cache.toolbox.map(cache.toolbox.evaluate, temp))

    simulations = {}
    for scores, csv_record, results in fitnesses:
        for name, value in results.items():
            sims = simulations.get(name, [])
            sims.append(value['simulation'])

            simulations[name] = sims
                       
    cadetValues = util.roundParameter(cadetValues, cache)

    mcmc_dir = Path(cache.settings['resultsDirMCMC'])

    mcmc_csv = Path(cache.settings['resultsDirMCMC']) / "prob.csv"

    pd = pandas.DataFrame(cadetValues, columns = cache.parameter_headers_actual)
    labels = ['MLE', '5', '10', '50', '90', '95']
    pd.insert(0, 'name', labels)
    pd.to_csv(mcmc_csv, index=False)

    plot_mle(simulations, cache, labels)

def plot_mle(simulations, cache, labels):
    mcmc_dir = Path(cache.settings['resultsDirMCMC'])
    target = cache.target
    settings = cache.settings
    for experiment in settings['experiments']:
        experimentName = experiment['name']
        
        file_name = '%s_stats.png' % experimentName
        dst = mcmc_dir / file_name

        numPlotsSeq = [1]
        #Shape and ShapeDecay have a chromatogram + derivative
        for feature in experiment['features']:
            if feature['type'] in ('Shape', 'ShapeDecay'):
                numPlotsSeq.append(2)
            elif feature['type'] in ('AbsoluteTime', 'AbsoluteHeight'):
                pass
            else:
                numPlotsSeq.append(1)

        numPlots = sum(numPlotsSeq)

        exp_time = target[experimentName]['time']
        exp_value = target[experimentName]['valueFactor']

        fig = figure.Figure(figsize=[10, numPlots*10])
        canvas = FigureCanvas(fig)

        graph_simulations(simulations[experimentName], labels, fig.add_subplot(numPlots, 1, 1))

        graphIdx = 2
        for idx, feature in enumerate(experiment['features']):
            featureName = feature['name']
            featureType = feature['type']

            feat = target[experimentName][featureName]

            selected = feat['selected']
            exp_time = feat['time'][selected]
            exp_value = feat['value'][selected]

            if featureType in ('similarity', 'similarityDecay', 'similarityHybrid', 'similarityHybrid2', 'similarityHybrid2_spline', 'similarityHybridDecay', 
                               'similarityHybridDecay2', 'curve', 'breakthrough', 'dextran', 'dextranHybrid', 'dextranHybrid2', 'dextranHybrid2_spline',
                               'similarityCross', 'similarityCrossDecay', 'breakthroughCross', 'SSE', 'LogSSE', 'breakthroughHybrid', 'breakthroughHybrid2',
                               'Shape', 'ShapeDecay', 'Dextran', 'DextranAngle'):
                
                graph = fig.add_subplot(numPlots, 1, graphIdx) #additional +1 added due to the overview plot

                for idx, (sim, label) in enumerate(zip(simulations[experimentName],labels)):
                    sim_time, sim_value = util.get_times_values(sim, target[experimentName][featureName])

                    if idx == 0:
                        linewidth = 2
                    else:
                        linewidth = 1
                    
                    graph.plot(sim_time, sim_value, '--', label=label, color=get_color(idx, len(simulations[experimentName]) + 1, cm_plot), linewidth = linewidth)

                graph.plot(exp_time, exp_value, '-', label='Experiment', color=get_color(len(simulations[experimentName]), len(simulations[experimentName]) + 1, cm_plot), linewidth=2)
                graphIdx += 1
            
            if featureType in ('derivative_similarity', 'derivative_similarity_hybrid', 'derivative_similarity_hybrid2', 'derivative_similarity_cross', 'derivative_similarity_cross_alt',
                                 'derivative_similarity_hybrid2_spline', 'similarityHybridDecay2_spline',
                                 'Shape', 'ShapeDecay'):

                graph = fig.add_subplot(numPlots, 1, graphIdx) #additional +1 added due to the overview plot
                for idx, (sim, label) in enumerate(zip(simulations[experimentName],labels)):
                    sim_time, sim_value = util.get_times_values(sim, target[experimentName][featureName])
                    sim_spline = scipy.interpolate.UnivariateSpline(sim_time, util.smoothing(sim_time, sim_value), s=util.smoothing_factor(sim_value)).derivative(1)

                    if idx == 0:
                        linewidth = 2
                    else:
                        linewidth = 1
                    
                    graph.plot(sim_time, sim_spline(sim_time), '--', label=label, color=get_color(idx, len(simulations[experimentName]) + 1, cm_plot), linewidth = linewidth)

                
                exp_spline = scipy.interpolate.UnivariateSpline(exp_time, util.smoothing(exp_time, exp_value), s=util.smoothing_factor(exp_value)).derivative(1)
                graph.plot(exp_time, exp_spline(exp_time), '-', label='Experiment', color=get_color(len(simulations[experimentName]), len(simulations[experimentName]) + 1, cm_plot), linewidth=2)
                graphIdx += 1
                        
            graph.legend()

        fig.savefig(str(dst))    

def graph_simulations(simulations, simulation_labels, graph):
    linestyles = ['-', '--', '-.', ':']
    for idx_sim, (simulation, label_sim) in enumerate(zip(simulations, simulation_labels)):

        comps = []

        ncomp = int(simulation.root.input.model.unit_001.ncomp)
        isotherm = bytes(simulation.root.input.model.unit_001.adsorption_model)

        hasSalt = isotherm in saltIsotherms

        solution_times = simulation.root.output.solution.solution_times

        hasColumn = isinstance(simulation.root.output.solution.unit_001.solution_outlet_comp_000, Dict)

        if hasColumn:
            for i in range(ncomp):
                comps.append(simulation.root.output.solution.unit_001['solution_column_outlet_comp_%03d' % i])
        else:
            for i in range(ncomp):
                comps.append(simulation.root.output.solution.unit_001['solution_outlet_comp_%03d' % i])

        if hasSalt:
            graph.set_title("Output")
            graph.plot(solution_times, comps[0], 'b-', label="Salt")
            graph.set_xlabel('time (s)')
        
            # Make the y-axis label, ticks and tick labels match the line color.
            graph.set_ylabel('mMol Salt', color='b')
            graph.tick_params('y', colors='b')

            axis2 = graph.twinx()
            for idx, comp in enumerate(comps[1:]):
                axis2.plot(solution_times, comp, linestyles[idx], color=get_color(idx_sim, len(simulation_labels), cm_plot), label="P%s %s" % (idx, label_sim))
            axis2.set_ylabel('mMol Protein', color='r')
            axis2.tick_params('y', colors='r')


            lines, labels = graph.get_legend_handles_labels()
            lines2, labels2 = axis2.get_legend_handles_labels()
            axis2.legend(lines + lines2, labels + labels2, loc=0)
        else:
            graph.set_title("Output")
        
            for idx, comp in enumerate(comps):
                graph.plot(solution_times, comp, linestyles[idx], color=get_color(idx_sim, len(simulation_labels), cm_plot), label="P%s %s" % (idx, label_sim))
            graph.set_ylabel('mMol Protein', color='r')
            graph.tick_params('y', colors='r')
            graph.set_xlabel('time (s)')

            lines, labels = graph.get_legend_handles_labels()
            graph.legend(lines, labels, loc=0)

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
        checkpoint['state'] = 'burn_in'
        checkpoint['p_burn'] = SALib.sample.sobol_sequence.sample(populationSize, parameters)
        checkpoint['ln_prob_burn'] = None
        checkpoint['rstate_burn'] = None
        checkpoint['idx_burn'] = 0
        
        checkpoint['p_chain'] = None
        checkpoint['ln_prob_chain'] = None
        checkpoint['rstate_chain'] = None
        checkpoint['idx_chain'] = 0
    
    checkpoint['length_chain'] = cache.settings.get('chainLength', 10000)
    checkpoint['length_burn'] = cache.settings.get('burnIn', 10000)
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

#auto correlation support functions

def autocorr_new(y, c=5.0):
    y = y[~numpy.all(y == 0, axis=1)]
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

def process_chain(chain, cache, idx):
    chain = chain[:, :idx+1, :]
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])

    chain_transform = numpy.array(chain)
    for walker in range(chain_shape[0]):
        for position in range(chain_shape[1]):
            chain_transform[walker, position,:] = util.convert_individual(chain_transform[walker, position, :], cache)[0]

    flat_chain_transform = chain_transform.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])

    return chain, flat_chain, chain_transform, flat_chain_transform

def writeMCMC(cache, train_chain, chain, burn_seq, chain_seq, parameters):
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

    with h5py.File(mcmc_h5, 'w') as hf:
        #if we don't have a file yet then we have to be doing burn in so no point in checking
        
        if burn_seq:
            data = numpy.array(burn_seq).reshape(-1, 1)
            hf.create_dataset("burn_in_acceptance", data=data, compression="gzip")
        
        if chain_seq:
            data = numpy.array(chain_seq).reshape(-1, 1)   
            hf.create_dataset("mcmc_acceptance", data=data, compression="gzip")
        
        if chain is not None:    
            hf.create_dataset("full_chain", data=chain, compression="gzip")
            hf.create_dataset("full_chain_transform", data=chain_transform, compression="gzip")

            hf.create_dataset("flat_chain", data=chain_flat, compression="gzip")
            hf.create_dataset("flat_chain_transform", data=chain_flat_transform, compression="gzip")

        hf.create_dataset("train_full_chain", data=train_chain, compression="gzip")
        hf.create_dataset("train_full_chain_transform", data=train_chain_transform, compression="gzip")

        hf.create_dataset("train_flat_chain", data=train_chain_flat, compression="gzip")
        hf.create_dataset("train_flat_chain_transform", data=train_chain_flat_transform, compression="gzip")

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
    "want about 1000 items and will be removing about 10% of them"
    size = 1100
    chain = chain[~numpy.all(chain == 0, axis=1)]
    temp = chain
    if len(chain) > size:
        indexes = numpy.random.choice(chain.shape[0], size, replace=False)
        chain = chain[indexes,:]

    lb, ub = numpy.percentile(chain, [5, 95], 0)
    selected = (chain >= lb) & (chain <= ub)
    bools = numpy.all(selected, 1)
    chain = chain[bools, :]

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
        if result is not None:
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
        else:
            scoop.logger.info("Failure in random choice: fit: %s  csv_line: %s   result:%s", fit, csv_line, result)

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
        if 'mcmc_selected' in hf:
            del hf['mcmc_selected']
            del hf['mcmc_selected_transformed']
            del hf['mcmc_selected_score']
            del hf['mcmc_score']
        hf.create_dataset("mcmc_selected", data=numpy.array(mcmc_selected), compression="gzip")
        hf.create_dataset("mcmc_selected_transformed", data=numpy.array(mcmc_selected_transformed), compression="gzip")
        hf.create_dataset("mcmc_selected_score", data=numpy.array(mcmc_selected_score), compression="gzip")
        hf.create_dataset("mcmc_score", data=numpy.array(mcmc_score), compression="gzip")

def plotTube(cache, chain, kde, pca, scaler):
    results, combinations = processChainForPlots(cache, chain, kde, pca, scaler)

    output_mcmc = cache.settings['resultsDirSpace'] / "mcmc"
    output_mcmc.mkdir(parents=True, exist_ok=True)

    mcmc_h5 = output_mcmc / "mcmc_plots.h5"
    with h5py.File(mcmc_h5, 'w') as hf:

        for expName,value in combinations.items():
            exp_name = expName.split('_')[0]
            plot_mcmc(output_mcmc, value, expName, "combine", cache.target[exp_name]['time'], cache.target[exp_name]['value'])
            hf.create_dataset(expName, data=value['data'], compression="gzip")
            hf.create_dataset('exp_%s_time' % expName, data=cache.target[exp_name]['time'], compression="gzip")
            hf.create_dataset('exp_%s_value' % expName, data=cache.target[exp_name]['value'], compression="gzip")

        for exp, units in results.items():
            for unitName, unit in units.items():
                for comp, data in unit.items():
                    expName = '%s_%s' % (exp, unitName)
                    plot_mcmc(output_mcmc, data, expName, comp, cache.target[exp]['time'], cache.target[exp]['value'])
                    hf.create_dataset('%s_%s' % (expName, comp), data=data['data'], compression="gzip")
                    hf.create_dataset('exp_%s_%s_time' % (expName, comp), data=cache.target[exp_name]['time'], compression="gzip")
                    hf.create_dataset('exp_%s_%s_value' % (expName, comp), data=cache.target[exp_name]['value'], compression="gzip")

def plot_mcmc(output_mcmc, value, expName, name, expTime, expValue):
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
    plt.plot(expTime, expValue, 'r')
    plt.savefig(str(output_mcmc / ("%s_%s.png" % (expName, name) ) ), bbox_inches='tight')
    plt.close()

    row, col = data.shape
    alpha = 0.005
    plt.plot(times, data.transpose(), 'g', alpha=alpha)
    plt.plot(times, mean, 'k')
    plt.plot(expTime, expValue, 'r')
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