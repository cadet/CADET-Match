import numpy
import pandas

import hashlib

from deap import tools
import scipy.signal
from pathlib import Path

from addict import Dict

import tempfile
import os
from cadet import Cadet, H5
import subprocess
import sys
import json
import time
import csv
import psutil

import random
import calc_coeff
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import scoop

import decimal as decim
decim.getcontext().prec = 64
__logBase10of2_decim = decim.Decimal(2).log10()
__logBase10of2 = float(__logBase10of2_decim)

import SALib.sample.sobol_sequence
import loggerwriter
import synthetic_error

def smoothing_factor(y):
    return max(y)/1000000.0

def find_extreme(seq):
    try:
        return max(seq, key=lambda x: abs(x[1]))
    except ValueError:
        return [0, 0]

def get_times_values(simulation, target, selected = None):

    try:
        times = simulation.root.output.solution.solution_times

        isotherm = target['isotherm']

        if isinstance(isotherm, list):
            values = numpy.sum([simulation[i] for i in isotherm], 0) 
        else:
            values = simulation[isotherm]
    except (AttributeError, KeyError):
        times = simulation[:,0]
        values = simulation[:,1]
    
    if selected is None:
        selected = target['selected']

    return times[selected], values[selected]* target['factor']

def sse(data1, data2):
    return numpy.sum( (data1 - data2)**2 )

def find_peak(times, data):
    "Return tuples of (times,data) for the peak we need"
    minIdx = numpy.argmin(data)
    maxIdx = numpy.argmax(data)

    return (times[maxIdx], data[maxIdx]), (times[minIdx], data[minIdx])

def find_breakthrough(times, data):
    "return tupe of time,value for the start breakthrough and end breakthrough"
    selected = data > 0.999 * max(data)
    selected_times = times[selected]
    return (selected_times[0], max(data)), (selected_times[-1], max(data))

def generateIndividual(icls, size, imin, imax, cache):
    return icls(numpy.random.uniform(imin, imax))

def initIndividual(icls, cache, content):
    return icls(content)

def sobolGenerator(icls, cache, n):
    if n > 0:
        populationDimension = len(cache.MIN_VALUE)
        populationSize = n
        sobol = SALib.sample.sobol_sequence.sample(populationSize, populationDimension)
        data = numpy.apply_along_axis(list, 1, sobol)
        data = list(map(icls, data))
        return data
    else:
        return []

def calcMetaScores(scores, cache):
    #scoop.logger.info("calcMetaScores %s %s", scores, cache.meta_mask)
    scores = numpy.array(scores)[cache.meta_mask]
    prod_score = product_score(scores)
    min_score = min(scores)
    mean_score = sum(scores)/len(scores)
    norm_score = numpy.sign(min_score) * numpy.linalg.norm(scores)/numpy.sqrt(len(scores))
    human = [prod_score, min_score, mean_score, norm_score]
    return human

def product_score(values):
    values = numpy.array(values)
    if numpy.all(values >= 0.0):
        return numpy.prod(values)**(1.0/len(values))
    else:
        return -numpy.prod(numpy.abs(values))**(1.0/len(values))

def averageFitness(offspring, cache):
    total = 0.0
    number = 0.0
    bestMin = -sys.float_info.max
    bestProd = -sys.float_info.max

    #scoop.logger.info(offspring)

    for i in offspring:
        #scoop.logger.info('idx: %s  value:%s', i, i.fitness.values)
        total += sum(i.fitness.values)
        number += len(i.fitness.values)
        bestMin = max(bestMin, min(i.fitness.values))
        bestProd = max(bestProd, product_score(i.fitness.values))
    #scoop.logger.info('number: %s', number)
    result = [total/number, bestMin, bestProd]

    return result

def smoothing(times, values):
    #temporarily get rid of smoothing for debugging
    #return values
    #filter length must be odd, set to 10% of the feature size and then make it odd if necesary
    filter_length = int(.1 * len(values))
    if filter_length % 2 == 0:
        filter_length += 1
    return scipy.signal.savgol_filter(values, filter_length, 3)

def mutPolynomialBoundedAdaptive(individual, eta, low, up, indpb):
    """Adaptive eta for mutPolynomialBounded"""
    scores = individual.fitness.values
    mult = min(scores)
    eta = eta + mult * 0
    return tools.mutPolynomialBounded(individual, eta, low, up, indpb)

def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Adaptive eta for mutPolynomialBounded"""
    
    scores = individual.fitness.values
    mult = min(scores)

    a,b = calc_coeff.linear_coeff(0.0, 1, 0.996, 10)
    eta = calc_coeff.linear(mult, a, b) * eta

    individual =  tools.mutPolynomialBounded(individual, eta, low, up, indpb)
    return individual

def mutationBoundedAdaptive(individual, low, up, indpb):
    scores = individual.fitness.values
    mult = min(scores)
    
    rand = numpy.random.rand(len(individual))

    for idx, i in enumerate(individual):
        if rand[idx] <= indpb:
            #scale = (1e-3 - 1.0) * prod + 1  (linear does not work well)
            scale = numpy.exp(-5.824*mult) * (up[idx] - low[idx])/1.0
            dist = numpy.random.normal(scale, scale/2.0) * random.sample([-1, 1], 1)[0]
            individual[idx] = max(min(i + dist, up[idx]), low[idx])
    return individual,

def mutationBoundedAdaptive2(individual, low, up, indpb):
    scores = individual.fitness.values
    mult = min(scores)

    if numpy.isnan(mult):
        mult = 0.0

    if mult < 0.9:
        m,b = calc_coeff.linear_coeff(0.1, 4, 0.9, 1)
        center = calc_coeff.linear(mult, m, b)
        sigma = center/2
    else:
        m,b = calc_coeff.exponential_coeff(0.9, 1, 1.0, 1e-2)

        center = 0
        sigma = calc_coeff.exponential(mult, m, b)
    
    rand = numpy.random.rand(len(individual))

    for idx, i in enumerate(individual):
        if rand[idx] <= indpb:
            if sigma == 0:
                dist = numpy.random.normal(center, sigma)
            else:
                dist = numpy.random.normal(center, sigma) * random.sample([-1, 1], 1)[0]
            individual[idx] = max(min(i + dist, up[idx]), low[idx])
    return individual,

def saveExperiments(save_name_base, settings, target, results, directory, file_pattern):
    for experiment in settings['experiments']:
        experimentName = experiment['name']
        simulation = results[experimentName]['simulation']

        dst = Path(directory, file_pattern % (save_name_base, experimentName))

        if dst.is_file():  #File already exists don't try to write over it
            return False
        else:
            simulation.filename = dst.as_posix()

            for (header, score) in zip(experiment['headers'], results[experimentName]['scores']):
                simulation.root.score[header] = score
            simulation.save()

    return True

def convert_individual(individual, cache):
    cadetValues = []
    cadetValuesExtended = []

    idx = 0
    for parameter in cache.settings['parameters']:
        transform = parameter['transform']
        count = cache.transforms[transform].count
        seq = individual[idx:idx+count]
        values, headerValues = cache.transforms[transform].untransform(seq, cache, parameter)
        cadetValues.extend(values)
        cadetValuesExtended.extend(headerValues)
        idx += count

    return cadetValues, cadetValuesExtended

def convert_individual_grad(individual, cache):
    cadetValues = []

    idx = 0
    for parameter in cache.settings['parameters']:
        transform = parameter['transform']
        count = cache.transforms[transform].count
        seq = individual[idx:idx+count]
        values = cache.transforms[transform].grad_untransform(seq, cache, parameter)
        cadetValues.extend(values)
        idx += count

    return cadetValues

def convert_population(population, cache):
    cadetValues = numpy.zeros(population.shape)

    idx = 0
    for parameter in cache.settings['parameters']:
        transform = parameter['transform']
        count = cache.transforms[transform].count
        if count:
            matrix = population[:,idx:idx+count]
            values = cache.transforms[transform].untransform_matrix(matrix, cache, parameter)
            cadetValues[:,idx:idx+count] = values
            idx += count

    return cadetValues

def convert_individual_inverse(individual, cache):
    return numpy.array([f(v) for f, v in zip(cache.settings['transform'], individual)])

def convert_individual_inverse_grad(individual, cache):
    return numpy.array([f(v) for f, v in zip(cache.settings['grad_transform'], individual)])

def set_simulation(individual, simulation, settings, cache, experiment):
    scoop.logger.debug("individual %s", individual)

    cadetValues = []
    cadetValuesKEQ = []

    idx = 0
    for parameter in settings['parameters']:
        transform = parameter['transform']
        count = cache.transforms[transform].count
        seq = individual[idx:idx+count]
        values, headerValues = cache.transforms[transform].setSimulation(simulation, parameter, seq, cache, experiment)
        cadetValues.extend(values)
        cadetValuesKEQ.extend(headerValues)
        idx += count

    scoop.logger.debug("finished setting hdf5")
    return cadetValues, cadetValuesKEQ

def getBoundOffset(unit):
    if unit.unit_type == b'CSTR':
        NBOUND = unit.nbound

        if not NBOUND:
            "For a CSTR with NBOUND not set the default is all 0"
            NBOUND = [0.0] * unit.ncomp
    else:
        NBOUND = unit.discretization.nbound

    boundOffset = numpy.cumsum(numpy.concatenate([[0,], NBOUND]))
    return boundOffset

def runExperiment(individual, experiment, settings, target, template_sim, timeout, cache, post_function=None):
    handle, path = tempfile.mkstemp(suffix='.h5', dir=cache.tempDir)
    os.close(handle)

    simulation = Cadet(template_sim.root)
    simulation.filename = path

    simulation.root.input.solver.nthreads = int(settings.get('nThreads', 1))

    if individual is not None:
        cadetValues, cadetValuesKEQ = set_simulation(individual, simulation, settings, cache, experiment)
    else:
        cadetValues = []
        cadetValuesKEQ = []

    simulation.save()

    try:
        simulation.run(timeout = timeout, check=True)
    except subprocess.TimeoutExpired:
        scoop.logger.warn("Simulation Timed Out")
        os.remove(path)
        return None

    except subprocess.CalledProcessError as error:
        scoop.logger.error("The simulation failed %s", individual)
        logError(cache, cadetValuesKEQ, error)
        #os.remove(path)
        return None

    #read sim data
    simulation.load()
    os.remove(path)

    simulationFailed = isinstance(simulation.root.output.solution.solution_times, Dict)
    if simulationFailed:
        scoop.logger.error("%s sim must have failed %s", individual, path)
        return None
    scoop.logger.debug('Everything ran fine')

    if post_function:
        post_function(simulation)

    temp = {}
    temp['simulation'] = simulation
    temp['path'] = path
    temp['scores'] = []
    temp['error'] = 0.0
    temp['error_count'] = 0.0
    temp['cadetValues'] = cadetValues
    temp['cadetValuesKEQ'] = cadetValuesKEQ

    if individual is not None:
        temp['individual'] = tuple(individual)
    temp['diff'] = []
    temp['minimize'] = []
    temp['sim_time'] = []
    temp['sim_value'] = []
    temp['exp_value'] = []

    for feature in experiment['features']:
        start = float(feature['start'])
        stop = float(feature['stop'])
        featureType = feature['type']
        featureName = feature['name']

        if featureType in cache.scores:
            scores, sse, sse_count, sim_time, sim_value, exp_value, minimize = cache.scores[featureType].run(temp, target[experiment['name']][featureName])
            diff = sim_value - exp_value
 
        temp['scores'].extend(scores)
        temp['error'] += sse
        temp['error_count'] += sse_count
        temp['diff'].extend(diff)
        temp['minimize'].extend(minimize)
        temp['sim_time'].append(sim_time)
        temp['sim_value'].append(sim_value)
        temp['exp_value'].append(exp_value)

    return temp

def logError(cache, values, error):
    with cache.error_path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        row = list(values)
        row.append(error.returncode)
        row.append(error.stdout)
        row.append(error.stderr)

        writer.writerow(row)


def repeatSimulation(idx):
    "read the original json file and copy it to a subdirectory for each repeat and change where the target data is written"
    settings_file = Path(sys.argv[1])
    with settings_file.open() as json_data:
        settings = json.load(json_data)

        baseDir = settings.get('baseDir', None)
        if baseDir is not None:
            baseDir = Path(baseDir)
            settings['resultsDir'] = baseDir / settings['resultsDir']

        resultDir = Path(settings['resultsDir']) / str(idx)
        resultDir.mkdir(parents=True, exist_ok=True)

        settings['resultsDirOriginal'] = settings['resultsDir'].as_posix()
        settings['resultsDir'] = resultDir.as_posix()

        if baseDir is not None:
            settings['baseDir'] = baseDir.as_posix()

        new_settings_file = resultDir / settings_file.name
        with new_settings_file.open(mode="w") as json_data:
            json.dump(settings, json_data, indent=4, sort_keys=True)
        return new_settings_file

def setupMCMC(cache, lb, ub):
    "read the original json file and make an mcmc file based on it with new boundaries"
    settings_file = Path(sys.argv[1])
    with settings_file.open() as json_data:
        settings = json.load(json_data)
        settings['continueMCMC'] = 0

        baseDir = settings.get('baseDir', None)
        if baseDir is not None:
            baseDir = Path(baseDir)
            settings['resultsDir'] = baseDir / settings['resultsDir']
        
        resultDirOriginal = Path(settings['resultsDir'])
        resultsOriginal = resultDirOriginal / "result.h5"
        resultDir = resultDirOriginal / "mcmc_refine"
        resultDir.mkdir(parents=True, exist_ok=True)        

        settings['resultsDirOriginal'] = resultDirOriginal.as_posix()
        settings['resultsDir'] = resultDir.as_posix()
        settings['PreviousResults'] = resultsOriginal.as_posix()

        if baseDir is not None:
            settings['baseDir'] = baseDir.as_posix()

        if 'mcmc_h5' in settings and 'parameters_mcmc' in settings:
            settings['parameters'].extend(settings['parameters_mcmc'])

        settings['searchMethod'] = 'MCMC'
        settings['graphSpearman'] = 0

        for experiment in settings['experiments']:
            foundAbsoluteTime = False
            foundAbsoluteHeight = False
            for feature in experiment['features']:
                if feature["type"] == "AbsoluteTime":
                    foundAbsoluteTime = True
                if feature["type"] == "AbsoluteHeight":
                    foundAbsoluteHeight = True
            if foundAbsoluteTime is False:
                experiment['features'].append({"name":"AbsoluteTime", "type":"AbsoluteTime"})
            if foundAbsoluteHeight is False:
                experiment['features'].append({"name":"AbsoluteHeight", "type":"AbsoluteHeight"})
         
        if "kde_synthetic" in settings:
            individual = getBestIndividual(cache)
            found = createSimulationBestIndividual(individual, cache)
            for experiment in settings['kde_synthetic']:
                experiment['file_path'] = found[experiment['name']]

        new_settings_file = resultDir / settings_file.name
        with new_settings_file.open(mode="w") as json_data:
            json.dump(settings, json_data, indent=4, sort_keys=True)
        return new_settings_file

def setupAltFeature(cache, name):
    "read the original json file and make an mcmc file based on it with new boundaries"
    settings_file = Path(sys.argv[1])
    with settings_file.open() as json_data:
        settings = json.load(json_data)

        baseDir = settings.get('baseDir', None)
        if baseDir is not None:
            baseDir = Path(baseDir)
            settings['resultsDir'] = baseDir / settings['resultsDir']
        
        resultDirOriginal = Path(settings['resultsDir'])
        resultsOriginal = resultDirOriginal / "result.h5"
        resultDir = resultDirOriginal / "altScore" / name
        resultDir.mkdir(parents=True, exist_ok=True)        

        settings['resultsDirOriginal'] = resultDirOriginal.as_posix()
        settings['resultsDir'] = resultDir.as_posix()
        settings['PreviousResults'] = resultsOriginal.as_posix()

        if baseDir is not None:
            settings['baseDir'] = baseDir.as_posix()

        settings['searchMethod'] = 'ScoreTest'

        data = H5()
        data.filename = resultsOriginal.as_posix()
        data.load(paths=['/meta_population_transform',])

        settings['seeds'] = [list(i) for i in data.root.meta_population_transform]

        for experiment in settings['experiments']:
            for feature in experiment['featuresAlt']:
                if feature['name'] == name:
                    experiment['features'] = feature['features']
         
        new_settings_file = resultDir / settings_file.name
        with new_settings_file.open(mode="w") as json_data:
            json.dump(settings, json_data, indent=4, sort_keys=True)
        return new_settings_file

def createSimulationBestIndividual(individual, cache):
    temp = {}
    for experiment in cache.settings['experiments']:
        name = experiment['name']
        templatePath = Path(cache.settings['resultsDirMisc'], "template_%s_base.h5" % name)
        templateSim = Cadet()
        templateSim.filename = templatePath.as_posix()
        templateSim.load()

        cadetValues, cadetValuesKEQ = set_simulation(individual, templateSim, cache.settings, cache, experiment)

        bestPath = Path(cache.settings['resultsDirMisc'], "best_%s_base.h5" % name)
        templateSim.filename = bestPath.as_posix()
        templateSim.save()
        temp[name] = bestPath.as_posix()
    return temp

def getBestIndividual(cache):
    "return the path to the best item based on meta min score"
    progress_path = Path(cache.settings['resultsDirBase']) / "result.h5"
    results = H5()
    results.filename = progress_path.as_posix()
    results.load(paths=['/meta_score', '/meta_population'])

    idx = numpy.argmax(results.root.meta_score[:,1])
    individual = results.root.meta_population[idx,:]
    return individual

def copyCSVWithNoise(idx, center, noise):
    "read the original json file and create a new set of simulation data and simulation file in a subdirectory with noise"
    settings_file = Path(sys.argv[1])
    with settings_file.open() as json_data:
        settings = json.load(json_data)

        baseDir = Path(settings['resultsDir']) / str(idx)
        baseDir.mkdir(parents=True, exist_ok=True)

        settings['resultsDir'] = str(baseDir)

        del settings['bootstrap']

        #find CSV files
        for experiment in settings['experiments']:
            if 'CSV' in experiment:
                data = numpy.genfromtxt(experiment['CSV'], delimiter=',')
                addNoise(data, center, noise)
                csv_path = Path(experiment['CSV'])
                new_csv_path = baseDir / csv_path.name
                numpy.savetxt(str(new_csv_path), data, delimiter=',')
                experiment['CSV'] = str(new_csv_path)
            for feature in experiment['features']:
                if 'CSV' in feature:
                    data = numpy.genfromtxt(feature['CSV'], delimiter=',')
                    addNoise(data, center, noise)
                    csv_path = Path(feature['CSV'])
                    new_csv_path = baseDir / csv_path.name
                    numpy.savetxt(str(new_csv_path), data, delimiter=',')
                    feature['CSV'] = str(new_csv_path)

        new_settings_file = baseDir / settings_file.name
        with new_settings_file.open(mode="w") as json_data:
            json.dump(settings, json_data)
        return new_settings_file
        

def addNoise(array, center, noise):
    "add noise to an array"
    maxValue = numpy.max(array[:, 1])
    randomNoise = numpy.random.normal(center, noise*maxValue, len(array))
    array[:, 1] += randomNoise

def bestMinScore(hof):
    "find the best score based on the minimum of the scores"
    idxMax = numpy.argmax([min(i.fitness.values) for i in hof])
    return hof[idxMax]

def similar(a, b, cache):
    "we only need a parameter to 4 digits of accuracy so have the pareto front only keep up to 5 digits for members of the front"
    a = numpy.array(convert_individual(a,cache)[0])
    b = numpy.array(convert_individual(b,cache)[0])
    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < 1e-3)

def RoundToSigFigs( x, sigfigs ):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.
    Return value has the same type as x.
    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value or an array like object containing only real values.
    """
    if not ( type(sigfigs) is int or isinstance(sigfigs, numpy.integer) ):
        raise TypeError( "RoundToSigFigs: sigfigs must be an integer." )

    if sigfigs <= 0:
        raise ValueError( "RoundtoSigFigs: sigfigs must be positive." )
    
    if not numpy.all(numpy.isreal( x )):
        raise TypeError( "RoundToSigFigs: all x must be real." )

    matrixflag = False
    if isinstance(x, numpy.matrix): #Convert matrices to arrays
        matrixflag = True
        x = numpy.asarray(x)
    
    xsgn = numpy.sign(x)
    absx = xsgn * x
    mantissas, binaryExponents = numpy.frexp( absx )
    
    decimalExponents = __logBase10of2 * binaryExponents
    omags = numpy.floor(decimalExponents)

    mantissas *= 10.0**(decimalExponents - omags)
    
    if type(mantissas) is float or isinstance(mantissas, numpy.floating):
        if mantissas < 1.0:
            mantissas *= 10.0
            omags -= 1.0
            
    else: #elif np.all(np.isreal( mantissas )):
        fixmsk = mantissas < 1.0
        mantissas[fixmsk] *= 10.0
        omags[fixmsk] -= 1.0

    result = xsgn * numpy.around( mantissas, decimals=sigfigs - 1 ) * 10.0**omags
    if matrixflag:
        result = numpy.matrix(result, copy=False)
    
    return result

def fracStat(time_center, value):
    mean_time = numpy.sum(time_center*value)/numpy.sum(value)
    variance_time = numpy.sum( (time_center - mean_time)**2 * value )/numpy.sum(value)
    skew_time = numpy.sum( (time_center - mean_time)**3 * value )/numpy.sum(value)

    mean_value = numpy.sum(time_center*value)/numpy.sum(time_center)
    variance_value = numpy.sum( (value - mean_value)**2 * time_center )/numpy.sum(time_center)
    skew_value = numpy.sum( (value - mean_value)**3 * time_center )/numpy.sum(time_center)

    return mean_time, variance_time, skew_time, mean_value, variance_value, skew_value

def fractionate(start_seq, stop_seq, times, values):
    temp = []
    for (start, stop) in zip(start_seq, stop_seq):
        selected = (times >= start) & (times <= stop)
        local_times = times[selected]
        local_values = values[selected]

        #need to use the actual start and stop times from the data no what we want to cut at
        stop = local_times[-1]
        start = local_times[0]

        temp.append(numpy.trapz(local_values, local_times)/ (stop - start))
    return numpy.array(temp)

def writeProgress(cache, generation, population, halloffame, meta_halloffame, grad_halloffame, average_score, 
                  minimum_score, product_score, sim_start, generation_start, result_data=None, line_log=True):
    cpu_time = psutil.Process().cpu_times()
    now = time.time()

    results = Path(cache.settings['resultsDirBase'])

    hof = results / "hof.npy"
    meta_hof = results / "meta_hof.npy"
    grad_hof = results / "grad_hof.npy"

    data = numpy.array([i.fitness.values for i in halloffame])
    data_meta = numpy.array([i.fitness.values for i in meta_halloffame])
    data_grad = numpy.array([i.fitness.values for i in grad_halloffame])

    hof_param = numpy.array(halloffame)
    meta_param = numpy.array(meta_halloffame)
    grad_param = numpy.array(grad_halloffame)

    hof_param_transform = numpy.array([convert_individual(i, cache)[0] for i in halloffame])
    meta_param_transform = numpy.array([convert_individual(i, cache)[0] for i in meta_halloffame])
    grad_param_transform = numpy.array([convert_individual(i, cache)[0] for i in grad_halloffame])

    with hof.open('wb') as hof_file:
        numpy.save(hof_file, data)

    with meta_hof.open('wb') as hof_file:
        numpy.save(hof_file, data_meta)

    with grad_hof.open('wb') as hof_file:
        numpy.save(hof_file, data_grad)

    gen_data = numpy.array([generation, len(result_data['input'])]).reshape(1,2)

    if cache.debugWrite:
        population_input = []
        population_output = []
        for ind in population:
            temp = [generation]
            temp.extend(ind)
            population_input.append( temp)

            temp = [generation]
            temp.extend(ind.fitness.values)
            population_output.append( temp)

        population_input = numpy.array(population_input)
        population_output = numpy.array(population_output)

    if result_data is not None:
        resultDir = Path(cache.settings['resultsDir'])
        result_h5 = resultDir / "result.h5"

        if not result_h5.exists():
            with h5py.File(result_h5, 'w') as hf:
                hf.create_dataset("input", data=result_data['input'], maxshape=(None, len(result_data['input'][0])))

                if len(result_data['strategy']):
                    hf.create_dataset("strategy", data=result_data['strategy'], maxshape=(None, len(result_data['strategy'][0])))

                if len(result_data['mean']):
                    hf.create_dataset("mean", data=result_data['mean'], maxshape=(None, len(result_data['mean'][0])))

                if len(result_data['confidence']):
                    hf.create_dataset("confidence", data=result_data['confidence'], maxshape=(None, len(result_data['confidence'][0])))

                if cache.correct is not None:
                    distance = cache.correct - result_data['input']
                    hf.create_dataset("distance_correct", data=distance, maxshape=(None, len(result_data['input'][0])))

                hf.create_dataset("output", data=result_data['output'], maxshape=(None, len(result_data['output'][0])))
                hf.create_dataset("output_meta", data=result_data['output_meta'], maxshape=(None, len(result_data['output_meta'][0])))

                hf.create_dataset("input_transform", data=result_data['input_transform'], maxshape=(None, len(result_data['input_transform'][0])))
                if result_data['input_transform'] == result_data['input_transform_extended']:
                    hf.create_dataset("is_extended_input", data=False)                    
                else:
                    hf.create_dataset("is_extended_input", data=True)
                    hf.create_dataset("input_transform_extended", data=result_data['input_transform_extended'], maxshape=(None, len(result_data['input_transform_extended'][0])))

                hf.create_dataset("generation", data=gen_data, maxshape=(None, 2))
                
                if cache.debugWrite:
                    hf.create_dataset("population_input", data=population_input, maxshape=(None, population_input.shape[1] ))
                    hf.create_dataset("population_output", data=population_output, maxshape=(None, population_output.shape[1] ))

                    mcmc_score = result_data.get('mcmc_score', None)
                    if mcmc_score is not None:
                        mcmc_score = numpy.array(mcmc_score)
                        hf.create_dataset("mcmc_score", data=mcmc_score, maxshape=(None, len(mcmc_score[0])))

                if len(hof_param):
                    hf.create_dataset('hof_population', data=hof_param, maxshape=(None, hof_param.shape[1] ))
                    hf.create_dataset('hof_population_transform', data=hof_param_transform, maxshape=(None, hof_param_transform.shape[1] ))
                    hf.create_dataset('hof_score', data=data, maxshape=(None, data.shape[1] ))

                if len(meta_param):
                    hf.create_dataset('meta_population', data=meta_param, maxshape=(None, meta_param.shape[1] ))
                    hf.create_dataset('meta_population_transform', data=meta_param_transform, maxshape=(None, meta_param_transform.shape[1] ))
                    hf.create_dataset('meta_score', data=data_meta, maxshape=(None, data_meta.shape[1] ))

                if len(grad_param):
                    hf.create_dataset('grad_population', data=grad_param, maxshape=(None, grad_param.shape[1] ))
                    hf.create_dataset('grad_population_transform', data=grad_param_transform, maxshape=(None, grad_param_transform.shape[1] ))
                    hf.create_dataset('grad_score', data=data_grad, maxshape=(None, data_grad.shape[1] ))

                if cache.fullTrainingData:

                    for filename, chroma in result_data['results'].items():
                        hf.create_dataset(filename, data=chroma, maxshape=(None, len(chroma[0])))

                    for filename, chroma in result_data['times'].items():
                        hf.create_dataset(filename, data=chroma)
        else:
            with h5py.File(result_h5, 'a') as hf:
                hf["input"].resize((hf["input"].shape[0] + len(result_data['input'])), axis = 0)
                hf["input"][-len(result_data['input']):] = result_data['input']

                if len(result_data['strategy']):
                    hf["strategy"].resize((hf["strategy"].shape[0] + len(result_data['strategy'])), axis = 0)
                    hf["strategy"][-len(result_data['strategy']):] = result_data['strategy']

                if len(result_data['mean']):
                    hf["mean"].resize((hf["mean"].shape[0] + len(result_data['mean'])), axis = 0)
                    hf["mean"][-len(result_data['mean']):] = result_data['mean']

                if len(result_data['confidence']):
                    hf["confidence"].resize((hf["confidence"].shape[0] + len(result_data['confidence'])), axis = 0)
                    hf["confidence"][-len(result_data['confidence']):] = result_data['confidence']

                if cache.correct is not None:
                    distance = cache.correct - result_data['input']
                    hf["distance_correct"].resize((hf["distance_correct"].shape[0] + len(result_data['input'])), axis = 0)
                    hf["distance_correct"][-len(result_data['input']):] = distance

                if cache.debugWrite:
                    mcmc_score = result_data.get('mcmc_score', None)
                    if mcmc_score is not None:
                        mcmc_score = numpy.array(mcmc_score)
                        hf["mcmc_score"].resize((hf["mcmc_score"].shape[0] + len(mcmc_score)), axis = 0)
                        hf["mcmc_score"][-len(mcmc_score):] = mcmc_score

                    hf["population_input"].resize((hf["population_input"].shape[0] + population_input.shape[0]), axis = 0)
                    hf["population_input"][-population_input.shape[0]:] = population_input

                    hf["population_output"].resize((hf["population_output"].shape[0] + population_output.shape[0]), axis = 0)
                    hf["population_output"][-population_output.shape[0]:] = population_output

                hf["output"].resize((hf["output"].shape[0] + len(result_data['output'])), axis = 0)
                hf["output"][-len(result_data['output']):] = result_data['output']

                hf["output_meta"].resize((hf["output_meta"].shape[0] + len(result_data['output_meta'])), axis = 0)
                hf["output_meta"][-len(result_data['output_meta']):] = result_data['output_meta']

                hf["input_transform"].resize((hf["input_transform"].shape[0] + len(result_data['input_transform'])), axis = 0)
                hf["input_transform"][-len(result_data['input_transform']):] = result_data['input_transform']

                if result_data['input_transform'] != result_data['input_transform_extended']:
                    hf["input_transform_extended"].resize((hf["input_transform_extended"].shape[0] + len(result_data['input_transform_extended'])), axis = 0)
                    hf["input_transform_extended"][-len(result_data['input_transform_extended']):] = result_data['input_transform_extended']

                if len(hof_param):
                    hf["hof_population"].resize((hof_param.shape[0]), axis = 0)
                    hf["hof_population"][:] = hof_param

                    hf["hof_population_transform"].resize((hof_param_transform.shape[0]), axis = 0)
                    hf["hof_population_transform"][:] = hof_param_transform

                    hf["hof_score"].resize((data.shape[0]), axis = 0)
                    hf["hof_score"][:] = data

                if len(meta_param):
                    hf["meta_population"].resize((meta_param.shape[0]), axis = 0)
                    hf["meta_population"][:] = meta_param

                    hf["meta_population_transform"].resize((meta_param_transform.shape[0]), axis = 0)
                    hf["meta_population_transform"][:] = meta_param_transform

                    hf["meta_score"].resize((data_meta.shape[0]), axis = 0)
                    hf["meta_score"][:] = data_meta

                if len(grad_param):
                    if "grad_population" in hf:
                        hf["grad_population"].resize((grad_param.shape[0]), axis = 0)
                        hf["grad_population"][:] = grad_param

                        hf["grad_population_transform"].resize((grad_param_transform.shape[0]), axis = 0)
                        hf["grad_population_transform"][:] = grad_param_transform

                        hf["grad_score"].resize((grad_score.shape[0]), axis = 0)
                        hf["grad_score"][:] = grad_score
                    else:
                        hf.create_dataset('grad_population', data=grad_param, maxshape=(None, grad_param.shape[1] ))
                        hf.create_dataset('grad_population_transform', data=grad_param_transform, maxshape=(None, grad_param_transform.shape[1] ))
                        hf.create_dataset('grad_score', data=data_grad, maxshape=(None, data_grad.shape[1] ))
                
                if cache.fullTrainingData:

                    for filename, chroma in result_data['results'].items():
                        hf[filename].resize((hf[filename].shape[0] + len(chroma)), axis = 0)
                        hf[filename][-len(chroma):] = chroma

        result_data['input'] = []
        result_data['strategy'] = []
        result_data['mean'] = []
        result_data['confidence'] = []
        result_data['output'] = []
        result_data['output_meta'] = []
        result_data['input_transform'] = []
        result_data['input_transform_extended'] = []
        result_data['results'] = {}
        result_data['times'] = {}

        if 'mcmc_score' in result_data:
            result_data['mcmc_score'] = []

    with cache.progress_path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        #row, col = data.shape
        meta_mean = numpy.mean(data_meta, 0)
        meta_max = numpy.max(data_meta, 0)

        if len(data) and data.ndim > 1:

            population_average = numpy.mean(data)
            population_average_best = numpy.max(numpy.mean(data, 1))

            population_min = numpy.min(data)
            population_min_best = numpy.max(numpy.min(data, 1))

            population_product = numpy.prod(data)**(1.0/data.size)
            population_product_best = numpy.max(numpy.prod(data,1)**(1.0/data.shape[1]))

            line_format = 'Generation: %s \tPopulation: %s \tAverage Score: %.3g \tBest: %.3g \tMinimum Score: %.3g \tBest: %.3g \tProduct Score: %.3g \tBest: %.3g'
 
            if line_log:
                scoop.logger.info(line_format, generation, len(population),
                      RoundToSigFigs(population_average,3), RoundToSigFigs(population_average_best,3),
                      RoundToSigFigs(population_min,3), RoundToSigFigs(population_min_best,3),
                      RoundToSigFigs(population_product,3), RoundToSigFigs(population_product_best,3))
        else:
            if line_log:
                scoop.logger.info("Generation: %s \tPopulation: %s \t No Stats Avaialable", generation, len(population))
        
        writer.writerow([generation,
                         len(population),
                         len(cache.MIN_VALUE),
                         cache.numGoals,
                         cache.settings.get('searchMethod', 'SPEA2'),
                         len(halloffame),
                         average_score,
                         minimum_score,
                         product_score,
                         meta_mean[2],
                         meta_mean[1],
                         meta_mean[0],
                         now - sim_start,
                         now - generation_start,
                         cpu_time.user + cpu_time.system,
                         cache.lastProgressGeneration,
                         cache.generationsOfProgress])

def metaCSV(cache):
    repeat = int(cache.settings['repeat'])

    generations = []
    timeToComplete = []
    timePerGeneration = []
    totalCPUTime = []
    paretoFront = []
    avergeScore = []
    minimumScore = []
    productScore = []
    paretoAverageScore = []
    paretoMinimumScore = []
    paretoProductScore = []
    population = None
    dimensionIn = None
    dimensionOut = None
    searchMethod = None


    #read base progress csv and append each score
    base_dir = Path(cache.settings['resultsDirOriginal'])
    paths = [base_dir / "progress.csv",]
    for idx in range(repeat):
        paths.append(base_dir / str(idx) / "progress.csv")

    for idx, path in enumerate(paths):
        data = pandas.read_csv(path)

        if idx == 0:
            population = data['Population'].iloc[-1]
            dimensionIn = data['Dimension In'].iloc[-1]
            dimensionOut= data['Dimension Out'].iloc[-1]
            searchMethod = data['Search Method'].iloc[-1] 

        generations.append(data['Generation'].iloc[-1])
        timeToComplete.append(data['Elapsed Time'].iloc[-1])
        timePerGeneration.append(numpy.mean(data['Generation Time']))
        paretoFront.append(data['Pareto Front'].iloc[-1])
        avergeScore.append(data['Average Score'].iloc[-1])
        minimumScore.append(data['Minimum Score'].iloc[-1])
        productScore.append(data['Product Score'].iloc[-1])
        totalCPUTime.append(data['Total CPU Time'].iloc[-1])
        paretoAverageScore = [data['Pareto Mean Average Score'].iloc[-1]]
        paretoMinimumScore = [data['Pareto Mean Minimum Score'].iloc[-1]]
        paretoProductScore = [data['Pareto Mean Product Score'].iloc[-1]]
        paretoProductScore = [data['Pareto Mean Product Score'].iloc[-1]]


        
    meta_progress = base_dir / "meta_progress.csv"
    with meta_progress.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        writer.writerow(['Population', 'Dimension In', 'Dimension Out', 'Search Method',
                         'Generation', 'Generation STDDEV',
                         'Elapsed Time', 'Elapsed Time STDDEV',
                         'Generation Time', 'Geneation Time STDDEV',
                         'Pareto Front', 'Paret Front STDDEV',
                         'Average Score', 'Average Score STDDEV',
                         'Minimum Score', 'Minimum Score STDDEV',
                         'Product Score', 'Product Score STDDEV',
                         'Pareto Mean Average Score', 'Pareto Mean Average Score STDDEV',
                         'Pareto Mean Minimum Score', 'Pareto Mean Minimum Score STDDEV',
                         'Pareto Mean Product Score', 'Pareto Mean Product Score STDDEV',
                         'Total CPU Time', 'Total CPU Time STDDEV'])

        writer.writerow([population, dimensionIn, dimensionOut, searchMethod,
                         numpy.mean(generations), numpy.std(generations),
                         numpy.mean(timeToComplete), numpy.std(timeToComplete),
                         numpy.mean(timePerGeneration), numpy.std(timePerGeneration),
                         numpy.mean(paretoFront), numpy.std(paretoFront),
                         numpy.mean(avergeScore), numpy.std(avergeScore),
                         numpy.mean(minimumScore), numpy.std(minimumScore),
                         numpy.mean(productScore), numpy.std(productScore),
                         numpy.mean(paretoAverageScore), numpy.std(paretoAverageScore),
                         numpy.mean(paretoMinimumScore), numpy.std(paretoMinimumScore),
                         numpy.mean(paretoProductScore), numpy.std(paretoProductScore),
                         numpy.mean(totalCPUTime), numpy.std(totalCPUTime),])

def update_result_data(cache, ind, fit, result_data, results):
    if result_data is not None and results is not None:
        result_data['input'].append(tuple(ind))

        if ind.strategy is not None:
            result_data['strategy'].append(tuple(ind.strategy))
        if ind.mean is not None:
            result_data['mean'].append(tuple(ind.mean))
        if ind.confidence is not None:
            result_data['confidence'].append(tuple(ind.confidence))
        result_data['output'].append(tuple(fit))
        result_data['output_meta'].append(tuple(calcMetaScores(fit, cache)))

        for result in results.values():
            result_data['input_transform'].append(tuple(result['cadetValues']))
            result_data['input_transform_extended'].append(tuple(result['cadetValuesKEQ']))

            #All results have the same parameter set so we only need the first one
            break

        if cache.fullTrainingData:
            if 'results' not in result_data:
                result_data['results'] = {}

            if 'times' not in result_data:
                result_data['times'] = {}

            for experimentName, experiment in results.items():
                sim = experiment['simulation']
                times = sim.root.output.solution.solution_times

                timeName = '%s_time' % experimentName

                if timeName not in result_data['times']:
                    result_data['times'][timeName] = times

                for unitName, unit in sim.root.output.solution.items():
                    if unitName.startswith('unit_') and sim.root.input.model[unitName].unit_type == b'OUTLET':
                        for solutionName, solution in unit.items():
                            if solutionName.startswith('solution_outlet_comp'):
                                comp = solutionName.replace('solution_outlet_comp_', '')

                                name = '%s_%s_%s' % (experimentName, unitName, comp)

                                if name not in result_data['results']:
                                    result_data['results'][name] = []

                                result_data['results'][name].append(tuple(solution))

def calcFitness(scores, cache):
    return tuple(numpy.array(scores)[cache.meta_mask])

def process_population(toolbox, cache, population, fitnesses, writer, csvfile, halloffame, meta_hof, generation, result_data=None):
    csv_lines = []
    meta_csv_lines = []

    made_progress = False

    for ind, result in zip(population, fitnesses):
        fit, csv_line, results = result

        csv_line[2:] = RoundToSigFigs(csv_line[2:], 4)
        
        save_name_base = hashlib.md5(str(list(ind)).encode('utf-8', 'ignore')).hexdigest()
        
        ind.fitness.values = calcFitness(fit, cache)

        ind_meta = toolbox.individualMeta(ind)

        ind_meta.fitness.values = calcMetaScores(fit, cache)
       
        update_result_data(cache, ind, fit, result_data, results)

        if csv_line:
            csv_lines.append([time.ctime(), save_name_base] + csv_line)

            onFront = updateParetoFront(halloffame, ind, cache)
            if onFront and not cache.metaResultsOnly:
                processResults(save_name_base, ind, cache, results)

            onFrontMeta = updateParetoFront(meta_hof, ind_meta, cache)
            if onFrontMeta:
                meta_csv_lines.append([time.ctime(), save_name_base] + csv_line)
                processResultsMeta(save_name_base, ind, cache, results)
                made_progress = True

    writer.writerows(csv_lines)

    #flush before returning
    csvfile.flush()

    if made_progress:
        if generation != cache.lastProgressGeneration:
            cache.generationsOfProgress += 1
            cache.lastProgressGeneration = generation
    elif generation != cache.lastProgressGeneration:
        cache.generationsOfProgress = 0
    
    path_meta_csv = cache.settings['resultsDirMeta'] / 'results.csv'
    with path_meta_csv.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerows(meta_csv_lines)

    cleanupFront(cache, halloffame, meta_hof)
    writeMetaFront(cache, meta_hof, path_meta_csv)

    stalled = (generation - cache.lastProgressGeneration) >= cache.stallGenerations
    stallWarn = (generation - cache.lastProgressGeneration) >= cache.stallCorrect
    progressWarn = cache.generationsOfProgress >= cache.progressCorrect

    return stalled, stallWarn, progressWarn

def eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, generation, result_data=None):
    fitnesses = toolbox.map(toolbox.evaluate, map(list, invalid_ind))

    return process_population(toolbox, cache, invalid_ind, fitnesses, writer, csvfile, halloffame, meta_hof, generation, result_data)

def updateParetoFront(halloffame, offspring, cache):
    before = set(map(tuple, halloffame.items))
    halloffame.update([offspring,], cache)
    after = set(map(tuple, halloffame.items))

    return tuple(offspring) in after and tuple(offspring) not in before

def writeMetaFront(cache, meta_hof, path_meta_csv):
    data = pandas.read_csv(path_meta_csv)

    new_data = []

    allowed = {hashlib.md5(str(list(individual)).encode('utf-8', 'ignore')).hexdigest() for individual in meta_hof.items}

    for index, row in data.iterrows():
        if row[1] in allowed:
            new_data.append(row.to_dict())
           
    new_data = pandas.DataFrame(new_data, columns=data.columns.values)

    new_data.to_csv(path_meta_csv, quoting=csv.QUOTE_ALL, index=False)
    new_data.to_excel(cache.settings['resultsDirMeta'] / 'results.xlsx', index=False)

def processResultsGrad(save_name_base, individual, cache, results):
    saveExperiments(save_name_base, cache.settings, cache.target, results, cache.settings['resultsDirGrad'], '%s_%s_GRAD.h5')

def processResults(save_name_base, individual, cache, results):
    saveExperiments(save_name_base, cache.settings, cache.target, results, cache.settings['resultsDirEvo'], '%s_%s_EVO.h5')
 
def processResultsMeta(save_name_base, individual, cache, results):
    saveExperiments(save_name_base, cache.settings, cache.target, results, cache.settings['resultsDirMeta'], '%s_%s_meta.h5')    

def cleanupFront(cache, halloffame=None, meta_hof=None, grad_hof=None):
    if halloffame is not None:
        cleanDir(Path(cache.settings['resultsDirEvo']), halloffame)
    
    if meta_hof is not None:
        cleanDir(Path(cache.settings['resultsDirMeta']), meta_hof)

    if grad_hof is not None:
        cleanDir(Path(cache.settings['resultsDirGrad']), grad_hof)

def cleanDir(dir, hof):
    #find all items in directory
    paths = dir.glob('*.h5')

    #make set of items based on removing everything after _
    exists = {str(path.name).split('_', 1)[0] for path in paths}

    #make set of allowed keys based on hall of hame
    allowed = {hashlib.md5(str(list(individual)).encode('utf-8', 'ignore')).hexdigest() for individual in hof.items}

    #remove everything not in hall of fame
    remove = exists - allowed

    for save_name_base in remove:
        for path in dir.glob('%s*' % save_name_base):
            path.unlink()

def graph_process(cache, generation, last=0):
    if 'lastGraphTime' not in graph_process.__dict__:
        graph_process.lastGraphTime = time.time()
    if 'lastMetaTime' not in graph_process.__dict__:
        graph_process.lastMetaTime = time.time()

    if 'child' in graph_process.__dict__:
        if graph_process.child.poll() is None:  #This is false if the child has completed
            if last:
                graph_process.child.wait()
            else:
                return

    cwd = str(Path(__file__).parent)

    if cache.graphSpearman:  #This is mostly just for debugging now and is not run async
        ret = subprocess.run([sys.executable, 'graph_spearman.py', str(cache.json_path), str(generation)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        log_subprocess('graph_spearman.py', ret)
    
    if last:
        ret = subprocess.run([sys.executable, '-m', 'scoop', 'generate_graphs.py', str(cache.json_path), str(cache.graphType)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,  cwd=cwd)
        graph_process.lastGraphTime = time.time()
    elif (time.time() - graph_process.lastGraphTime) > cache.graphGenerateTime:
        #graph_process.child = subprocess.Popen([sys.executable, '-m', 'scoop', 'generate_graphs.py', str(cache.json_path), '1'], 
        #    stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
        subprocess.run([sys.executable, '-m', 'scoop', 'generate_graphs.py', str(cache.json_path), str(cache.graphType)], 
            stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
        graph_process.lastGraphTime = time.time()
    else:
        if (time.time() - graph_process.lastMetaTime) > cache.graphMetaTime:
            #graph_process.child = subprocess.Popen([sys.executable, '-m', 'scoop', 'generate_graphs.py', str(cache.json_path), '0'], 
            #    stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
            subprocess.run([sys.executable, '-m', 'scoop', 'generate_graphs.py', str(cache.json_path), '0'], 
                stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
            graph_process.lastMetaTime = time.time()

def log_subprocess(name, ret):
    for line in ret.stdout.splitlines():
        scoop.logger.info('%s stdout: %s', name, line)

    for line in ret.stderr.splitlines():    
        scoop.logger.info('%s stderr: %s', name, line)

def finish(cache):
    graph_process(cache, "Last", last=True)

    if cache.graphSpearman:
        cwd = str(Path(__file__).parent)
        ret = subprocess.run([sys.executable, 'video_spearman.py', str(cache.json_path)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,  cwd=cwd)
        log_subprocess('video_spearman.py', ret)

def find_outliers(data, lower_percent=10, upper_percent=90):
    lb, ub = numpy.percentile(data, [lower_percent, upper_percent], 0)
    selected = (data >= lb) & (data <= ub)
    bools = numpy.all(selected, 1)
    return selected, bools

def get_confidence(data, lb=5, ub=95):
    lb, ub = numpy.percentile(data, [lb, ub], 0)
    selected = (data >= lb) & (data <= ub)
    bools = numpy.all(selected, 1)
    return data[bools, :]

def findOutlets(simulation):
    "find all the outlets from a simulation along with their number of components"
    outlets = []
    for key,value in simulation.root.input.model.items():
        try:
            unitType = value.get('unit_type', None)
        except AttributeError:
            unitType = None
        if unitType == b'OUTLET':
            outlets.append((key, value.ncomp))
    return outlets

def test_eta(eta, xl, xu, size):
    "return delta_q log10 power for each eta to use with distribution"
    x = (xu-xl)/2.0
    
    delta_1 = (x - xl)/(xu - xl)
    delta_2 = (xu - x)/(xu - xl)
    
    print(x, delta_1, delta_2)
    
    rand = numpy.random.rand(size)
    
    mut_pow = 1.0 / (eta + 1.0)
    
    xy = numpy.zeros(size)
    val = numpy.zeros(size)
    delta_q = numpy.zeros(size)
    
    xy[rand < 0.5] = 1.0 - delta_1
    val[rand < 0.5] = 2.0 * rand[rand < 0.5] + (1.0 - 2.0 * rand[rand < 0.5]) * xy[rand < 0.5]**(eta + 1)
    delta_q[rand < 0.5] = val[rand < 0.5]**mut_pow - 1.0
    
    xy[rand >= 0.5] = 1.0 - delta_2
    val[rand >= 0.5] = 2.0 * (1.0 - rand[rand >= 0.5]) + 2.0 * (rand[rand >= 0.5] - 0.5) * xy[rand >= 0.5]**(eta + 1)
    delta_q[rand >= 0.5] = 1.0 - val[rand >= 0.5]**mut_pow
    
    delta_q = delta_q *  (xu - xl)
    
    return delta_q

def confidence_eta(eta, xl, xu):
    seq = test_eta(eta, xl, xu, 100000)
    confidence = numpy.max(numpy.abs(numpy.percentile(seq, [5, 95])))
    mean = numpy.mean(numpy.abs(seq))

    return mean, confidence

def setupSimulation(sim, times):
    "set the user solution times to match the times vector and adjust other sim parameters to required settings"

    try:
        del sim.root.input.solver.user_solution_times
    except KeyError:
        pass

    try:
        del sim.root.output
    except KeyError:
        pass

    sim.root.input.solver.user_solution_times = times
    sim.root.input.solver.sections.section_times[-1] = times[-1]
    sim.root.input['return'].unit_001.write_solution_particle = 0
    sim.root.input['return'].unit_001.write_solution_column_inlet = 1
    sim.root.input['return'].unit_001.write_solution_column_outlet = 1
    sim.root.input['return'].unit_001.write_solution_inlet = 1
    sim.root.input['return'].unit_001.write_solution_outlet = 1
    sim.root.input['return'].unit_001.split_components_data = 0
    sim.root.input.solver.nthreads = 1

def graph_corner_process(cache, last=False, interval=1200):
    if 'last_time' not in graph_corner_process.__dict__:
        graph_corner_process.last_time = time.time()

    if 'child' in graph_corner_process.__dict__:
        if graph_corner_process.child.poll() is None:  #This is false if the child has completed
            if last:
                graph_corner_process.child.wait()
            else:
                return

    cwd = str(Path(__file__).parent)

    if last:
        ret = subprocess.run([sys.executable, '-m', 'scoop', '-n', '1', 'generate_corner_graphs.py', str(cache.json_path),], 
            stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
        graph_corner_process.last_time = time.time()
    elif (time.time() - graph_corner_process.last_time) > interval:
        #graph_corner_process.child = subprocess.Popen([sys.executable, '-m', 'scoop', '-n', '1', 'generate_corner_graphs.py', str(cache.json_path),], 
        #    stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
        subprocess.run([sys.executable, '-m', 'scoop', '-n', '1', 'generate_corner_graphs.py', str(cache.json_path),], 
            stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
        graph_corner_process.last_time = time.time()

def biasSimulation(simulation, experiment, cache):

    bias_simulation = Cadet(simulation.root)

    if cache.errorBias:
        name = experiment['name']

        error_model = None
        for error in cache.settings['kde_synthetic']:
            if error['name'] == name:
                error_model = error
                break

        delay_settings = error_model['delay']
        flow_settings = error_model['flow']
        load_settings = error_model['load']

        nsec = bias_simulation.root.input.solver.sections.nsec

        delays = numpy.ones(nsec) * sum(delay_settings)/2.0    
 
        synthetic_error.pump_delay(bias_simulation, delays)

        flow = numpy.ones(nsec) * flow_settings[0]
    
        synthetic_error.error_flow(bias_simulation, flow)

        load = numpy.ones(nsec) * load_settings[0]
  
        synthetic_error.error_load(bias_simulation, load)

    return bias_simulation