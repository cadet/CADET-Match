import math
import numpy
import pandas

import hashlib

from deap import tools
import scipy.signal
from scipy.spatial.distance import cdist
from pathlib import Path

from addict import Dict

import tempfile
import os
from cadet import Cadet
import subprocess
import sys
import json
import itertools
import time
import csv
import psutil

from scoop import futures
import random
import calc_coeff
import h5py

import decimal as decim
decim.getcontext().prec = 64
__logBase10of2_decim = decim.Decimal(2).log10()
__logBase10of2 = float(__logBase10of2_decim)

import SALib.sample.sobol_sequence

def smoothing_factor(y):
    return max(y)/1000000.0

def find_extreme(seq):
    try:
        return max(seq, key=lambda x: abs(x[1]))
    except ValueError:
        return [0, 0]

def get_times_values(simulation, target, selected = None):

    times = simulation.root.output.solution.solution_times

    isotherm = target['isotherm']

    if isinstance(isotherm, list):
        values = numpy.sum([simulation[i] for i in isotherm], 0) * target['factor']
    else:
        values = simulation[isotherm] * target['factor']
    
    if selected is None:
        selected = target['selected']

    return times[selected], values[selected]

def sse(data1, data2):
    return numpy.sum( (data1 - data2)**2 )

def find_peak(times, data):
    "Return tuples of (times,data) for the peak we need"
    #[highs, lows] = peakdetect.peakdetect(data, times, 1)

    #return find_extreme(highs), find_extreme(lows)

    minIdx = numpy.argmin(data)
    maxIdx = numpy.argmax(data)

    return (times[maxIdx], data[maxIdx]), (times[minIdx], data[minIdx])

def find_breakthrough(times, data):
    "return tupe of time,value for the start breakthrough and end breakthrough"
    selected = data > 0.999 * max(data)
    selected_times = times[selected]
    return (selected_times[0], max(data)), (selected_times[-1], max(data))

def generateIndividual(icls, size, imin, imax, cache):
    if cache.roundParameters is not None:
        return icls(RoundToSigFigs(numpy.random.uniform(imin, imax), cache.roundParameters))
    else:
        return icls(numpy.random.uniform(imin, imax))

def initIndividual(icls, cache, content):
    if cache.roundParameters is not None:
        return icls(RoundToSigFigs(content, cache.roundParameters))
    else:
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

print_log = 0

def log(*args):
    if print_log:
        print(args)

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

    for i in offspring:
        total += sum(i.fitness.values)
        number += len(i.fitness.values)
        bestMin = max(bestMin, min(i.fitness.values))
        bestProd = max(bestProd, product_score(i.fitness.values))

    result = [total/number, bestMin, bestProd]

    if cache.roundScores is not None:
        return RoundToSigFigs(result, cache.roundScores)
    else:
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
    a,b = calc_coeff.exponential_coeff(0.0, 1, 0.99, 100)
    eta = calc_coeff.exponential(mult, a, b) * eta

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
            simulation.filename = bytes(dst)

            for (header, score) in zip(experiment['headers'], results[experimentName]['scores']):
                simulation.root.score[header] = score
            simulation.save()

    return True

def convert_individual(individual, cache):
    cadetValues = []

    idx = 0
    for parameter in cache.settings['parameters']:
        transform = parameter['transform']
        count = cache.transforms[transform].count
        seq = individual[idx:idx+count]
        values, headerValues = cache.transforms[transform].untransform(seq, cache, parameter, False)
        cadetValues.extend(values)
        idx += count

    return cadetValues

def set_simulation(individual, simulation, settings, cache, fullPrecision):
    log("individual", individual)

    cadetValues = []
    cadetValuesKEQ = []

    idx = 0
    for parameter in settings['parameters']:
        transform = parameter['transform']
        count = cache.transforms[transform].count
        seq = individual[idx:idx+count]
        values, headerValues = cache.transforms[transform].setSimulation(simulation, parameter, seq, cache, fullPrecision)
        cadetValues.extend(values)
        cadetValuesKEQ.extend(headerValues)
        idx += count

    log("finished setting hdf5")
    return cadetValues, cadetValuesKEQ

def getBoundOffset(unit):
    if unit.unit_type == b'CSTR':
        NBOUND = unit.nbound
    else:
        NBOUND = unit.discretization.nbound

    boundOffset = numpy.cumsum(numpy.concatenate([[0,], NBOUND]))
    return boundOffset

def runExperiment(individual, experiment, settings, target, template_sim, timeout, cache, fullPrecision=False):
    handle, path = tempfile.mkstemp(suffix='.h5')
    os.close(handle)

    simulation = Cadet(template_sim.root)
    simulation.filename = path

    simulation.root.input.solver.nthreads = int(settings.get('nThreads', 1))
    cadetValues, cadetValuesKEQ = set_simulation(individual, simulation, settings, cache, fullPrecision)

    simulation.save()

    def leave():
        os.remove(path)
        return None

    try:
        simulation.run(timeout = timeout, check=True)
    except subprocess.TimeoutExpired:
        print("Simulation Timed Out")
        return leave()
    except subprocess.CalledProcessError as error:
        print("The simulation failed")
        logError(cache, cadetValuesKEQ, error)
        return leave()

    #read sim data
    simulation.load()

    simulationFailed = isinstance(simulation.root.output.solution.solution_times, Dict)
    if simulationFailed:
        log(individual, "sim must have failed", path)
        return leave()
    log("Everything ran fine")


    temp = {}
    temp['simulation'] = simulation
    temp['path'] = path
    temp['scores'] = []
    temp['error'] = 0.0
    temp['error_count'] = 0.0
    temp['cadetValues'] = cadetValues
    temp['cadetValuesKEQ'] = cadetValuesKEQ

    for feature in experiment['features']:
        start = float(feature['start'])
        stop = float(feature['stop'])
        featureType = feature['type']
        featureName = feature['name']

        if featureType in cache.scores:
            scores, sse, sse_count = cache.scores[featureType].run(temp, target[experiment['name']][featureName])
 
        temp['scores'].extend(scores)
        temp['error'] += sse
        temp['error_count'] += sse_count

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

        baseDir = Path(settings['resultsDir']) / str(idx)
        baseDir.mkdir(parents=True, exist_ok=True)

        settings['resultsDirOriginal'] = settings['resultsDir']
        settings['resultsDir'] = str(baseDir)

        new_settings_file = baseDir / settings_file.name
        with new_settings_file.open(mode="w") as json_data:
            json.dump(settings, json_data)
        return new_settings_file

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
    a = numpy.array(convert_individual(a,cache))
    b = numpy.array(convert_individual(b,cache))
    diff = numpy.abs((a-b)/a)
    return numpy.all(diff < 1e-3)

def RoundOffspring(cache, offspring, hof):
    for child in offspring:
        if cache.roundParameters is not None:
            temp = RoundToSigFigs(child, cache.roundParameters)
        else:
            temp = numpy.array(child)
        if all(child == temp):
            pass
        else:
            for idx, i in enumerate(temp):
                child[idx] = i
            del child.fitness.values

    #make offspring unique
    unique = set()
    new_offspring = []
    for child in offspring:
        key = tuple(child)
        if key not in unique:
            new_offspring.append(child)
            unique.add(key)

    #population size needs to remain the same so add more children randomly if we have removed duplicates
    while len(new_offspring) < len(offspring):
        if len(hof):
            ind = hof[random.sample(range(len(hof)), 1)[0]]
        else:
            ind = cache.toolbox.individual()

        child = cache.toolbox.clone(ind)
        cache.toolbox.force_mutate(child)

        key = tuple(child)
        if key not in unique:
            new_offspring.append(child)
            unique.add(key)

    return new_offspring

def RoundToSigFigs( x, sigfigs ):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.
    Return value has the same type as x.
    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value or an array like object containing only real values.
    """
    if not ( type(sigfigs) is int or type(sigfigs) is long or
             isinstance(sigfigs, numpy.integer) ):
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

def writeProgress(cache, generation, population, halloffame, meta_halloffame, grad_halloffame, average_score, minimum_score, product_score, sim_start, generation_start, training=None):
    cpu_time = psutil.Process().cpu_times()
    now = time.time()

    results = Path(cache.settings['resultsDirBase'])

    hof = results / "hof.npy"
    meta_hof = results / "meta_hof.npy"
    grad_hof = results / "grad_hof.npy"

    data = numpy.array([i.fitness.values for i in halloffame])
    data_meta = numpy.array([i.fitness.values for i in halloffame])
    data_grad = numpy.array([i.fitness.values for i in halloffame])


    #data distances
    dist = scipy.spatial.distance.cdist(data, data)
    if len(dist):
        dist_std = numpy.std(dist)
        dist_min = numpy.min(dist[dist>0])
        dist_max = numpy.max(dist)
        dist_mean = numpy.mean(dist)

        dist_info = "Diversity Std \t %.3f Min \t %.3f Mean \t %.3f Max \t %.3f" % (dist_std, dist_min, dist_mean, dist_max)
    else:
        dist_info = "No Diversity Information"

    with hof.open('wb') as hof_file:
        numpy.save(hof_file, data)

    with meta_hof.open('wb') as hof_file:
        numpy.save(hof_file, data_meta)

    with grad_hof.open('wb') as hof_file:
        numpy.save(hof_file, data_grad)

    gen_data = numpy.array([generation, len(training['input'])]).reshape(1,2)

    if training is not None:
        trainingDir = Path(cache.settings['resultsDirTraining'])

        training_h5 = trainingDir / "training.h5"

        if not training_h5.exists():
            with h5py.File(training_h5, 'w') as hf:
                hf.create_dataset("input", data=training['input'], maxshape=(None, len(training['input'][0])), compression="gzip")
                hf.create_dataset("output", data=training['output'], maxshape=(None, len(training['output'][0])), compression="gzip")
                hf.create_dataset("output_meta", data=training['output_meta'], maxshape=(None, len(training['output_meta'][0])), compression="gzip")
                hf.create_dataset("generation", data=gen_data, maxshape=(None, 2), compression="gzip")
                
                if cache.fullTrainingData:

                    for filename, chroma in training['results'].items():
                        hf.create_dataset(filename, data=chroma, maxshape=(None, len(chroma[0])), compression="gzip")

                    for filename, chroma in training['times'].items():
                        hf.create_dataset(filename, data=chroma)
        else:
            with h5py.File(training_h5, 'a') as hf:
                hf["input"].resize((hf["input"].shape[0] + len(training['input'])), axis = 0)
                hf["input"][-len(training['input']):] = training['input']

                hf["output"].resize((hf["output"].shape[0] + len(training['output'])), axis = 0)
                hf["output"][-len(training['output']):] = training['output']

                hf["output_meta"].resize((hf["output_meta"].shape[0] + len(training['output_meta'])), axis = 0)
                hf["output_meta"][-len(training['output_meta']):] = training['output_meta']

                hf["generation"].resize((hf["generation"].shape[0] + 1), axis = 0)
                hf["generation"][-1] = gen_data
                
                if cache.fullTrainingData:

                    for filename, chroma in training['results'].items():
                        hf[filename].resize((hf[filename].shape[0] + len(chroma)), axis = 0)
                        hf[filename][-len(chroma):] = chroma

        training['input'] = []
        training['output'] = []
        training['output_meta'] = []
        training['results'] = {}
        training['times'] = {}

    with cache.progress_path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)

        row, col = data.shape
        data_mean = numpy.mean(data_meta, 1)
        data_mean_mean = numpy.mean(data_mean)
        data_mean_best = numpy.max(data_mean)

        data_min = numpy.min(data_meta, 1)
        data_min_mean = numpy.mean(data_min)
        data_min_best = numpy.max(data_min)

        data_prod = numpy.power(numpy.prod(data_meta, 1), 1.0/col)
        data_prod_mean = numpy.mean(data_prod)
        data_prod_best = numpy.max(data_prod)
 
        print("Generation: ", generation, "\tPopulation: ", len(population),
              "\tAverage Score: %.4f \tBest: %.4f" % (RoundToSigFigs(data_mean_mean,4), RoundToSigFigs(data_mean_best,4)),
              "\tMinimum Score: %.4f \tBest: %.4f" % (RoundToSigFigs(data_min_mean,4), RoundToSigFigs(data_min_best,4)),
              "\tProduct Score: %.4f \tBest: %.4f" % (RoundToSigFigs(data_prod_mean,4), RoundToSigFigs(data_prod_best,4)),
              "\n\t", dist_info)
        
        writer.writerow([generation,
                         len(population),
                         len(cache.MIN_VALUE),
                         cache.numGoals,
                         cache.settings.get('searchMethod', 'SPEA2'),
                         len(halloffame),
                         average_score,
                         minimum_score,
                         product_score,
                         data_mean_mean,
                         data_min_mean,
                         data_prod_mean,
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

def meta_calc(scores):
    return [product_score(scores), 
            numpy.min(scores), 
            numpy.sum(scores)/len(scores), 
            numpy.linalg.norm(scores)/numpy.sqrt(len(scores))]

def update_training(cache, ind, fit, training, results):
    if training is not None and results is not None:
        training['input'].append(tuple(ind))
        training['output'].append(tuple(fit))
        training['output_meta'].append(tuple(meta_calc(fit)))

        if cache.fullTrainingData:
            if 'results' not in training:
                training['results'] = {}

            if 'times' not in training:
                training['times'] = {}

            for experimentName, experiment in results.items():
                sim = experiment['simulation']
                times = sim.root.output.solution.solution_times

                timeName = '%s_time' % experimentName

                if timeName not in training['times']:
                    training['times'][timeName] = times

                for unitName, unit in sim.root.output.solution.items():
                    if unitName.startswith('unit_') and sim.root.input.model[unitName].unit_type == b'OUTLET':
                        for solutionName, solution in unit.items():
                            if solutionName.startswith('solution_outlet_comp'):
                                comp = solutionName.replace('solution_outlet_comp_', '')

                                name = '%s_%s_%s' % (experimentName, unitName, comp)

                                if name not in training['results']:
                                    training['results'][name] = []

                                training['results'][name].append(tuple(solution))

def process_population(toolbox, cache, population, fitnesses, writer, csvfile, halloffame, meta_hof, generation, training=None):
    csv_lines = []
    meta_csv_lines = []

    made_progress = False

    for ind, result in zip(population, fitnesses):
        fit, csv_line, results = result
        
        save_name_base = hashlib.md5(str(list(ind)).encode('utf-8', 'ignore')).hexdigest()
        
        ind.fitness.values = fit

        ind_meta = toolbox.clone(ind)
        ind_meta.fitness.values = meta_calc(fit)

        update_training(cache, ind, fit, training, results)

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

            cleanupProcess(results)

    writer.writerows(csv_lines)

    if made_progress:
        if generation != cache.lastProgressGeneration:
            cache.generationsOfProgress += 1
            cache.lastProgressGeneration = generation
    elif generation != cache.lastProgressGeneration:
        cache.generationsOfProgress = 0
    
    #flush before returning
    csvfile.flush()

    path_meta_csv = cache.settings['resultsDirMeta'] / 'results.csv'
    with path_meta_csv.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerows(meta_csv_lines)

    #print("Current front", len(halloffame))
    cleanupFront(cache, halloffame, meta_hof)
    writeMetaFront(cache, meta_hof, path_meta_csv)

    stalled = (generation - cache.lastProgressGeneration) >= cache.stallGenerations
    stallWarn = (generation - cache.lastProgressGeneration) >= cache.stallCorrect
    progressWarn = cache.generationsOfProgress >= cache.progressCorrect
    return stalled, stallWarn, progressWarn

def eval_population(toolbox, cache, invalid_ind, writer, csvfile, halloffame, meta_hof, generation, training=None):
    fitnesses = toolbox.map(toolbox.evaluate, map(list, invalid_ind))

    return process_population(toolbox, cache, invalid_ind, fitnesses, writer, csvfile, halloffame, meta_hof, generation, training)

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
    notDuplicate = saveExperiments(save_name_base, cache.settings, cache.target, results, cache.settings['resultsDirGrad'], '%s_%s_GRAD.h5')

def processResults(save_name_base, individual, cache, results):
    notDuplicate = saveExperiments(save_name_base, cache.settings, cache.target, results, cache.settings['resultsDirEvo'], '%s_%s_EVO.h5')
 
def processResultsMeta(save_name_base, individual, cache, results):
    notDuplicate = saveExperiments(save_name_base, cache.settings, cache.target, results, cache.settings['resultsDirMeta'], '%s_%s_meta.h5')    

def cleanupProcess(results):
    #cleanup
    for result in results.values():
        if result['path']:
            os.remove(result['path'])

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
    if cache.lastGraphTime is None:
        cache.lastGraphTime = time.time()
    if cache.lastMetaTime is None:
        cache.lastMetaTime = time.time()

    cwd = str(Path(__file__).parent)

    subprocess.run([sys.executable, 'graph_spearman.py', str(cache.json_path), str(generation)], 
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, cwd=cwd)
    
    if last or (time.time() - cache.lastGraphTime) > cache.graphGenerateTime:
        subprocess.run([sys.executable, '-m', 'scoop', 'generate_graphs.py', str(cache.json_path), '1'], 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, cwd=cwd)
        cache.lastGraphTime = time.time()
    else:
        if (time.time() - cache.lastMetaTime) > cache.graphMetaTime:
            subprocess.run([sys.executable, '-m', 'scoop', 'generate_graphs.py', str(cache.json_path), '0'], 
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, cwd=cwd)
            cache.lastMetaTime = time.time()

def finish(cache):
    graph_process(cache, "Last", last=True)

    cwd = str(Path(__file__).parent)
    subprocess.run([sys.executable, 'video_spearman.py', str(cache.json_path)], 
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, cwd=cwd)