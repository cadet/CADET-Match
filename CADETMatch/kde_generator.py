#This module simulates noise (systemic and random) to use for kernel density estimation based on experimental data

#Currently the randomness is

#Pump flow rate
#Pump delay
#Base noise
#Signal noise

import itertools
import score
from cadet import Cadet
from pathlib import Path
import numpy
import scoop
import sys
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import cross_val_score

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import scipy.optimize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def bandwidth_score(bw, data, store):
    bandwidth = 10**bw[0]
    kde_bw = KernelDensity(kernel='exponential', bandwidth=bandwidth, atol=1e-5, rtol=1e-5)
    scores = cross_val_score(kde_bw, data, cv=3)
    store.append( [bandwidth, -max(scores)] )
    return -max(scores)

def get_bandwidth(scores, cache):
    store = []
    result = scipy.optimize.differential_evolution(bandwidth_score, bounds = [(-5, 1),], 
                                               args = (scores,store,))
    bandwidth = 10**result.x[0]
    scoop.logger.info("selected bandwidth %s", bandwidth)

    store = numpy.array(store)

    mcmcDir = Path(cache.settings['resultsDirMCMC'])
    plt.scatter(numpy.log10(store[:,0]), store[:,1])
    plt.xlabel('bandwidth')
    plt.ylabel('cross_val_score')
    plt.savefig(str(mcmcDir / "log_bandwidth.png" ), bbox_inches='tight')
    plt.close()

    plt.scatter(store[:,0], store[:,1])
    plt.xlabel('bandwidth')
    plt.ylabel('cross_val_score')
    plt.savefig(str(mcmcDir / "bandwidth.png" ), bbox_inches='tight')
    plt.close()

    return bandwidth, store

def getKDE(cache, scores, bw):
    #scores = generate_data(cache)

    kde = KernelDensity(kernel='exponential', bandwidth=bw).fit(scores)

    plotKDE(cache, kde, scores)

    #do kde stuff
    #return kde
    return kde

def generate_data(cache):
    reference_result = setupReferenceResult(cache)

    try:
        variations = cache.settings['kde']['variations']
    except KeyError:
        variations = 100

    temp = [reference_result]
    for i in range(variations):
        temp.append(mutate(cache, reference_result))

    plotVariations(cache, temp)

    scores = []
    for first,second in itertools.combinations(temp, 2):
        scores.append(score_sim(first, second, cache))

    scores = numpy.array(scores)

    mcmcDir = Path(cache.settings['resultsDirMCMC'])
    save_scores = mcmcDir / 'scores_used.npy'

    numpy.save(save_scores, scores)

    bandwidth, store = get_bandwidth(scores, cache)

    writeVariations(cache, scores, bandwidth, temp, store)

    return scores, bandwidth

def writeVariations(cache, scores, bandwidth, simulations, store):

    mcmcDir = Path(cache.settings['resultsDirMCMC'])

    mcmc_kde = mcmcDir / 'mcmc_kde.h5'

    bandwidth = numpy.array(bandwidth, dtype="f8")

    data, times = getData(simulations)

    store = numpy.array(store)

    with h5py.File(mcmc_kde, 'w') as hf:
        hf.create_dataset("bandwidth", data=bandwidth, maxshape=tuple(None for i in range(bandwidth.ndim)), fillvalue=[0])

        hf.create_dataset("scores", data=scores, maxshape=(None, len(scores[0])), compression="gzip")

        hf.create_dataset("bandwidth_scores", data=store, maxshape=(None,2), compression="gzip")

        for name, value in times.items():
            hf.create_dataset(name, data=value, maxshape=(None,), compression="gzip")

        for name, value in data.items():
            hf.create_dataset(name, data=value, maxshape=(None, len(value[0])), compression="gzip")

def getData(simulaltions):

    data = {}
    times = {}

    for sim in simulaltions:
        for experimentName, experiment in sim.items():
            simulation = experiment['simulation']

            key = '%s_%s' % (experimentName, "times")
            if key not in data:
                times[key] = simulation.root.output.solution.solution_times

            for ((unitName, solutionName), solution) in get_outputs(simulation):
                key = '%s_%s_%s' % (experimentName, unitName, solutionName)
                if key not in data:
                    data[key] = []
                data[key].append(solution)

    for key,value in data.items():
        data[key] = numpy.array(value)

    return data, times

def plotVariations(cache, temp):

    data, times = getData(temp)

    mcmcDir = Path(cache.settings['resultsDirMCMC'])

    for key,value in data.items():
        experiment = key.split('_unit', maxsplit=1)[0]

        time = times['%s_times' % experiment]

        mean = numpy.mean(value, 0)
        std = numpy.std(value, 0)
        minValues = numpy.min(value, 0)
        maxValues = numpy.max(value, 0)

        plt.plot(time, mean)
        plt.fill_between(time, mean - std, mean + std,
                    color='green', alpha=0.2)
        plt.fill_between(time, minValues, maxValues,
                    color='red', alpha=0.2)
        plt.xlabel('time(s)')
        plt.ylabel('conc mol/m^3')
        plt.savefig(str(mcmcDir / ("%s.png" % key ) ), bbox_inches='tight')
        plt.close()

        plt.plot(time, value.transpose())
        plt.xlabel('time(s)')
        plt.ylabel('conc mol/m^3')
        plt.savefig(str(mcmcDir / ("%s_lines.png" % key ) ), bbox_inches='tight')
        plt.close()

def plotKDE(cache, kde, scores):
    return None

def setupReferenceResult(cache):
    results = {}
    for experiment in cache.settings['experiments']:
        
        templatePath = experiment['reference']
        templateSim = Cadet()
        templateSim.filename = templatePath
        templateSim.run()
        templateSim.load()
        
        results[experiment['name']] = {'simulation':templateSim}

    return results


def mutate(cache, reference_result):
    "generate variations of the data with different types of noise"
    result = {}

    for key,value in reference_result.items():
        tempSim = Cadet(value['simulation'].root.copy())

        pump_delay(cache, tempSim)
        #pump_flow(cache, tempSim)
        base_noise(cache, tempSim)
        signal_noise(cache, tempSim)

        result[key] = {'simulation':tempSim}
    return result

def pump_delay(cache, tempSim):
    "systemic error related to delays in the pump"
    "assume flat random from 0 to 60s"
    "assume data is mono-spaced in time"
    "time is column 0 and values is column 1"
    times = tempSim.root.output.solution.solution_times

    #delay = numpy.random.uniform(0.0, 60.0, 1)
    
    try:
        pump_mean = cache.settings['kde']['pump_delay_mean']
    except KeyError:
        pump_mean = 0.0

    try:
        pump_std = cache.settings['kde']['pump_delay_std']
    except KeyError:
        pump_std = 1.0

    delay = -1
    while delay < 0:
        delay = numpy.random.normal(pump_mean, pump_std, 1)

    interval = times[1] - times[0]
    delay = quantize_delay(delay[0], interval)

    for (unitName, solutionName), outlet in get_outputs(tempSim):
        tempSim.root.output.solution[unitName][solutionName] = score.roll(outlet, delay)

def pump_flow(cache, tempSim):
    "random noise related to the pump flow rate"
    "assume 5% margin of error but no other changes to chromatogram"
    "assume change is small enough that the chromatogram shape does not change"
    "can't currently model this without simulation work"
    pass

def base_noise(cache, tempSim):
    "add random noise based on baseline noise"
    "based on looking at experimental data"
    times = tempSim.root.output.solution.solution_times
    
    try:
        noise_std = cache.settings['kde']['base_noise_std']
    except KeyError:
        noise_std = 1e-7

    noise = numpy.random.normal(0.0, noise_std, len(times))

    for (unitName, solutionName), outlet in get_outputs(tempSim):
        tempSim.root.output.solution[unitName][solutionName] = outlet + noise * max(outlet)

def signal_noise(cache, tempSim):
    "add noise to the signal"
    "based on looking at experimental error about +/- .5%"
    times = tempSim.root.output.solution.solution_times

    #0.003 base on experiments

    try:
        noise_std = cache.settings['kde']['signal_noise_std']
    except KeyError:
        noise_std = 0.003

    noise = numpy.random.normal(1.0, noise_std, len(times))

    for (unitName, solutionName), outlet in get_outputs(tempSim):
        tempSim.root.output.solution[unitName][solutionName] = outlet * noise

def get_outputs(tempSim):
    "get the outputs for tempSim so they can be mutated"
    temp = []
    for unitName, unit in tempSim.root.output.solution.items():
        if "unit" in unitName:
            for solutionName, solution in unit.items():
                if 'outlet' in solutionName:
                    temp.append( ((unitName, solutionName), solution) )
    return temp

def quantize_delay(delay, interval):
    return int(round(delay/float(interval))*float(interval)/interval)

def score_sim(first,second, cache):
    "score first vs second simulation"
    target = {}
    for experiment in cache.settings['experiments']:
        target[experiment["name"]] = cache.setupExperiment(experiment, first[experiment["name"]]['simulation'], dataFromSim=1)

    score_sim = []
    diff = 0
    for experimentName in cache.settings['experiments']:
        experimentName = experiment['name']

        firstSim = first[experimentName]['simulation']
        secondSim = second[experimentName]['simulation']
        for (unitName, solutionName), outlet in get_outputs(firstSim):
            outlet_second = secondSim.root.output.solution[unitName][solutionName]
            diff = diff + numpy.sum( (outlet-outlet_second)**2 )
            a = 1

        for feature in experiment['features']:
            featureType = feature['type']
            featureName = feature['name']
            if featureType in cache.scores:
                scores, sse, sse_count = cache.scores[featureType].run(second[experimentName], target[experiment['name']][featureName])
                score_sim.extend(scores)

    return score_sim