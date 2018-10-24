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

def getKDE(cache):
    scores = generate_data(cache)

    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(scores)

    #do kde stuff
    #return kde
    return kde

def generate_data(cache):
    reference_result = setupReferenceResult(cache)

    variations = 200

    temp = [reference_result]
    for i in range(variations):
        temp.append(mutate(reference_result))

    scores = []
    for first,second in itertools.combinations(temp, 2):
        scores.append(score_sim(first, second, cache))

    scoop.logger.info("scores %s", scores)

    scores = numpy.array(scores)
    numpy.save('scores.npy', scores)
    sys.exit()

    return scores

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


def mutate(reference_result):
    "generate variations of the data with different types of noise"
    result = {}

    for key,value in reference_result.items():
        tempSim = Cadet(value['simulation'].root.copy())

        pump_delay(tempSim)
        pump_flow(tempSim)
        base_noise(tempSim)
        signal_noise(tempSim)

        result[key] = {'simulation':tempSim}
    return result

def pump_delay(tempSim):
    "systemic error related to delays in the pump"
    "assume flat random from 0 to 60s"
    "assume data is mono-spaced in time"
    "time is column 0 and values is column 1"
    times = tempSim.root.output.solution.solution_times

    delay = numpy.random.uniform(0.0, 60.0, 1)
    interval = times[1] - times[0]
    delay = quantize_delay(delay[0], interval)

    for (unitName, solutionName), outlet in get_outputs(tempSim):
        tempSim.root.output.solution[unitName][solutionName] = score.roll(outlet, delay)

def pump_flow(tempSim):
    "random noise related to the pump flow rate"
    "assume 5% margin of error but no other changes to chromatogram"
    "assume change is small enough that the chromatogram shape does not change"
    "can't currently model this without simulation work"
    pass

def base_noise(tempSim):
    "add random noise based on baseline noise"
    "based on looking at experimental data"
    times = tempSim.root.output.solution.solution_times
    
    noise = numpy.random.normal(0.0, 1e-4, len(times))

    for (unitName, solutionName), outlet in get_outputs(tempSim):
        tempSim.root.output.solution[unitName][solutionName] = outlet + noise * max(outlet)

def signal_noise(tempSim):
    "add noise to the signal"
    "based on looking at experimental error about +/- .5%"
    times = tempSim.root.output.solution.solution_times

    noise = numpy.random.normal(1.0, 0.003, len(times))

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