import numpy
import util

#import csv
import hashlib
import time

from pathlib import Path

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import os

from cadet import Cadet

from cache import cache

import copy

ERROR = {'scores': None,
         'path': None,
         'simulation' : None,
         'error': None,
         'cadetValues':None,
         'cadetValuesKEQ': None}

def fitness(individual, json_path):
    if json_path != cache.json_path:
        cache.setup(json_path)
    
    if not(util.feasible(individual)):
        return cache.WORST, []

    #return numpy.random.uniform(cache.WORST, [1] * len(cache.WORST)), [0.5 for i in range(60)]

    scores = []
    error = 0.0

    results = {}
    for experiment in cache.settings['experiments']:
        result = runExperiment(individual, experiment, cache.settings, cache.target, cache)
        if result is not None:
            results[experiment['name']] = result
            scores.extend(results[experiment['name']]['scores'])
            error += results[experiment['name']]['error']
        else:
            return cache.WORST, []

    #need

    scores = util.RoundToSigFigs(scores, 4)

    #human scores
    humanScores = numpy.array( [util.product_score(scores), 
                                min(scores), sum(scores)/len(scores), 
                                numpy.linalg.norm(scores)/numpy.sqrt(len(scores)), 
                                -error] )

    humanScores = util.RoundToSigFigs(humanScores, 4)

    #best
    cache.target['bestHumanScores'] = numpy.max(numpy.vstack([cache.target['bestHumanScores'], humanScores]), 0)

    #save
    keepTop = cache.settings['keepTop']

    keep_result = 0
    if any(humanScores >= (keepTop * cache.target['bestHumanScores'])):
        keep_result = 1

    #flip sign of SSE for writing out to file
    humanScores[-1] = -1 * humanScores[-1]

    #generate save name
    save_name_base = hashlib.md5(str(individual).encode('utf-8', 'ignore')).hexdigest()

    for result in results.values():
        if result['cadetValuesKEQ']:
            cadetValuesKEQ = result['cadetValuesKEQ']
            break

    #generate csv
    csv_record = []
    csv_record.extend([time.ctime(), save_name_base, 'EVO', 'NA'])
    csv_record.extend(["%.5g" % i for i in cadetValuesKEQ])
    csv_record.extend(["%.5g" % i for i in scores])
    csv_record.extend(["%.5g" % i for i in humanScores])

    if keep_result:
        notDuplicate = saveExperiments(save_name_base, cache.settings, cache.target, results)
        if notDuplicate:
            plotExperiments(save_name_base, cache.settings, cache.target, results)

    #cleanup
    for result in results.values():
        if result['path']:
            os.remove(result['path'])
       
    return scores, csv_record, results

def saveExperiments(save_name_base, settings, target, results):
    return util.saveExperiments(save_name_base, settings, target, results, settings['resultsDirEvo'], '%s_%s_EVO.h5')

def plotExperiments(save_name_base, settings, target, results):
    util.plotExperiments(save_name_base, settings, target, results, settings['resultsDirEvo'], '%s_%s_EVO.png')

def runExperiment(individual, experiment, settings, target, cache):
    if 'simulation' not in experiment:
        templatePath = Path(settings['resultsDirMisc'], "template_%s.h5" % experiment['name'])
        templateSim = Cadet()
        templateSim.filename = templatePath
        templateSim.load()
        experiment['simulation'] = templateSim

    return util.runExperiment(individual, experiment, settings, target, experiment['simulation'], float(experiment['timeout']), cache)

def run(cache):
    "run the parameter estimation"
    searchMethod = cache.settings.get('searchMethod', 'SPEA2')
    return cache.search[searchMethod].run(cache, tools, creator)
