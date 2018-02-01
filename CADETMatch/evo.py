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
            return cache.WORST, [], None

    #need
    scores = util.RoundToSigFigs(scores, 3)

    #human scores
    humanScores = numpy.array( [util.product_score(scores), 
                                min(scores), sum(scores)/len(scores), 
                                numpy.linalg.norm(scores)/numpy.sqrt(len(scores)), 
                                error] )

    humanScores = util.RoundToSigFigs(humanScores, 3)

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
