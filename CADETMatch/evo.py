import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
import pandas
import util
import score

import functools
import subprocess
import json
import csv
import h5py
import operator
import hashlib
import time
import sys
import pickle

import scipy.interpolate
import math
import array
import random

from pathlib import Path

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import os
import shutil

import spea2
import nsga2
import nsga3
import multistart
from cadet import Cadet

#parallelization
from scoop import futures

import scipy.signal

from cache import cache

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
        return cache.WORST

    

    scores = []
    error = 0.0

    results = {}
    for experiment in cache.settings['experiments']:
        result = runExperiment(individual, experiment, cache.settings, cache.target)
        if result is not None:
            results[experiment['name']] = result
            scores.extend(results[experiment['name']]['scores'])
            error += results[experiment['name']]['error']
        else:
            return cache.WORST

    #need

    #human scores
    humanScores = numpy.array( [util.product_score(scores), 
                                min(scores), sum(scores)/len(scores), 
                                numpy.linalg.norm(scores)/numpy.sqrt(len(scores)), 
                                -error] )

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
    save_name_base = hashlib.md5(str(individual).encode('utf-8','ignore')).hexdigest()

    for result in results.values():
        if result['cadetValuesKEQ']:
            cadetValuesKEQ = result['cadetValuesKEQ']
            break

    #generate csv
    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow([time.ctime(), save_name_base, 'EVO', 'NA'] + 
                        ["%.5g" % i for i in cadetValuesKEQ] + 
                        ["%.5g" % i for i in scores] + 
                        list(humanScores)) 

    #print('keep_result', keep_result)
    if keep_result:
        notDuplicate = saveExperiments(save_name_base, cache.settings, cache.target, results)
        #print('notDuplicate', notDuplicate)
        if notDuplicate:
            plotExperiments(save_name_base, cache.settings, cache.target, results)

    #cleanup
    for result in results.values():
        if result['path']:
            os.remove(result['path'])
            
    return scores

def saveExperiments(save_name_base, settings,target, results):
    return util.saveExperiments(save_name_base, settings,target, results, settings['resultsDirEvo'], '%s_%s_EVO.h5')

def plotExperiments(save_name_base, settings, target, results):
    util.plotExperiments(save_name_base, settings, target, results, settings['resultsDirEvo'], '%s_%s_EVO.png')

def runExperiment(individual, experiment, settings, target):
    if 'simulation' not in experiment:
        templatePath = Path(settings['resultsDirMisc'], "template_%s.h5" % experiment['name'])
        templateSim = Cadet()
        templateSim.filename = templatePath
        templateSim.load()
        experiment['simulation'] = templateSim

    return util.runExperiment(individual, experiment, settings, target, experiment['simulation'], float(experiment['timeout']))

def run(cache):
    "run the parameter estimation"
    searchMethod = cache.settings.get('searchMethod', 'SPEA2')
    if searchMethod == 'SPEA2':
        return spea2.run(cache, tools, creator)
    if searchMethod == 'NSGA2':
        return nsga2.run(cache, tools, creator)
    if searchMethod == 'NSGA3':
        return nsga3.run(cache, tools, creator)
    if searchMethod == 'Multistart':
        return multistart.run(cache, tools, creator)
