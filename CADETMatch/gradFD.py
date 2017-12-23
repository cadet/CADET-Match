import shutil
import h5py
import util
from pathlib import Path
import scipy.optimize
import numpy
import numpy.linalg
import functools
import operator
import hashlib
import score
import os
import subprocess
import csv
import time
import sys
from cadet import Cadet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from cache import cache

class GradientException(Exception):
    pass

def search(gradCheck, offspring, cache, check_all=False):
    if check_all:
        checkOffspring = offspring
    else:
        checkOffspring = (ind for ind in offspring if util.product_score(ind.fitness.values) > gradCheck)
    newOffspring = cache.toolbox.map(gradSearch, map(list, checkOffspring))

    temp = []
    print("Running gradient check")
    failed = []
    for i in newOffspring:
        if i is None:
            failed.append(1)
        elif i.success:
            a = cache.toolbox.individual_guess(i.x)
            fit = cache.toolbox.evaluate(a)
            failed.append(0)
            a.fitness.values = fit
            print(i.x, fit)
            temp.append(a)
    
    if temp:
        avg, bestMin, bestProd = util.averageFitness(temp)
        print('avg', avg, 'bestMin', bestMin, 'bestProd', bestProd)
        if 0.9 * bestProd > gradCheck:
            gradCheck = 0.9 * bestProd
        #if len(temp) > 0 or all(failed):
        #    gradCheck = (1-gradCheck)/2.0 + gradCheck
        print("Finished running on ", len(temp), " individuals new threshold", gradCheck)
    return gradCheck, temp

def gradSearch(x):
    try:
       val = scipy.optimize.least_squares(fitness_sens_grad, x, jac='3-point', method='trf', bounds=(cache.MIN_VALUE, cache.MAX_VALUE), 
           gtol=1e-14, ftol=1e-5, xtol=1e-14, diff_step=1e-7, x_scale="jac")
       scores = fitness_sens(val.x, finished=1)
       print(val.x, numpy.exp(val.x), val.jac, scores, val.message)
       return val
    except GradientException:
        #If the gradient fails return None as the point so the optimizer can adapt
        print("Gradient Failure")
        print(sys.exc_info()[0])
        return None

def fitness_sens_grad(individual, finished=0):
    return fitness_sens(individual, finished)

def fitness_sens(individual, finished=1):
    if not(util.feasible(individual)):
        return [0.0] * cache.numGoals
    scores = []
    error = 0.0

    results = {}
    for experiment in cache.settings['experiments']:
        result = runExperimentSens(individual, experiment, cache.settings, cache.target)
        if result is not None:
            results[experiment['name']] = result
            scores.extend(results[experiment['name']]['scores'])
            error += results[experiment['name']]['error']
        else:
            raise GradientException("Gradient caused simulation failure, aborting")

    #need
    humanScores = None
    if finished:

        #human scores
        humanScores = numpy.array( [util.product_score(scores), 
                                    min(scores), 
                                    sum(scores)/len(scores), 
                                    numpy.linalg.norm(scores)/numpy.sqrt(len(scores)), 
                                    -error] )

        #save
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
            writer.writerow([time.ctime(), save_name_base, 'GRAD', ''] + ["%.5g" % i for i in cadetValuesKEQ] + ["%.5g" % i for i in scores] + list(humanScores)) 

        notDuplicate = saveExperimentsSens(save_name_base, cache.settings, cache.target, results)
        if notDuplicate:
            plotExperimentsSens(save_name_base, cache.settings, cache.target, results)

    #cleanup
    for result in results.values():
        if result['path']:
            os.remove(result['path'])
    
    return [1.0 - score for score in scores]

def saveExperimentsSens(save_name_base, settings,target, results):
    return util.saveExperiments(save_name_base, settings,target, results, settings['resultsDirGrad'], '%s_%s_GRAD.h5')

def plotExperimentsSens(save_name_base, settings, target, results):
    util.plotExperiments(save_name_base, settings, target, results, settings['resultsDirGrad'], '%s_%s_GRAD.png')

def runExperimentSens(individual, experiment, settings, target):
    if 'simulationSens' not in experiment:
        templatePath = Path(settings['resultsDirMisc'], "template_%s.h5" % experiment['name'])
        templateSim = Cadet()
        templateSim.filename = templatePath
        templateSim.load()
        experiment['simulationSens'] = templateSim

    return util.runExperiment(individual, experiment, settings, target, experiment['simulationSens'], float(experiment['timeout']))