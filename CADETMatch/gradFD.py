import util
from pathlib import Path
import scipy.optimize
import numpy
import numpy.linalg
import hashlib
import os
import csv
import time
import sys

from cadet import Cadet

from cache import cache

class GradientException(Exception):
    pass

def search(gradCheck, offspring, cache, writer, csvfile, check_all=False):
    if check_all:
        checkOffspring = offspring
    else:
        checkOffspring = (ind for ind in offspring if util.product_score(ind.fitness.values) > gradCheck)
    newOffspring = cache.toolbox.map(cache.toolbox.evaluate_grad, map(list, checkOffspring))

    temp = []
    failed = []
    csv_lines = []

    for i in newOffspring:
        #print(i, dir(i))
        if i is None:
            failed.append(1)
        elif i.success:
            a = cache.toolbox.individual_guess(i.x)
            fit, csv_line, results = cache.toolbox.evaluate(a)

            csv_line[0] = 'GRAD'

            save_name_base = hashlib.md5(str(list(i.x)).encode('utf-8', 'ignore')).hexdigest()

            if csv_line:
                csv_lines.append([time.ctime(), save_name_base] + csv_line)

            failed.append(0)
            a.fitness.values = fit
            #print(i.x, fit)
            temp.append(a)
    
    if temp:
        avg, bestMin, bestProd = util.averageFitness(temp, cache)
        #print('avg', avg, 'bestMin', bestMin, 'bestProd', bestProd)
        if 0.9 * bestProd > gradCheck:
            gradCheck = 0.9 * bestProd
        #if len(temp) > 0 or all(failed):
        #    gradCheck = (1-gradCheck)/2.0 + gradCheck
        #print("Finished running on ", len(temp), " individuals new threshold", gradCheck)

    writer.writerows(csv_lines)
    
    #flush before returning
    csvfile.flush()

    return gradCheck, temp

def gradSearch(x, json_path):
    if json_path != cache.json_path:
        cache.setup(json_path)

    try:
        val = scipy.optimize.least_squares(fitness_sens_grad, x, jac='3-point', method='trf', bounds=(cache.MIN_VALUE, cache.MAX_VALUE), 
            gtol=1e-14, ftol=1e-5, xtol=1e-14, diff_step=1e-7, x_scale="jac")
        #scores = fitness_sens(val.x, finished=1)
        #print(val.x, numpy.exp(val.x), val.jac, scores, val.message)
        return val
    except GradientException:
        #If the gradient fails return None as the point so the optimizer can adapt
        #print("Gradient Failure")
        #print(sys.exc_info()[0])
        return None
    except ValueError:
        print('Gradient failure start point', x, cache.MIN_VALUE, cache.MAX_VALUE)
        return None

def fitness_sens_grad(individual, finished=0):
    return fitness_sens(individual, finished)

def fitness_sens(individual, finished=1):
    scores = []
    error = 0.0

    results = {}
    for experiment in cache.settings['experiments']:
        result = runExperimentSens(individual, experiment, cache.settings, cache.target, cache)
        if result is not None:
            results[experiment['name']] = result
            scores.extend(results[experiment['name']]['scores'])
            error += results[experiment['name']]['error']
        else:
            raise GradientException("Gradient caused simulation failure, aborting")

    #cleanup
    for result in results.values():
        if result['path']:
            os.remove(result['path'])
    
    return [1.0 - score for score in scores]

def saveExperimentsSens(save_name_base, settings, target, results):
    return util.saveExperiments(save_name_base, settings, target, results, settings['resultsDirGrad'], '%s_%s_GRAD.h5')

def runExperimentSens(individual, experiment, settings, target, cache):
    if 'simulationSens' not in experiment:
        templatePath = Path(settings['resultsDirMisc'], "template_%s.h5" % experiment['name'])
        templateSim = Cadet()
        templateSim.filename = templatePath
        templateSim.load()
        experiment['simulationSens'] = templateSim

    return util.runExperiment(individual, experiment, settings, target, experiment['simulationSens'], float(experiment['timeout']), cache)
