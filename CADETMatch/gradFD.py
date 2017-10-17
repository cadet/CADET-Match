import shutil
import h5py
import util
from pathlib import Path
import evo
import scipy.optimize
import numpy
import numpy.linalg
import functools
import operator
import hashlib
import score
import tempfile
import os
import subprocess
import csv
import time
import sys
from cadet import Cadet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ConditionException(Exception):
    pass

class GradientException(Exception):
    pass


def setupTemplates(settings, target):
    pass


def search(gradCheck, offspring, toolbox):
    checkOffspring = (ind for ind in offspring if min(ind.fitness.values) > gradCheck)
    newOffspring = toolbox.map(gradSearch, checkOffspring)

    temp = []
    print("Running gradient check")
    failed = []
    for i in newOffspring:
        if i is None:
            failed.append(1)
        elif i.success:
            a = toolbox.individual_guess(i.x)
            fit = toolbox.evaluate(a)
            failed.append(0)
            a.fitness.values = fit
            print(i.x, fit)
            temp.append(a)
    
    if temp:
        avg, bestMin = util.averageFitness(temp)
        if 0.9 * bestMin > gradCheck:
            gradCheck = 0.9 * bestMin
        #if len(temp) > 0 or all(failed):
        #    gradCheck = (1-gradCheck)/2.0 + gradCheck
        print("Finished running on ", len(temp), " individuals new threshold", gradCheck)
    return gradCheck, temp

def gradSearch(x):
    cache = {}
    #x0 = scipy.optimize.least_squares(fitness_sens, x, jac=jacobian, method='trf', bounds=(evo.MIN_VALUE, evo.MAX_VALUE), kwargs={'cache':cache})
    #return scipy.optimize.least_squares(fitness_sens, x, jac=jacobian, method='lm', kwargs={'cache':cache}, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    try:
       return scipy.optimize.least_squares(fitness_sens, x, jac='3-point', method='trf', bounds=(evo.MIN_VALUE, evo.MAX_VALUE))
    except GradientException:
        #If the gradient fails return None as the point so the optimizer can adapt
        print("Gradient Failure")
        print(sys.exc_info()[0])
        return None
    except ConditionException:
        print("Condition Failure")
        print(sys.exc_info()[0])
        return None

def jacobian(x, cache):
    jac = numpy.concatenate(cache[tuple(x)], 1)
    return jac.transpose()

def fitness_sens(individual, cache):
    if not(util.feasible(individual)):
        return [0.0] * evo.numGoals
    print("Gradient Running for ", individual)
    scores = []
    error = 0.0

    results = {}
    diffs = []
    cache[tuple(individual)] = []
    for experiment in evo.settings['experiments']:
        result = runExperimentSens(individual, experiment, evo.settings, evo.target, cache[tuple(individual)])
        if result is not None:
            results[experiment['name']] = result
            scores.extend(results[experiment['name']]['scores'])
            error += results[experiment['name']]['error']
            diffs.append(result['diff'])
        else:
            raise GradientException("Gradient caused simulation failure, aborting")

    #need

    #human scores
    humanScores = numpy.array( [functools.reduce(operator.mul, scores, 1)**(1.0/len(scores)), 
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
        if result['cadetValues']:
            cadetValues = result['cadetValues']
            break

    #generate csv
    path = Path(evo.settings['resultsDirBase'], evo.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow([time.ctime(), save_name_base, 'GRAD', ''] + ["%.5g" % i for i in cadetValues] + ["%.5g" % i for i in scores] + list(humanScores)) 

    notDuplicate = saveExperimentsSens(save_name_base, evo.settings, evo.target, results)
    if notDuplicate:
        plotExperimentsSens(save_name_base, evo.settings, evo.target, results)

    #cleanup
    for result in results.values():
        if result['path']:
            os.remove(result['path'])
    
    #cond = numpy.linalg.cond(jacobian(tuple(individual), cache))
    #if cond > 1000:
    #    raise ConditionException("Condition Number is %s. This location is poorly conditioned. Aborting gradient search" % cond)
    
    #return numpy.concatenate(diffs, 0)
    return -scores

def saveExperimentsSens(save_name_base, settings,target, results):
    for experiment in settings['experiments']:
        experimentName = experiment['name']
        src = results[experimentName]['path']
        dst = Path(settings['resultsDirGrad'], '%s_%s_GRAD.h5' % (save_name_base, experimentName))

        if dst.is_file():  #File already exists don't try to write over it
            return False
        else:
            shutil.copy(src, bytes(dst))

            with h5py.File(dst, 'a') as h5:
                scoreGroup = h5.create_group("score")

                for (header, score) in zip(experiment['headers'], results[experimentName]['scores']):
                    scoreGroup.create_dataset(header, data=numpy.array(score, 'f8'))
    return True

def plotExperimentsSens(save_name_base, settings, target, results):
    for experiment in settings['experiments']:
        experimentName = experiment['name']
        dst = Path(settings['resultsDirGrad'], '%s_%s_GRAD.png' % (save_name_base, experimentName))

        numPlots = len(experiment['features'])

        exp_time = target[experimentName]['time']
        exp_value = target[experimentName]['value']

        sim_time = results[experimentName]['time']
        sim_value = results[experimentName]['value']

        fig = plt.figure(figsize=[10, numPlots*10])

        for idx,feature in enumerate(experiment['features']):
            graph = fig.add_subplot(numPlots, 1, idx+1)

            featureName = feature['name']
            featureType = feature['type']

            selected = target[experimentName][featureName]['selected']

            

            if featureType in ('similarity', 'curve', 'breakthrough', 'dextrane'):
                #sim_spline = scipy.interpolate.UnivariateSpline(sim_time[selected], sim_value[selected], s=1e-6)
                #exp_spline = scipy.interpolate.UnivariateSpline(exp_time[selected], exp_value[selected], s=1e-6)
                sim_spline = scipy.interpolate.UnivariateSpline(sim_time[selected], sim_value[selected], s=util.smoothing_factor(sim_value[selected]))
                exp_spline = scipy.interpolate.UnivariateSpline(exp_time[selected], exp_value[selected], s=util.smoothing_factor(exp_value[selected]))
                graph.plot(sim_time[selected], sim_spline(sim_time[selected]), 'r--', label='Simulation')
                graph.plot(exp_time[selected], exp_spline(exp_time[selected]), 'g:', label='Experiment')
            elif featureType == 'derivative_similarity':
                sim_spline = scipy.interpolate.UnivariateSpline(sim_time[selected], sim_value[selected], s=util.smoothing_factor(sim_value[selected])).derivative(1)
                exp_spline = scipy.interpolate.UnivariateSpline(exp_time[selected], exp_value[selected], s=util.smoothing_factor(exp_value[selected])).derivative(1)

                graph.plot(sim_time[selected], util.smoothing(sim_time[selected], sim_spline(sim_time[selected])), 'r--', label='Simulation')
                graph.plot(exp_time[selected], util.smoothing(exp_time[selected], exp_spline(exp_time[selected])), 'g:', label='Experiment')
            graph.legend()

        plt.savefig(bytes(dst), dpi=100)
        plt.close()

def runExperimentSens(individual, experiment, settings, target, jac):
    handle, path = tempfile.mkstemp(suffix='.h5')
    os.close(handle)

    if 'simulationSens' not in experiment:
        templatePath = Path(settings['resultsDirMisc'], "template_%s_sens.h5" % experiment['name'])
        templateSim = Cadet()
        templateSim.filename = templatePath
        templateSim.load()
        experiment['simulationSens'] = templateSim


    simulation = Cadet(experiment['simulationSens'].root)
    simulation.filename = path

    simulation.root.input.solver.nthreads = 1
    cadetValues = set_simulation(individual, simulation, evo.settings)

    simulation.save()

    def leave():
        os.remove(path)
        return None

    try:
        simulation.run(timeout = experiment['timeout'] * len(target['sensitivities']))
    except subprocess.TimeoutExpired:
        print("Simulation Timed Out")
        return leave()

    #read sim data
    simulation.load()
    try:
        #get the solution times
        times = simulation.root.output.solution.solution_times
    except KeyError:
        #sim must have failed
        util.log(individual, "sim must have failed", path)
        return leave()
    util.log("Everything ran fine")

    gradient_components = experiment['gradient']['components']
    gradient_CSV = experiment['gradient']['CSV']
    gradient_stop = experiment['gradient']['stop']
    gradient_start = experiment['gradient']['start']

    if len(gradiegradient_componentsnt_CSV) > 1 and len(gradient_CSV) == 1:
        combine_components = True
    else:
        combine_components = False

    selected = (times >= gradient_start) & (times <= gradient_stop)

    #write out jacobian to jac
    temp = []
    sens = simulation.root.output.sensitivity
    for idx, parm in enumerate( target['sensitivities']):
        name, unit, comp, bound = parm
        #-1 means comp independent but the entry is still stored in comp 0
        if comp == -1:
            comp = 0

        temp.append([sens["param_%03d" % idx]["unit_%03d" % unit]["sens_column_outlet_comp_%03d" % comp]])

    temp = transform(temp, target, settings, cadetValues)
    jacobian = numpy.array(temp)
    jac.append(numpy.array(temp))

    temp = {}
    temp['time'] = times
    if isinstance(experiment['isotherm'], list):
        temp['value'] = numpy.sum([numpy.array(h5[i]) for i in experiment['isotherm']],0)
    else:
        temp['value'] = numpy.array(h5[experiment['isotherm']])
    temp['path'] = path
    temp['scores'] = []
    temp['error'] = 0.0
    temp['cond'] = numpy.linalg.cond(jacobian, None)
    temp['cadetValues'] = cadetValues
    temp['simulation'] = simulation

    for feature in experiment['features']:
        start = feature['start']
        stop = feature['stop']
        featureType = feature['type']
        featureName = feature['name']

        if featureType in ('similarity', 'similarityDecay'):
            scores, sse = score.scoreSimilarity(temp, target[experiment['name']], target[experiment['name']][featureName])
        elif featureType in ('similarityCross', 'similarityCrossDecay'):
            scores, sse = score.scoreSimilarityCrossCorrelate(temp, target[experiment['name']], target[experiment['name']][featureName])
        elif featureType == 'derivative_similarity':
            scores, sse = score.scoreDerivativeSimilarity(temp, target[experiment['name']], target[experiment['name']][featureName])
        elif featureType == 'curve':
            scores, sse = score.scoreCurve(temp, target[experiment['name']], target[experiment['name']][featureName])
        elif featureType == 'breakthrough':
            scores, sse = score.scoreBreakthrough(temp, target[experiment['name']], target[experiment['name']][featureName])
        elif featureType == 'breakthroughCross':
            scores, sse = score.scoreBreakthroughCross(temp, target[experiment['name']], target[experiment['name']][featureName])
        elif featureType == 'dextrane':
            scores, sse = score.scoreDextrane(temp, target[experiment['name']], target[experiment['name']][featureName])
        temp['scores'].extend(scores)
        temp['error'] += sse

    return temp

def transform(tempJac, target, settings, cadetValues):
    jac = []

    # (name, unit, comp, bound)
    idx = 0
    for parameter in settings['parameters']:
        transform = parameter['transform']

        if transform == 'keq':
            for bound in parameter['bound']:
                jac.append(tempJac[idx+1] * cadetValues[idx+1] + cadetValues[idx] * tempJac[idx])   
                jac.append(-tempJac[idx+1] * cadetValues[idx+1])   
                idx += 2

        elif transform == "log":
            for bound in parameter['bound']:
                jac.append(tempJac[idx] * cadetValues[idx])
                idx += 1
    return jac
