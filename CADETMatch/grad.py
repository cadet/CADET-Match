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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def setupTemplates(settings, target):
    parms = target['sensitivities']

    for experiment in settings['experiments']:
        HDF5 = experiment['HDF5']
        name = experiment['name']

        template_path = Path(settings['resultsDirMisc'], "template_%s.h5" % name)
        template_path_sens = Path(settings['resultsDirMisc'], "template_%s_sens.h5" % name)

        shutil.copy(bytes(template_path),  bytes(template_path_sens))

        with h5py.File(template_path_sens, 'a') as h5:
            h5['/input/sensitivity/NSENS'][:] = len(parms)

            sensitivity = h5['/input/sensitivity']

            for idx, parm in enumerate(parms):
                name, unit, comp, bound = parm
                sens = sensitivity.create_group('param_%03d' % idx)
        
                util.set_value(sens, 'SENS_UNIT', 'i4', [unit,])
    
                util.set_value_enum(sens, 'SENS_NAME', [name,])

                util.set_value(sens, 'SENS_COMP', 'i4', [comp,])
                util.set_value(sens, 'SENS_REACTION', 'i4', [-1,])
                util.set_value(sens, 'SENS_BOUNDPHASE', 'i4', [bound,])
                util.set_value(sens, 'SENS_SECTION', 'i4', [-1,])

                util.set_value(sens, 'SENS_ABSTOL', 'f8', [1e-6,])
                util.set_value(sens, 'SENS_FACTOR', 'f8', [1.0,])

def search(gradCheck, offspring, toolbox):
    checkOffspring = (ind for ind in offspring if min(ind.fitness.values) > gradCheck)
    newOffspring = toolbox.map(gradSearch, checkOffspring)

    temp = []
    print("Running gradient check")
    for i in newOffspring:
        if i.success:
            a = toolbox.individual_guess(i.x)
            fit = toolbox.evaluate(a)
            a.fitness.values = fit
            print(i.x, fit)
            temp.append(a)
    
    if len(temp) > 0:
        gradCheck = (1-gradCheck)/2.0 + gradCheck
    print("Finished running on ", len(temp), " individuals new threshold", gradCheck)
    return gradCheck, temp

def gradSearch(x):
    cache = {}
    #x0 = scipy.optimize.least_squares(fitness_sens, x, jac=jacobian, method='trf', bounds=(evo.MIN_VALUE, evo.MAX_VALUE), kwargs={'cache':cache})
    #return scipy.optimize.least_squares(fitness_sens, x, jac=jacobian, method='lm', kwargs={'cache':cache}, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return scipy.optimize.least_squares(fitness_sens, x, jac=jacobian, method='trf', bounds=(evo.MIN_VALUE, evo.MAX_VALUE), kwargs={'cache':cache}, x_scale='jac')

def jacobian(x, cache):
    jac = numpy.concatenate(cache[tuple(x)], 1)
    return jac.transpose()

def fitness_sens(individual, cache):
    if not(util.feasible(individual)):
        return [0.0] * numGoals

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
            return [0.0] * numGoals

    #need

    #human scores
    humanScores = numpy.array( [functools.reduce(operator.mul, scores, 1)**(1.0/len(scores)), min(scores), sum(scores)/len(scores), numpy.linalg.norm(scores)/numpy.sqrt(len(scores)), -error] )

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
        writer.writerow([time.ctime(), save_name_base, 'GRAD', numpy.linalg.cond(jacobian(tuple(individual), cache))] + ["%.5g" % i for i in cadetValues] + ["%.5g" % i for i in scores] + list(humanScores)) 

    notDuplicate = saveExperimentsSens(save_name_base, evo.settings, evo.target, results)
    if notDuplicate:
        plotExperimentsSens(save_name_base, evo.settings, evo.target, results)

    #cleanup
    for result in results.values():
        if result['path']:
            os.remove(result['path'])
            
    return numpy.concatenate(diffs, 0)

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

            

            if featureType in ('similarity', 'curve', 'breakthrough'):
                sim_spline = scipy.interpolate.UnivariateSpline(sim_time[selected], sim_value[selected], s=1e-6)
                exp_spline = scipy.interpolate.UnivariateSpline(exp_time[selected], exp_value[selected], s=1e-6)
                graph.plot(sim_time[selected], sim_spline(sim_time[selected]), 'r--', label='Simulation')
                graph.plot(exp_time[selected], exp_spline(exp_time[selected]), 'g:', label='Experiment')
            elif featureType == 'derivative_similarity':
                sim_spline = scipy.interpolate.UnivariateSpline(sim_time[selected], sim_value[selected], s=1e-5).derivative(1)
                exp_spline = scipy.interpolate.UnivariateSpline(exp_time[selected], exp_value[selected], s=1e-5).derivative(1)

                graph.plot(sim_time[selected], util.smoothing(sim_time[selected], sim_spline(sim_time[selected])), 'r--', label='Simulation')
                graph.plot(exp_time[selected], util.smoothing(exp_time[selected], exp_spline(exp_time[selected])), 'g:', label='Experiment')
            graph.legend()

        plt.savefig(bytes(dst), dpi=100)
        plt.close()

def runExperimentSens(individual, experiment, settings, target, jac):
    template = Path(settings['resultsDirMisc'], "template_%s_sens.h5" % experiment['name'])

    handle, path = tempfile.mkstemp(suffix='.h5')
    os.close(handle)
    util.log(template, path)
    shutil.copy(bytes(template), path)
    
    #change file
    with h5py.File(path, 'a') as h5:
        cadetValues = evo.set_h5(individual, h5, evo.settings)
        h5['/input/solver/NTHREADS'][:] = 1

    def leave():
        os.remove(path)
        return None

    try:
        subprocess.run([settings['CADETPath'], path], timeout = experiment['timeout'] * len(target['sensitivities']))
    except subprocess.TimeoutExpired:
        return leave()

    #read sim data
    with h5py.File(path, 'r') as h5:
        try:
            #get the solution times
            times = numpy.array(h5['/output/solution/SOLUTION_TIMES'].value)
        except KeyError:
            #sim must have failed
            util.log(individual, "sim must have failed", path)
            return leave()
    util.log("Everything ran fine")

    #write out jacobian to jac
    with h5py.File(path, 'r') as h5:
        temp = []
        for idx, parm in enumerate( target['sensitivities']):
            name, unit, comp, bound = parm
            temp.append(h5['/output/sensitivity/param_%03d/unit_%03d/SENS_COLUMN_OUTLET_COMP_%03d' % (idx, unit, comp)][:])
        temp = transform(temp, target, settings, cadetValues)
        jac.append(numpy.array(temp))

        temp = {}
        temp['time'] = times
        temp['value'] = numpy.array(h5[experiment['isotherm']])
        temp['path'] = path
        temp['scores'] = []
        temp['error'] = sum((temp['value']-target[experiment['name']]['value'])**2)
        temp['diff'] = temp['value']-target[experiment['name']]['value']
        temp['cond'] = numpy.linalg.cond(jac)
        temp['cadetValues'] = cadetValues


    for feature in experiment['features']:
        start = feature['start']
        stop = feature['stop']
        featureType = feature['type']
        featureName = feature['name']

        if featureType == 'similarity':
            temp['scores'].extend(score.scoreSimilarity(temp, target[experiment['name']], target[experiment['name']][featureName]))
        elif featureType == 'derivative_similarity':
            temp['scores'].extend(score.scoreDerivativeSimilarity(temp, target[experiment['name']], target[experiment['name']][featureName]))
        elif featureType == 'curve':
            temp['scores'].extend(score.scoreCurve(temp, target[experiment['name']], target[experiment['name']][featureName]))
        elif featureType == 'breakthrough':
            temp['scores'].extend(score.scoreBreakthrough(temp, target[experiment['name']], target[experiment['name']][featureName]))

    return temp

def transform(tempJac, target, settings, cadetValues):
    jac = []

    # (name, unit, comp, bound)
    idx = 0
    for parameter in settings['parameters']:
        transform = parameter['transform']

        if transform == 'keq':
            for bound in parameter['bound']:
                jac.append(tempJac[idx+1] * cadetValues[idx+1] + cadetValues[idx]/cadetValues[idx+1] * tempJac[idx])   
                jac.append(-tempJac[idx+1] * cadetValues[idx+1])   
                idx += 2

        elif transform == "log":
            for bound in parameter['bound']:
                jac.append(tempJac[idx] * cadetValues[idx])
                idx += 1
    return jac