import CADETMatch.util as util
from pathlib import Path
import scipy.optimize
import numpy
import numpy.linalg
import hashlib
import csv
import time

import CADETMatch.cache as cache
import scoop
from cadet import Cadet

class GradientException(Exception):
    pass

def setupTemplates(cache):
    settings = cache.settings
    target = cache.target

    for experiment in settings['experiments']:
        name = experiment['name']

        simulationGrad = Cadet(experiment['simulation'].root)

        template_path_grad = Path(settings['resultsDirMisc'], "template_%s_grad.h5" % name)
        simulationGrad.filename = template_path_grad.as_posix()

        simulationGrad.root.input.solver.time_integrator.abstol = cache.abstolFactorGrad * cache.target[name]['smallest_peak']
        simulationGrad.root.input.solver.time_integrator.reltol = cache.reltolGrad

        start = time.time()
        util.runExperiment(None, experiment, cache.settings, cache.target, simulationGrad, float(experiment['timeout'])*10, cache)
        elapsed = time.time() - start

        #timeout needs to be stored in the template so all processes have it without calculating it
        simulationGrad.root.timeout = elapsed * 10
        simulationGrad.save()

        scoop.logger.info("grad simulation took %s", elapsed)

        scoop.logger.info('grad %s abstol=%.3g  reltol=%.3g', simulationGrad.filename, 
                          simulationGrad.root.input.solver.time_integrator.abstol,
                          simulationGrad.root.input.solver.time_integrator.reltol)

def search(gradCheck, offspring, cache, writer, csvfile, grad_hof, meta_hof, generation, check_all=False, result_data=None):
    if check_all:
        checkOffspring = offspring
    else:
        checkOffspring = (ind for ind in offspring if util.product_score(ind.fitness.values) > gradCheck)
    checkOffspring = filterOverlapArea(cache, checkOffspring)
    newOffspring = cache.toolbox.map(cache.toolbox.evaluate_grad, checkOffspring)

    temp = []
    csv_lines = []
    meta_csv_lines = []

    for i in newOffspring:
        if i is None:
            #failed.append(1)
            pass
        elif i.success:
            ind = cache.toolbox.individual_guess(i.x)
            fit, csv_line, results = cache.toolbox.evaluate(ind)

            ind.fitness.values = fit

            csv_line[0] = 'GRAD'

            save_name_base = hashlib.md5(str(list(ind)).encode('utf-8', 'ignore')).hexdigest()

            ind_meta = cache.toolbox.individualMeta(ind)
            ind_meta.fitness.values = csv_line[-4:] #util.calcMetaScores(fit, cache)

            util.update_result_data(cache, ind, fit, result_data, results, csv_line[-4:])

            if csv_line:
                csv_lines.append([time.ctime(), save_name_base] + csv_line)
                onFront = util.updateParetoFront(grad_hof, ind, cache)
                if onFront and not cache.metaResultsOnly:
                    util.processResultsGrad(save_name_base, ind, cache, results)

                onFrontMeta = util.updateParetoFront(meta_hof, ind_meta, cache)
                if onFrontMeta:
                    meta_csv_lines.append([time.ctime(), save_name_base] + csv_line)
                    util.processResultsMeta(save_name_base, ind, cache, results)
                    cache.lastProgressGeneration = generation

            #failed.append(0)
            #ind.fitness.values = fit
            temp.append(ind)
    
    if temp:
        avg, bestMin, bestProd = util.averageFitness(temp, cache)
        if 0.9 * bestProd > gradCheck:
            gradCheck = 0.9 * bestProd
        #if len(temp) > 0 or all(failed):
        #    gradCheck = (1-gradCheck)/2.0 + gradCheck

    writer.writerows(csv_lines)
    
    #flush before returning
    csvfile.flush()

    path_meta_csv = cache.settings['resultsDirMeta'] / 'results.csv'
    with path_meta_csv.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerows(meta_csv_lines)

    util.cleanupFront(cache, None, meta_hof, grad_hof)
    util.writeMetaFront(cache, meta_hof, path_meta_csv)

    return gradCheck, temp

def filterOverlapArea(cache, checkOffspring, cutoff=0.05):
    """if there is no overlap between the simulation and the data there is no gradient to follow and these entries need to be skipped
    This only applies if the score is SSE or gradVector is True"""
    if not cache.gradVector or (cache.gradVector and cache.badScore == 0):
        return map(list, checkOffspring)

    temp = cache.toolbox.map(cache.toolbox.evaluate, map(list, checkOffspring))

    temp_offspring = []

    for ind, (fit, csv_line, results) in zip(map(list, checkOffspring), temp):
        temp_area_total = 0.0
        temp_area_overlap = 0.0
        if results is not None:
            for exp in results.values():
                sim_times = exp['sim_time']
                sim_values = exp['sim_value']
                exp_values = exp['exp_value']

                for sim_time, sim_value, exp_value in zip(sim_times, sim_values, exp_values):
                    if len(sim_time) < len(exp_value) and len(exp_value) % len(sim_time) == 0:
                        #dealing with fractionation data, there are multiple data sets of len(sim_time)
                        exp_value = exp_value.reshape(-1, len(sim_time))
                        sim_value = sim_value.reshape(-1, len(sim_time))
                        for exp_row, sim_row in zip(exp_value, sim_value):
                            temp_area_total += numpy.trapz(exp_row, sim_time)
                            temp_area_overlap += numpy.trapz(numpy.min([sim_row, exp_row], 0), sim_time)
                    else: 
                        temp_area_total += numpy.trapz(exp_value, sim_time)
                        temp_area_overlap += numpy.trapz(numpy.min([sim_value, exp_value], 0), sim_time)

            percent = temp_area_overlap / temp_area_total
            if percent > cutoff:
                temp_offspring.append(ind)
            else:
                scoop.logger.info('removed %s for insufficient overlap in gradient descent', ind)
        else:
            scoop.logger.info('removed %s for failure', ind)

    if checkOffspring:
        scoop.logger.info("overlap okay offspring %s", temp_offspring)
    return temp_offspring
    
def gradSearch(x, json_path):
    if json_path != cache.cache.json_path:
        cache.cache.setup(json_path)
    
    try:
        x = numpy.clip(x, cache.cache.MIN_VALUE, cache.cache.MAX_VALUE)
        val = scipy.optimize.least_squares(fitness_sens_grad, x, jac='3-point', method='trf', 
                                           bounds=(cache.cache.MIN_VALUE, cache.cache.MAX_VALUE), 
                                           gtol=None, ftol=None, xtol=1e-8, x_scale="jac")
        
        #scores = fitness_sens(val.x, finished=1)
        return val
    except GradientException:
        #If the gradient fails return None as the point so the optimizer can adapt
        return None

def fitness_sens_grad(individual, finished=0):
    return fitness_sens(individual, finished)

def fitness_sens(individual, finished=1):
    minimize = []
    error = 0.0

    diff = []

    results = {}
    for experiment in cache.cache.settings['experiments']:
        result = runExperimentSens(individual, experiment, cache.cache.settings, cache.cache.target, cache.cache)
        if result is not None:
            results[experiment['name']] = result
            minimize.extend(results[experiment['name']]['minimize'])
            error += results[experiment['name']]['error']
            diff.extend(results[experiment['name']]['diff'])
        else:
            raise GradientException("Gradient caused simulation failure, aborting")
   
    #need to minimize
    if cache.cache.gradVector:
        return numpy.array(diff)
    else:
        #return [-1.0 * i for i in scores]
        return numpy.array(minimize)

def saveExperimentsSens(save_name_base, settings, target, results):
    return util.saveExperiments(save_name_base, settings, target, results, settings['resultsDirGrad'], '%s_%s_GRAD.h5')

def runExperimentSens(individual, experiment, settings, target, cache):
    if 'simulationSens' not in experiment:
        templatePath = Path(settings['resultsDirMisc'], "template_%s_grad.h5" % experiment['name'])
        templateSim = Cadet()
        templateSim.filename = templatePath.as_posix()
        templateSim.load()
        experiment['simulationSens'] = templateSim

    return util.runExperiment(individual, experiment, settings, target, experiment['simulationSens'], experiment['simulationSens'].root.timeout, cache)
