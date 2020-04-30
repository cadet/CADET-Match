import shutil
from cadet import Cadet
import CADETMatch.util as util
from pathlib import Path
import CADETMatch.evo as evo
import scipy.optimize
import numpy
import numpy.linalg
import hashlib
import CADETMatch.score as score
import tempfile
import os
import subprocess
import csv
import time
import multiprocessing
import CADETMatch.cache as cache
import CADETMatch.pareto as pareto

class ConditionException(Exception):
    pass

class GradientException(Exception):
    pass


def setupTemplates(cache):
    settings = cache.settings
    target = cache.target
    parms = target['sensitivities']

    for experiment in settings['experiments']:
        HDF5 = experiment['HDF5']
        name = experiment['name']

        simulationSens = Cadet(experiment['simulation'].root)

        template_path_sens = Path(settings['resultsDirMisc'], "template_%s_sens.h5" % name)
        simulationSens.filename = template_path_sens.as_posix()

        simulationSens.root.input.sensitivity.nsens = len(parms)
        simulationSens.root.input.sensitivity.sens_method = 'ad1'

        sensitivity = simulationSens.root.input.sensitivity

        for idx, parm in enumerate(parms):
            name, unit, comp, bound = parm

            paramSection = 'param_%03d' % idx

            sensitivity[paramSection].sens_unit = [unit,]
            sensitivity[paramSection].sens_name = [name,]
            sensitivity[paramSection].sens_comp = [comp,]
            sensitivity[paramSection].sens_reaction = [-1,]
            sensitivity[paramSection].sens_boundphase = [bound,]
            sensitivity[paramSection].sens_section = [-1,]

        simulationSens.save()
        experiment['simulationSens'] = simulationSens


def search(gradCheck, offspring, cache, writer, csvfile, grad_hof, meta_hof, generation, check_all=False, result_data=None):
    if check_all:
        checkOffspring = offspring
    else:
        checkOffspring = (ind for ind in offspring if util.product_score(ind.fitness.values) > gradCheck)
    checkOffspring = filterOverlapArea(cache, checkOffspring)
    newOffspring = cache.toolbox.map(cache.toolbox.evaluate_grad, map(tuple, checkOffspring))
        
    temp = []
    csv_lines = []
    meta_csv_lines = []

    for i in newOffspring:
        if i is None:
            pass
        elif i.success:
            ind = cache.toolbox.individual_guess(i.x)
            fit, csv_line, results, individual = cache.toolbox.evaluate(ind)

            ind.fitness.values = fit

            csv_line[0] = 'GRAD'

            save_name_base = hashlib.md5(str(list(ind)).encode('utf-8', 'ignore')).hexdigest()

            ind_meta = cache.toolbox.individualMeta(ind)
            ind_meta.fitness.values = csv_line[-4:] #util.calcMetaScores(fit, cache)

            util.update_result_data(cache, ind, fit, result_data, results, csv_line[-4:])

            if csv_line:
                csv_lines.append([time.ctime(), save_name_base] + csv_line)
                onFront = pareto.updateParetoFront(grad_hof, ind, cache)
                if onFront and not cache.metaResultsOnly:
                    util.processResultsGrad(save_name_base, ind, cache, results)

                onFrontMeta = pareto.updateParetoFront(meta_hof, ind_meta, cache)
                if onFrontMeta:
                    meta_csv_lines.append([time.ctime(), save_name_base] + csv_line)
                    util.processResultsMeta(save_name_base, ind, cache, results)
                    cache.lastProgressGeneration = generation

            temp.append(ind)
    
    if temp:
        avg, bestMin, bestProd = util.averageFitness(temp, cache)
        if 0.9 * bestProd > gradCheck:
            gradCheck = 0.9 * bestProd

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

def gradSearch(x, json_path):
    if json_path != cache.cache.json_path:
        cache.cache.setup_dir(json_path)
        util.setupLog(cache.cache.settings['resultsDirLog'], "main.log")
        cache.cache.setup(json_path)

    jac_cache = []

    try:
        multiprocessing.get_logger().info("gradSearch least_squares")

        x = util.convert_individual_grad(x, cache.cache)

        result = scipy.optimize.least_squares(fitness_sens_grad, x, jac=jacobian, method='trf', 
                                               bounds=(cache.cache.MIN_VALUE_GRAD, cache.cache.MAX_VALUE_GRAD), 
                                               gtol=None, ftol=None, xtol=1e-10, x_scale="jac",
                                               kwargs={'jac_cache':jac_cache}, verbose=0)


        multiprocessing.get_logger().info("start: %s  stop: %s distance: %s  sse: %s", x, result.x, numpy.linalg.norm(x - result.x), numpy.sum(result.fun**2))
        #multiprocessing.get_logger().info("result %s", result)
        result.x = util.convert_individual_inverse_grad(result.x, cache.cache)
        return result
    except GradientException:
        #If the gradient fails return None as the point so the optimizer can adapt
        multiprocessing.get_logger().error("Gradient Failure", exc_info=True)
        return None
    except ConditionException:
        multiprocessing.get_logger().error("Condition Failure", exc_info=True)
        return None

def fitness_sens_grad(individual, jac_cache, finished=0):
    individual = util.convert_individual_inverse_grad(individual, cache.cache)
    return fitness_sens(individual, jac_cache, finished)

def fitness_sens(individual, jac_cache, finished=1):
    minimize = []
    error = 0.0

    diff = []

    jac_temp = []

    results = {}
    for experiment in cache.cache.settings['experiments']:
        result = runExperimentSens(individual, experiment, cache.cache.settings, cache.cache.target, cache.cache)
        if result is not None:
            jac = getJacobianSimulation(result['simulation'], experiment)
            jac_temp.append(jac)
            results[experiment['name']] = result
            minimize.extend(result['minimize'])
            error += result['error']

            #need to get the right diff for SSE
            temp_diff = getDiff(result, experiment)
            diff.extend(temp_diff)
        else:
            raise GradientException("Gradient caused simulation failure, aborting")
   
    

    #jac_cache[tuple(individual)] = numpy.concatenate(jac_temp, 0)
    jac_cache.append(numpy.concatenate(jac_temp, 0))

    #cond = numpy.linalg.cond(jac_cache[tuple(individual)])

    cond = numpy.linalg.cond(jac_cache[0])

    #multiprocessing.get_logger().info('condition number %s = %s', individual, cond)

    #need to minimize
    diff = numpy.array(diff)
    sse = numpy.sum(diff**2)
    return diff

def getDiff(result, experiment):
    gradsetup = experiment['gradsetup']
    sim = result['simulation']
    solution = sim.root.output.solution
    times = solution.solution_times
    

    sim_value = []
    exp_value = []
    for grad in gradsetup:
        selected_sim = (times >= grad['start']) & (times <= grad['stop'])
        temp = [solution["unit_%03d" % grad['unit']]["solution_outlet_comp_%03d" % comp] for comp in grad['comps']]
        sim_value.append(numpy.sum(numpy.array(temp), axis=0)[selected_sim])

        data = numpy.loadtxt(grad['csv'], delimiter=',')

        time = data[:, 0]
        value = data[:, 1]
        selected_grad = (time >= grad['start']) & (time <= grad['stop'])
        exp_value.append(value[selected_grad])


    sim_value = numpy.concatenate(sim_value, axis=0)
    exp_value = numpy.concatenate(exp_value, axis=0)
    diff_value = sim_value - exp_value
    return diff_value
    

def jacobian(x, jac_cache):
    return jac_cache.pop()
    #x = util.convert_individual_inverse_grad(x, cache.cache)
    #try:
    #    return jac_cache[tuple(x)]
    #except KeyError:
    #    multiprocessing.get_logger().info('keyerror %s %s', repr(x), [repr(key) for key in jac_cache.keys()])

def saveExperimentsSens(save_name_base, settings, target, results):
    return util.saveExperiments(save_name_base, settings, target, results, settings['resultsDirGrad'], '%s_%s_GRAD.h5')

def runExperimentSens(individual, experiment, settings, target, cache):
    if 'simulationSens' not in experiment:
        templatePath = Path(settings['resultsDirMisc'], "template_%s_sens.h5" % experiment['name'])
        templateSim = Cadet()
        templateSim.filename = templatePath.as_posix()
        templateSim.load()
        experiment['simulationSens'] = templateSim

    return util.runExperiment(individual, experiment, settings, target, experiment['simulationSens'], float(experiment['timeout']), cache)

def getJacobianSimulation(sim, experiment):
    #write out jacobian to jac
    gradsetup = experiment['gradsetup']
    params = len(cache.cache.target['sensitivities'])
    sens = sim.root.output.sensitivity
    times = sim.root.output.solution.solution_times

    jacobians = []
    for grad in gradsetup:
        temp_jac = []
        selected = (times >= grad['start']) & (times <= grad['stop'])
        for idx in range(params):
            temp = [sens["param_%03d" % idx]["unit_%03d" % grad['unit']]["sens_outlet_comp_%03d" % comp] for comp in grad['comps']]
            temp_jac.append(numpy.sum(numpy.array(temp), axis=0))
        jacobian = numpy.squeeze(numpy.array(temp_jac)).T
        jacobian = jacobian[selected,:]
        jacobians.append(jacobian)
    
    jacobian = numpy.concatenate(jacobians, axis=0)
    return jacobian

def filterOverlapArea(cache, checkOffspring, cutoff=0.01):
    """if there is no overlap between the simulation and the data there is no gradient to follow and these entries need to be skipped
    This only applies if the score is SSE or gradVector is True"""
    temp = cache.toolbox.map(cache.toolbox.evaluate, map(list, checkOffspring))

    temp_offspring = []

    lookup = util.create_lookup(checkOffspring)

    for fit, csv_line, results, individual in temp:
        ind = util.pop_lookup(lookup, individual)
        temp_area_total = 0.0
        temp_area_overlap = 0.0
        if results is not None:
            for exp in results.values():
                sim_times = exp['sim_time']
                sim_values = exp['sim_value']
                exp_values = exp['exp_value']

                for sim_time, sim_value, exp_value in zip(sim_times, sim_values, exp_values):
                    temp_area_total += numpy.trapz(exp_value, sim_time)
                    temp_area_overlap += numpy.trapz(numpy.min([sim_value, exp_value], 0), sim_time)

            percent = temp_area_overlap / temp_area_total
            if percent > cutoff:
                temp_offspring.append(ind)
            else:
                multiprocessing.get_logger().info('removed %s for insufficient overlap in gradient descent', ind)
        else:
            multiprocessing.get_logger().info('removed %s for failure', ind)

    if checkOffspring:
        multiprocessing.get_logger().info("overlap okay offspring %s", temp_offspring)
    return temp_offspring
