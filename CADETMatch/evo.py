import numpy
import CADETMatch.util as util

from pathlib import Path

from deap import creator
from deap import tools

from cadet import Cadet

import CADETMatch.cache as cache

ERROR = {'scores': None,
         'path': None,
         'simulation' : None,
         'error': None,
         'cadetValues':None,
         'cadetValuesKEQ': None}

def fitness(individual, json_path, run_experiment=None):
    if json_path != cache.cache.json_path:
        cache.cache.setup_dir(json_path)
        util.setupLog(cache.cache.settings['resultsDirLog'], "main.log")
        cache.cache.setup(json_path)

    if run_experiment is None:
        run_experiment = runExperiment

    scores = []
    error = 0.0

    results = {}
    for experiment in cache.cache.settings['experiments']:
        result = run_experiment(individual, experiment, cache.cache.settings, cache.cache.target, cache.cache)
        if result is not None:
            results[experiment['name']] = result
            scores.extend(results[experiment['name']]['scores'])
            error += results[experiment['name']]['error']
        else:
            return cache.cache.WORST, [], None, individual

    if numpy.any(numpy.isnan(scores)):
        multiprocessing.get_logger().info("NaN found for %s %s", individual, scores)

    #human scores
    humanScores = numpy.concatenate([util.calcMetaScores(scores, cache.cache), [error,]])

    for result in results.values():
        if result['cadetValuesKEQ']:
            cadetValuesKEQ = result['cadetValuesKEQ']
            break

    #generate csv
    csv_record = []
    csv_record.extend(['EVO', 'NA'])
    csv_record.extend(cadetValuesKEQ)
    csv_record.extend(scores)
    csv_record.extend(humanScores)
      
    return scores, csv_record, results, tuple(individual)

def saveExperiments(save_name_base, settings, target, results):
    return util.saveExperiments(save_name_base, settings, target, results, settings['resultsDirEvo'], '%s_%s_EVO.h5')

def plotExperiments(save_name_base, settings, target, results):
    util.plotExperiments(save_name_base, settings, target, results, settings['resultsDirEvo'], '%s_%s_EVO.png')

def runExperiment(individual, experiment, settings, target, cache):
    if 'simulation' not in experiment:
        templatePath = Path(settings['resultsDirMisc'], "template_%s.h5" % experiment['name'])
        templateSim = Cadet()
        templateSim.filename = templatePath.as_posix()
        templateSim.load()
        experiment['simulation'] = templateSim

    return util.runExperiment(individual, experiment, settings, target, experiment['simulation'], experiment['simulation'].root.timeout, cache)

def run(cache):
    "run the parameter estimation"
    searchMethod = cache.settings.get('searchMethod', 'NSGA3')
    return cache.search[searchMethod].run(cache, tools, creator)
