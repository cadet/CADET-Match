import sys

import evo
import grad
import gradFD
import util
import time
import numpy
import shutil
import csv
from cadet import Cadet
from pathlib import Path

from deap import creator
from deap import tools
from deap import base

#parallelization
from scoop import futures
import scoop

import logging

from cache import cache
import loggerwriter
import h5py

#due to how scoop works and the need to break things up into multiple processes it is hard to use class based systems
#As a result most of the code is broken up into modules but is still based on pure functions

#setup scoop logging for all processes

def main():
    setup(cache, sys.argv[1])
    grad.setupTemplates(cache)
    hof = evo.run(cache)

    continue_mcmc(cache)

    if "repeat" in cache.settings:
        repeat = int(cache.settings['repeat'])

        for i in range(repeat):
            json_path = util.repeatSimulation(i)
            scoop.logger.info(json_path)

            setup(cache, json_path)

            hof = evo.run(cache)
        
        util.metaCSV(cache)

    if "bootstrap" in cache.settings:
        temp = []

        samples = int(cache.settings['bootstrap']['samples'])

        if samples:

            center = float(cache.settings['bootstrap']['center'])
            noise = float(cache.settings['bootstrap']['percentNoise'])/100.0

            bootstrap = cache.settings['resultsDirBase'] / "bootstrap_output"



            for i in range(samples):
                #copy csv files to a new directory with noise added
                #put a new json file in the directory that points to the new csv files
                json_path = util.copyCSVWithNoise(i, center, noise)
                scoop.logger.info(json_path)

                setup(cache, json_path)



                #call setup on all processes with the new json file as an argument to reset them
                #util.updateScores(json_path)

                hof = evo.run(cache)
                temp.append(util.bestMinScore(hof))

                numpy_temp = numpy.array(temp)
                cov = numpy.cov(numpy_temp.transpose())
                scoop.logger.info("in progress cov %s data %s det %s", cov, numpy_temp, numpy.linalg.det(cov))

            numpy_temp = numpy.array(temp)
            cov = numpy.cov(numpy_temp.transpose())
            scoop.logger.info("final cov %s data %s det %s", cov, numpy_temp, numpy.linalg.det(cov))

def setup(cache, json_path):
    "run seutp for the current json_file"
    cache.setup(json_path)

    createDirectories(cache, json_path)
    createCSV(cache)
    createProgressCSV(cache)
    createErrorCSV(cache)
    setupTemplates(cache)
    setupDeap(cache)
    setupLog(cache.settings['resultsDirLog'])

def setupLog(log_directory):
    logger = scoop.logger
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_directory / "main.log")
    fh.setLevel(logging.INFO)

    # add the handlers to the logger
    logger.addHandler(fh)

    sys.stdout = loggerwriter.LoggerWriter(logger.info)
    sys.stderr = loggerwriter.LoggerWriter(logger.warning)

def createDirectories(cache, json_path):
    cache.settings['resultsDirBase'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirGrad'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirMisc'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirSpace'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirEvo'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirProgress'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirMeta'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirLog'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirMCMC'].mkdir(parents=True, exist_ok=True)

    #copy simulation setting file to result base directory
    try:
        shutil.copy(str(json_path), str(cache.settings['resultsDirBase']))
    except shutil.SameFileError:
        pass

def createCSV(cache):
    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    if not path.exists():
        with path.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow(cache.headers)

    path = cache.settings['resultsDirMeta'] / 'results.csv'
    if not path.exists():
        with path.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow(cache.headers)

def createProgressCSV(cache):
    path = Path(cache.settings['resultsDirBase'], "progress.csv")
    cache.progress_path = path
    if not path.exists():
        with path.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow(cache.progress_headers)

def createErrorCSV(cache):
    path = Path(cache.settings['resultsDirBase'], "error.csv")
    cache.error_path = path
    if not path.exists():
        with path.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow(cache.parameter_headers + ['Return Code', 'STDOUT', 'STDERR'])

def setTemplateValues(simulation, set_values):
    for path, index, value in set_values:
        if index > 0:
            simulation[path][index] = value
        else:
            simulation[path] = value

def setupTemplates(cache):
    "setup all the experimental templates"
    for experiment in cache.settings['experiments']:
        HDF5 = experiment['HDF5']
        name = experiment['name']

        template_path = Path(cache.settings['resultsDirMisc'], "template_%s.h5" % name)

        template = Cadet()

        #load based on where the HDF5 file is
        template.filename = HDF5
        template.load()

        if 'set_values' in experiment:
            setTemplateValues(template, experiment['set_values'])

        util.setupSimulation(template, cache.target[name]['time'])

        if cache.settings['searchMethod'] != 'MCMC' and "kde_synthetic" in cache.settings:
            #the base case needs to be saved since the normal template file is what the rest of the code will look for
            template_base_path = Path(cache.settings['resultsDirMisc'], "template_%s_base.h5" % name)
            template.filename = template_base_path.as_posix()
            template.save()
            
            scoop.logger.info("create bias template for experiment %s", name)
            template_bias = util.biasSimulation(template, experiment, cache)
            template_bias_path = Path(cache.settings['resultsDirMisc'], "template_%s_bias.h5" % name)
            template_bias.filename = template_bias_path.as_posix()
            template_bias.save()
            template = template_bias

        #change to where we want the template created
        template.filename = template_path.as_posix()
        template.save()

        experiment['simulation'] = template

def setupDeap(cache):
    "setup the DEAP variables"
    searchMethod = cache.settings.get('searchMethod', 'SPEA2')
    cache.toolbox = base.Toolbox()

    if getattr(scoop, 'SIZE', 1) == 1:
        map_function = map
    else:
        map_function = futures.map

    #cache.search[searchMethod].setupDEAP(cache, evo.fitness, gradFD.gradSearch, gradFD.search, map_function, creator, base, tools)
    cache.search[searchMethod].setupDEAP(cache, evo.fitness, grad.gradSearch, grad.search, map_function, creator, base, tools)

def find_percentile(cache):
    "find the percentile boundaries for the input variables"
    resultDir = Path(cache.settings['resultsDir'])
    result_h5 = resultDir / "result.h5"

    with h5py.File(result_h5, 'r') as hf:
        data = hf["input"][:]
        score = hf["output_meta"][:]

        #product min average norm
        best_norm = numpy.max(score[:,3])

        data = data[score[:,3] > 0.9 * best_norm,:]

        lb, ub = numpy.percentile(data, [0, 100], 0)

        lb_trans = util.convert_individual(lb, cache)[1]
        ub_trans = util.convert_individual(ub, cache)[1]

        scoop.logger.info('lb %s  ub %s', lb, ub)
        scoop.logger.info('lb_trans %s  ub_trans %s', lb_trans, ub_trans)

        return lb_trans, ub_trans

def continue_mcmc(cache):
    if cache.continueMCMC:
        lb, ub = find_percentile(cache)

        json_path = util.setupMCMC(cache, lb, ub)
        scoop.logger.info(json_path)

        setup(cache, json_path)

        hof = evo.run(cache)

if __name__ == "__main__":
    start = time.time()
    main()
    scoop.logger.info('Sysem has finished')
    scoop.logger.info("The total runtime was %s seconds" % (time.time() - start))

