import sys

import CADETMatch.evo as evo
import CADETMatch.gradFD as gradFD
import CADETMatch.util as util
import CADETMatch.version as version
import time
import numpy
import shutil
import csv
from cadet import Cadet, H5
from pathlib import Path

from deap import creator
from deap import tools
from deap import base

#parallelization
import multiprocessing

import logging

from CADETMatch.cache import cache
import CADETMatch.loggerwriter as loggerwriter
import h5py

def main(map_function):
    path = sys.argv[1]
    setup(cache, path, map_function)
    gradFD.setupTemplates(cache)
    #grad.setupTemplates(cache)
    hof = evo.run(cache)

    multiprocessing.get_logger().info("altFeatures %s", cache.altFeatures)
    multiprocessing.get_logger().info("altFeatureNames %s", cache.altFeatureNames)

    if cache.altFeatures:
        for name in cache.altFeatureNames:
            json_path = util.setupAltFeature(cache, name)
            setup(cache, json_path, map_function)
            hof = evo.run(cache)

    continue_mcmc(cache, map_function)

    if "repeat" in cache.settings:
        repeat = int(cache.settings['repeat'])

        for i in range(repeat):
            json_path = util.repeatSimulation(i)
            multiprocessing.get_logger().info(json_path)

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
                multiprocessing.get_logger().info(json_path)

                setup(cache, json_path)



                #call setup on all processes with the new json file as an argument to reset them
                #util.updateScores(json_path)

                hof = evo.run(cache)
                temp.append(util.bestMinScore(hof))

                numpy_temp = numpy.array(temp)
                cov = numpy.cov(numpy_temp.transpose())
                multiprocessing.get_logger().info("in progress cov %s data %s det %s", cov, numpy_temp, numpy.linalg.det(cov))

            numpy_temp = numpy.array(temp)
            cov = numpy.cov(numpy_temp.transpose())
            multiprocessing.get_logger().info("final cov %s data %s det %s", cov, numpy_temp, numpy.linalg.det(cov))

def setup(cache, json_path, map_function):
    "run seutp for the current json_file"
    cache.setup_dir(json_path)    
    
    createDirectories(cache, json_path)
    util.setupLog(cache.settings['resultsDirLog'], "main.log")

    print_version()
    
    cache.setup(json_path)
    
    createCSV(cache)
    createProgressCSV(cache)
    createErrorCSV(cache)
    setupTemplates(cache)
    setupDeap(cache, map_function)    

def print_version():
    multiprocessing.get_logger().info('CADETMatch starting up version: %s', version.__version__)

    import importlib_metadata

    modules = [('joblib', '0.15.1'), ('addict', '2.2.1'),
               ('corner', '2.0.1'), ('emcee', '3.0.2'),
               ('SALib', '0.0+nnone'), ('deap', '1.3.1'),
               ('psutil', '5.7.0'), ('numpy', '1.18.5'),
               ('openpyxl', '3.0.3'), ('scipy', '1.5.0'),
               ('matplotlib', '3.2.1'), ('pandas', '1.0.5'),
               ('h5py', '2.10.0'), ('cadet', '0.4'),
               ('seaborn', '0.10.1'), ('scikit-learn', '0.23.1')]

    for module,version_tested in modules:
        multiprocessing.get_logger().info('%s version: %s tested with %s', module, importlib_metadata.version(module), version_tested)

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
    path = Path(cache.settings['resultsDirBase'], cache.settings['csv'])
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
        if index >= 0:
            simulation[path][index] = value
        else:
            simulation[path] = value

def setTemplateValuesAuto(simulation, set_values_auto, cache):
    if 'mcmc_h5' not in cache.settings:
        multiprocessing.get_logger().error("set_values_auto can't be used without mcmc_h5 as a prior")
    
    mcmc_h5 = Path(cache.settings['mcmc_h5'])
    mle_h5 = mcmc_h5.parent / "mle.h5"


    data = H5()
    data.filename = mle_h5.as_posix()
    data.load()
    stat_MLE = data.root.stat_MLE

    used = set()

    for path, index, mle_index in set_values_auto:
        value = stat_MLE[mle_index]
        if mle_index not in used:
            used.add(mle_index)
        if index >= 0:
            simulation[path][index] = value
        else:
            simulation[path] = value

    if len(used) != len(stat_MLE):
        multiprocessing.get_logger().warn("not all values from the prior where used, proceed with caution")
    

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
        template.root.experiment_name = name

        if 'set_values' in experiment:
            setTemplateValues(template, experiment['set_values'])

        if 'set_values_auto' in experiment:
            setTemplateValuesAuto(template, experiment['set_values_auto'], cache)

        util.setupSimulation(template, cache.target[name]['time'], cache.target[name]['smallest_peak'], cache)

        start = time.time()
        util.runExperiment(None, experiment, cache.settings, cache.target, template, 
                           experiment.get('timeout', 1800), cache)
        elapsed = time.time() - start

        multiprocessing.get_logger().info("simulation took %s", elapsed)
        
        #timeout needs to be stored in the template so all processes have it without calculating it
        template.root.timeout = max(10, elapsed * 10)
        
        if cache.settings['searchMethod'] != 'MCMC' and "kde_synthetic" in cache.settings:
            #the base case needs to be saved since the normal template file is what the rest of the code will look for
            template_base_path = Path(cache.settings['resultsDirMisc'], "template_%s_base.h5" % name)
            template.filename = template_base_path.as_posix()
            template.save()
            
            multiprocessing.get_logger().info("create bias template for experiment %s", name)
            template_bias = util.biasSimulation(template, experiment, cache)
            template_bias_path = Path(cache.settings['resultsDirMisc'], "template_%s_bias.h5" % name)
            template_bias.filename = template_bias_path.as_posix()
            template_bias.save()
            template = template_bias

        #change to where we want the template created
        template.filename = template_path.as_posix()
        
        template.save()

        experiment['simulation'] = template

        template_path = Path(cache.settings['resultsDirMisc'], "template_%s_final.h5" % name)
        template_final = Cadet(template.root.copy())
        template_final.filename =  template_path.as_posix()
        if cache.dynamicTolerance:
            template_final.root.input.solver.time_integrator.abstol = cache.abstolFactorGrad
            template_final.root.input.solver.time_integrator.reltol = 0.0

        start = time.time()
        util.runExperiment(None, experiment, cache.settings, cache.target, template_final, 
                           experiment.get('timeout', 1800), cache)
        elapsed = time.time() - start

        multiprocessing.get_logger().info("simulation final took %s", elapsed)
        
        #timeout needs to be stored in the template so all processes have it without calculating it
        template_final.root.timeout = max(10, elapsed * 10)

        template_final.save()
        experiment['simulation_final'] = template_final

def setupDeap(cache, map_function):
    "setup the DEAP variables"
    searchMethod = cache.settings.get('searchMethod', 'NSGA3')
    cache.toolbox = base.Toolbox()
    cache.search[searchMethod].setupDEAP(cache, evo.fitness, evo.fitness_final, gradFD.gradSearch, gradFD.search, gradFD.gradSearchFine, map_function, creator, base, tools)

def continue_mcmc(cache, map_function):
    if cache.continueMCMC:
        json_path = util.setupMCMC(cache)
        multiprocessing.get_logger().info(json_path)

        setup(cache, json_path, map_function)

        hof = evo.run(cache)

if __name__ == "__main__":
    start = time.time()
    map_function = util.getMapFunction()
    main(map_function=map_function)
    multiprocessing.get_logger().info('System has finished')
    multiprocessing.get_logger().info("The total runtime was %s seconds" % (time.time() - start))
    sys.exit()