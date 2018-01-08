
import sys
import evo
#import grad
#import gradFD
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

from cache import cache

#due to how scoop works and the need to break things up into multiple processes it is hard to use class based systems
#As a result most of the code is broken up into modules but is still based on pure functions

def main():
    setup(cache, sys.argv[1])
    #grad.setupTemplates(evo.settings, evo.target)
    hof = evo.run(cache)

    print("hall of fame")
    for i in hof:
        print(i, type(i), i.fitness.values)

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
                print(json_path)

                setup(cache, json_path)



                #call setup on all processes with the new json file as an argument to reset them
                #util.updateScores(json_path)

                hof = evo.run(cache)
                temp.append(util.bestMinScore(hof))

                numpy_temp = numpy.array(temp)
                cov = numpy.cov(numpy_temp.transpose())
                print("in progress cov", cov, "data", numpy_temp, "det", numpy.linalg.det(cov))
                print("in progress cov", cov, "data", numpy_temp, "det", numpy.linalg.det(cov), file=bootstrap.open('w'), flush=True)

            numpy_temp = numpy.array(temp)
            cov = numpy.cov(numpy_temp.transpose())
            print("final cov", cov, "data", numpy_temp, "det", numpy.linalg.det(cov))
            print("final cov", cov, "data", numpy_temp, "det", numpy.linalg.det(cov), file=bootstrap.open('w'), flush=True)

def setup(cache, json_path):
    "run seutp for the current json_file"
    cache.setup(json_path)

    createDirectories(cache, json_path)
    createCSV(cache)
    setupTemplates(cache)
    setupDeap(cache)

def createDirectories(cache, json_path):
    cache.settings['resultsDirBase'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirGrad'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirMisc'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirSpace'].mkdir(parents=True, exist_ok=True)
    cache.settings['resultsDirEvo'].mkdir(parents=True, exist_ok=True)

    #copy simulation setting file to result base directory
    try:
        shutil.copy(str(json_path), str(cache.settings['resultsDirBase']))
    except shutil.SameFileError:
        pass

def createCSV(cache):
    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    if not path.exists():
        with path.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            writer.writerow(cache.headers)

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

        #change to where we want the template created
        template.filename = template_path

        try:
            del template.root.input.solver.user_solution_times
        except KeyError:
            pass

        try:
            del template.root.output
        except KeyError:
            pass

        template.root.input.solver.user_solution_times = cache.target[name]['time']
        template.root.input.solver.sections.section_times[-1] = cache.target[name]['time'][-1]
        template.root.input['return'].unit_001.write_solution_particle = 0
        template.root.input['return'].unit_001.write_solution_column_inlet = 1
        template.root.input['return'].unit_001.write_solution_column_outlet = 1
        template.root.input['return'].unit_001.split_components_data = 0
        template.root.input.solver.nthreads = 1
        template.root.input.solver.time_integrator.init_step_size = 0
        template.root.input.solver.time_integrator.max_steps = 0

        template.save()

        experiment['simulation'] = template

def setupDeap(cache):
    "setup the DEAP variables"
    searchMethod = cache.settings.get('searchMethod', 'SPEA2')
    cache.toolbox = base.Toolbox()
    cache.search[searchMethod].setupDEAP(cache, evo.fitness, futures.map, creator, base, tools)

if __name__ == "__main__":
    start = time.time()
    main()
    print("System has finished")
    print("The total runtime was %s seconds" % (time.time() - start))
