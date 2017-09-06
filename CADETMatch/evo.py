import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
import util
import score

import functools
import subprocess
import json
import csv
import h5py
import operator
import hashlib
import time
import sys
import pickle

import scipy.interpolate
import math
import array
import random

from pathlib import Path

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import os
import tempfile
import shutil

import spea2
from cadet import Cadet

#parallelization
from scoop import futures

import scipy.signal

ERROR = {'scores': None,
         'path': None,
         'simulation' : None,
         'error': None,
         'cadetValues':None}

def fitness(individual):
    if not(util.feasible(individual)):
        return [0.0] * numGoals

    scores = []
    error = 0.0

    results = {}
    for experiment in settings['experiments']:
        result = runExperiment(individual, experiment, settings, target)
        if result is not None:
            results[experiment['name']] = result
            scores.extend(results[experiment['name']]['scores'])
            error += results[experiment['name']]['error']
        else:
            return [0.0] * numGoals

    #need

    #human scores
    humanScores = numpy.array( [functools.reduce(operator.mul, scores, 1)**(1.0/len(scores)), 
                                min(scores), sum(scores)/len(scores), 
                                numpy.linalg.norm(scores)/numpy.sqrt(len(scores)), 
                                -error] )

    #best
    target['bestHumanScores'] = numpy.max(numpy.vstack([target['bestHumanScores'], humanScores]), 0)

    #save
    keepTop = settings['keepTop']

    keep_result = 0
    if any(humanScores >= (keepTop * target['bestHumanScores'])):
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
    path = Path(settings['resultsDirBase'], settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow([time.ctime(), save_name_base, 'EVO', 'NA'] + 
                        ["%.5g" % i for i in cadetValues] + 
                        ["%.5g" % i for i in scores] + 
                        list(humanScores)) 

    #print('keep_result', keep_result)
    if keep_result:
        notDuplicate = saveExperiments(save_name_base, settings, target, results)
        #print('notDuplicate', notDuplicate)
        if notDuplicate:
            plotExperiments(save_name_base, settings, target, results)

    #cleanup
    for result in results.values():
        if result['path']:
            os.remove(result['path'])
            
    return scores

def saveExperiments(save_name_base, settings,target, results):
    for experiment in settings['experiments']:
        experimentName = experiment['name']

        dst = Path(settings['resultsDirEvo'], '%s_%s_EVO.h5' % (save_name_base, experimentName))

        if dst.is_file():  #File already exists don't try to write over it
            return False
        else:
            simulation = results[experimentName]['simulation']
            simulation.filename = bytes(dst)

            for (header, score) in zip(experiment['headers'], results[experimentName]['scores']):
                simulation.root.score[header] = score
            simulation.save()
    return True

def plotExperiments(save_name_base, settings, target, results):
    for experiment in settings['experiments']:
        experimentName = experiment['name']
        
        dst = Path(settings['resultsDirEvo'], '%s_%s_EVO.png' % (save_name_base, experimentName))

        numPlots = len(experiment['features'])

        exp_time = target[experimentName]['time']
        exp_value = target[experimentName]['value']

        fig = plt.figure(figsize=[10, numPlots*10])

        for idx,feature in enumerate(experiment['features']):
            graph = fig.add_subplot(numPlots, 1, idx+1)
            
            featureName = feature['name']
            featureType = feature['type']

            feat = target[experimentName][featureName]

            selected = feat['selected']
            exp_time = feat['time'][selected]
            exp_value = feat['value'][selected]

            sim_time, sim_value = util.get_times_values(results[experimentName]['simulation'],target[experimentName][featureName])

            if featureType in ('similarity', 'curve', 'breakthrough', 'dextrane', 'similarityCross', 'breakthroughCross'):
                #sim_spline = scipy.interpolate.UnivariateSpline(sim_time[selected], sim_value[selected], s=1e-6)
                #exp_spline = scipy.interpolate.UnivariateSpline(exp_time[selected], exp_value[selected], s=1e-6)

                #sim_spline = scipy.interpolate.UnivariateSpline(sim_time, sim_value, s=util.smoothing_factor(sim_value))
                #exp_spline = scipy.interpolate.UnivariateSpline(exp_time, exp_value, s=util.smoothing_factor(exp_value))
                #graph.plot(sim_time[selected], sim_spline(sim_time[selected]), 'r--', label='Simulation')
                #graph.plot(exp_time[selected], exp_spline(exp_time[selected]), 'g:', label='Experiment')

                graph.plot(sim_time, sim_value, 'r--', label='Simulation')
                graph.plot(exp_time, exp_value, 'g:', label='Experiment')
            elif featureType == 'derivative_similarity':
                sim_spline = scipy.interpolate.UnivariateSpline(sim_time, sim_value, s=util.smoothing_factor(sim_value)).derivative(1)
                exp_spline = scipy.interpolate.UnivariateSpline(exp_time, exp_value, s=util.smoothing_factor(exp_value)).derivative(1)

                graph.plot(sim_time, util.smoothing(sim_time, sim_spline(sim_time)), 'r--', label='Simulation')
                graph.plot(exp_time, util.smoothing(exp_time, exp_spline(exp_time)), 'g:', label='Experiment')
            graph.legend()

        plt.savefig(bytes(dst), dpi=100)
        plt.close()

def set_simulation(individual, simulation, settings):
    util.log("individual", individual)

    cadetValues = []

    idx = 0
    for parameter in settings['parameters']:
        location = parameter['location']
        transform = parameter['transform']
        comp = parameter['component']

        if transform == 'keq':
            unit = location[0].split('/')[3]
        elif transform == 'log':
            unit = location.split('/')[3]

        NBOUND = simulation.root.input.model[unit].discretization.nbound
        boundOffset = numpy.cumsum(numpy.concatenate([[0,], NBOUND]))

        if transform == 'keq':
            for bound in parameter['bound']:
                position = boundOffset[comp] + bound
                simulation[location[0].lower()][position] = math.exp(individual[idx])
                simulation[location[1].lower()][position] = math.exp(individual[idx])/(math.exp(individual[idx+1]))

                cadetValues.append(simulation[location[0]][position])
                cadetValues.append(simulation[location[1]][position])

                idx += 2

        elif transform == "log":
            for bound in parameter['bound']:
                if comp == -1:
                    position = ()
                    simulation[location.lower()] = math.exp(individual[idx])
                    cadetValues.append(simulation[location])
                else:
                    position = boundOffset[comp] + bound
                    simulation[location.lower()][position] = math.exp(individual[idx])
                    cadetValues.append(simulation[location][position])
                idx += 1
    util.log("finished setting hdf5")
    return cadetValues

def runExperiment(individual, experiment, settings, target):
    handle, path = tempfile.mkstemp(suffix='.h5')
    os.close(handle)

    if 'simulation' not in experiment:
        templatePath = Path(settings['resultsDirMisc'], "template_%s.h5" % experiment['name'])
        templateSim = Cadet()
        templateSim.filename = templatePath
        templateSim.load()
        experiment['simulation'] = templateSim

    simulation = Cadet(experiment['simulation'].root)
    simulation.filename = path

    simulation.root.input.solver.nthreads = 1
    cadetValues = set_simulation(individual, simulation, settings)

    simulation.save()

    def leave():
        os.remove(path)
        return None

    try:
        simulation.run(timeout = experiment['timeout'])
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


    temp = {}
    temp['simulation'] = simulation
    temp['path'] = path
    temp['scores'] = []
    temp['error'] = 0.0
    temp['cadetValues'] = cadetValues

    for feature in experiment['features']:
        start = feature['start']
        stop = feature['stop']
        featureType = feature['type']
        featureName = feature['name']

        if featureType == 'similarity':
            scores, sse = score.scoreSimilarity(temp, target[experiment['name']], target[experiment['name']][featureName])
        elif featureType == 'similarityCross':
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

def setup(settings_filename):
    "setup the parameter estimation"
    with open(settings_filename) as json_data:
        settings = json.load(json_data)
        Cadet.cadet_path = settings['CADETPath']
        headers, numGoals = genHeaders(settings)
        target = createTarget(settings)
        MIN_VALUE, MAX_VALUE = buildMinMax(settings)
        toolbox = setupDEAP(numGoals, settings, target, MIN_VALUE, MAX_VALUE)

        #create used paths in settings, only the root process will make the directories later
        settings['resultsDirEvo'] = Path(settings['resultsDir']) / "evo"
        settings['resultsDirGrad'] = Path(settings['resultsDir']) / "grad"
        settings['resultsDirMisc'] = Path(settings['resultsDir']) / "misc"
        settings['resultsDirBase'] = Path(settings['resultsDir'])


    return settings, headers, numGoals, target, MIN_VALUE, MAX_VALUE, toolbox

def createDirectories(settings):
    settings['resultsDirBase'].mkdir(parents=True, exist_ok=True)
    settings['resultsDirGrad'].mkdir(parents=True, exist_ok=True)
    settings['resultsDirMisc'].mkdir(parents=True, exist_ok=True)
    settings['resultsDirEvo'].mkdir(parents=True, exist_ok=True)

def setupDEAP(numGoals, settings, target, MIN_VALUE, MAX_VALUE):
    "setup the DEAP variables"
    searchMethod = settings.get('searchMethod', 'SPEA2')
    toolbox = base.Toolbox()
    if searchMethod == 'SPEA2':
        return spea2.setupDEAP(numGoals, settings, target, MIN_VALUE, MAX_VALUE, fitness, futures.map, creator, toolbox, base, tools)

def buildMinMax(settings):
    "build the minimum and maximum parameter boundaries"
    MIN_VALUE = []
    MAX_VALUE = []

    for parameter in settings['parameters']:
        transform = parameter['transform']
        location = parameter['location']

        if transform == 'keq':
            minKA = parameter['minKA']
            maxKA = parameter['maxKA']
            minKEQ = parameter['minKEQ']
            maxKEQ = parameter['maxKEQ']

            minValues = [item for pair in zip(minKA, minKEQ) for item in pair]
            maxValues = [item for pair in zip(maxKA, maxKEQ) for item in pair]

            minValues = numpy.log(minValues)
            maxValues = numpy.log(maxValues)

        elif transform == 'log':
            minValues = numpy.log(parameter['min'])
            maxValues = numpy.log(parameter['max'])

        MIN_VALUE.extend(minValues)
        MAX_VALUE.extend(maxValues)
    return MIN_VALUE, MAX_VALUE

def genHeaders(settings):
    headers = ['Time','Name', 'Method','Condition Number',]

    numGoals = 0

    for parameter in settings['parameters']:
        comp = parameter['component']
        if parameter['transform'] == 'keq':
            location = parameter['location']
            nameKA = location[0].rsplit('/',1)[-1]
            nameKD = location[1].rsplit('/',1)[-1]
            for bound in parameter['bound']:
                headers.append("%s Comp:%s Bound:%s" % (nameKA, comp, bound))
                headers.append("%s Comp:%s Bound:%s" % (nameKD, comp, bound))
        elif parameter['transform'] == 'log':
            location = parameter['location']
            name = location.rsplit('/',1)[-1]
            for bound in parameter['bound']:
                headers.append("%s Comp:%s Bound:%s" % (name, comp, bound))

    for idx,experiment in enumerate(settings['experiments']):
        experimentName = experiment['name']
        experiment['headers'] = []
        for feature in experiment['features']:
            if feature['type'] in ('similarity', 'similarityCross'):
                name = "%s_%s" % (experimentName, feature['name'])
                temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name]
                numGoals += 3

            elif feature['type'] == 'derivative_similarity':
                name = "%s_%s" % (experimentName, feature['name'])
                temp = ["%s_Derivative_Similarity" % name, "%s_High_Value" % name, "%s_High_Time" % name, "%s_Low_Value" % name, "%s_Low_Time" % name]
                numGoals += 5

            elif feature['type'] == 'curve':
                name = "%s_%s" % (experimentName, feature['name'])
                temp  = ["%s_Similarity" % name]
                numGoals += 1

            elif feature['type'] == 'breakthrough':
                name = "%s_%s" % (experimentName, feature['name'])
                temp  = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time_Start" % name, "%s_Time_Stop" % name]
                numGoals += 4
            elif feature['type'] == 'breakthroughCross':
                name = "%s_%s" % (experimentName, feature['name'])
                temp  = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name]
                numGoals += 3
            elif feature['type'] == 'dextrane':
                #name = "%s_%s" % (experimentName, feature['name'])
                #temp = ["%s_Front_Similarity" % name, "%s_High_Value" % name, "%s_High_Time" % name, "%s_Low_Value" % name, "%s_Low_Time" % name]
                #numGoals += 5
                name = "%s_%s" % (experimentName, feature['name'])
                temp = ["%s_Front_Similarity" % name, "%s_Derivative_Similarity" % name, "%s_Time" % name]
                numGoals += 3

            headers.extend(temp)
            experiment['headers'].extend(temp)

    headers.extend(['Product Root Score', 'Min Score', 'Mean Score', 'Norm', 'SSE'])
    return headers, numGoals

def createTarget(settings):
    target = {}

    for experiment in settings['experiments']:
        target[experiment["name"]] = createExperiment(experiment)
    target['bestHumanScores'] = numpy.zeros(5)

    #SSE are negative so they sort correctly with better scores being less negative
    target['bestHumanScores'][4] = -1e308;  

    #setup sensitivities
    parms = []
    for parameter in settings['parameters']:
        comp = parameter['component']
        transform = parameter['transform']

        if transform == 'keq':
            location = parameter['location']
            nameKA = location[0].rsplit('/',1)[-1]
            nameKD = location[1].rsplit('/',1)[-1]
            unit = int(location[0].split('/')[3].replace('unit_', ''))

            for bound in parameter['bound']:
                parms.append((nameKA, unit, comp, bound))
                parms.append((nameKD, unit, comp, bound))

        elif transform == 'log':
            location = parameter['location']
            name = location.rsplit('/',1)[-1]
            unit = int(location.split('/')[3].replace('unit_', ''))
            for bound in parameter['bound']:
                parms.append((name, unit, comp, bound))

    target['sensitivities'] = parms


    return target

def createExperiment(experiment):
    temp = {}

    HDF5 = Path(experiment['HDF5'])

    with h5py.File(HDF5, 'r') as h5:
        #CV needs to be based on superficial velocity not interstitial velocity
        length = h5['/input/model/unit_001/COL_LENGTH'].value

        try:
            velocity = h5['/input/model/unit_001/VELOCITY'].value
        except KeyError:
            velocity = 1

        try:
            area = h5['/input/model/unit_001/CROSS_SECTION_AREA'].value
        except KeyError:
            area = 1

        colPorosity = h5['/input/model/unit_001/COL_POROSITY'].value
        colLength = h5['/input/model/unit_001/COL_LENGTH'].value

        conn = h5['/input/model/connections/switch_000/CONNECTIONS'].value
        conn = numpy.array(conn)
        conn = numpy.reshape(conn, [-1, 5])

        #find all the entries that connect to the column
        filter = conn[:,1] == 1

        #flow is the sum of all flow rates that connect to this column which is in the last column
        flow = sum(conn[filter, -1])

        if area == 1 and abs(velocity) != 1:
            CV_time = h5['/input/model/unit_001/COL_LENGTH'].value / h5['/input/model/unit_001/VELOCITY'].value
        else:
            CV_time = (area * colLength) / flow

        #print("CV_time", CV_time)

    if 'CSV' in experiment:
        data = numpy.genfromtxt(experiment['CSV'], delimiter=',')

        temp['time'] = data[:,0]
        temp['value'] = data[:,1]

    for feature in experiment['features']:
        featureName = feature['name']
        featureType = feature['type']
        featureStart = feature['start']
        featureStop = feature['stop']

        temp[featureName] = {}

        if 'CSV' in feature:
            dataLocal = numpy.genfromtxt(feature['CSV'], delimiter=',')
            temp[featureName]['time'] = dataLocal[:,0]
            temp[featureName]['value'] = dataLocal[:,1]
        else:
            temp[featureName]['time'] = data[:,0]
            temp[featureName]['value'] = data[:,1]

        if 'isotherm' in feature:
            temp[featureName]['isotherm'] = feature['isotherm']
        else:
            temp[featureName]['isotherm'] = experiment['isotherm']

        temp[featureName]['selected'] = (temp[featureName]['time'] >= featureStart) & (temp[featureName]['time'] <= featureStop)
            
        selectedTimes = temp[featureName]['time'][temp[featureName]['selected']]
        selectedValues = temp[featureName]['value'][temp[featureName]['selected']]

        if featureType in ('similarity', 'similarityCross'):
            temp[featureName]['peak'] = util.find_peak(selectedTimes, selectedValues)[0]
            temp[featureName]['time_function'] = score.time_function(CV_time, temp[featureName]['peak'][0], diff_input = True if featureType == 'similarityCross' else False)
            temp[featureName]['value_function'] = score.value_function(temp[featureName]['peak'][1])

        if featureType == 'breakthrough':
            temp[featureName]['break'] = util.find_breakthrough(selectedTimes, selectedValues)
            temp[featureName]['time_function_start'] = score.time_function(CV_time, temp[featureName]['break'][0][0])
            temp[featureName]['time_function_stop'] = score.time_function(CV_time, temp[featureName]['break'][1][0])
            temp[featureName]['value_function'] = score.value_function(temp[featureName]['break'][0][1])

        if featureType == 'breakthroughCross':
            temp[featureName]['break'] = util.find_breakthrough(selectedTimes, selectedValues)
            temp[featureName]['time_function'] = score.time_function(CV_time, temp[featureName]['break'][0][0], diff_input=True)
            temp[featureName]['value_function'] = score.value_function(temp[featureName]['break'][0][1])

        if featureType == 'derivative_similarity':
            exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, selectedValues, s=util.smoothing_factor(selectedValues)).derivative(1)

            [high, low] = util.find_peak(selectedTimes, util.smoothing(selectedTimes, exp_spline(selectedTimes)))

            temp[featureName]['peak_high'] = high
            temp[featureName]['peak_low'] = low

            temp[featureName]['time_function_high'] = score.time_function(CV_time, high[0])
            temp[featureName]['value_function_high'] = score.value_function(high[1])
            temp[featureName]['time_function_low'] = score.time_function(CV_time, low[0])
            temp[featureName]['value_function_low'] = score.value_function(low[1])

        if featureType == "dextrane":
            #change the stop point to be where the max positive slope is along the searched interval
            exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, selectedValues, s=util.smoothing_factor(selectedValues), k=1).derivative(1)
            values = exp_spline(selectedTimes)
            #print([i for i in zip(selectedTimes, values)])
            max_index = numpy.argmax(values)
            max_time = selectedTimes[max_index]
            #print(max_time, values[max_index])
            
            temp[featureName]['origSelected'] = temp[featureName]['selected']
            temp[featureName]['selected'] = temp[featureName]['selected'] & (temp[featureName]['time'] <= max_time)
            temp[featureName]['max_time'] = max_time
            temp[featureName]['maxTimeFunction'] = score.time_function_decay(CV_time/10.0, max_time)
            
    return temp

def createCSV(settings, headers):
    path = Path(settings['resultsDirBase'], settings['CSV'])
    if not path.exists():
        with path.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            writer.writerow(headers)


def run(settings, toolbox):
    "run the parameter estimation"
    searchMethod = settings.get('searchMethod', 'SPEA2')
    if searchMethod == 'SPEA2':
        spea2.run(settings, toolbox, tools, creator)

def setupTemplates(settings, target):
    "setup all the experimental templates"
    for experiment in settings['experiments']:
        HDF5 = experiment['HDF5']
        name = experiment['name']

        template_path = Path(settings['resultsDirMisc'], "template_%s.h5" % name)

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

        template.root.input.solver.user_solution_times = target[name]['time']
        template.root.input.solver.section_times[-1] = target[name]['time'][-1]
        template.root.input['return'].unit_001.write_solution_particle = 0
        template.root.input['return'].unit_001.write_solution_column_inlet = 1
        template.root.input['return'].unit_001.write_solution_column_outlet = 1
        template.root.input['return'].unit_001.split_components_data = 0
        template.root.solver.nthreads = 1
        template.root.solver.time_integrator.init_step_size = 0
        template.root.solver.time_integrator.max_steps = 0

        template.save()

        experiment['simulation'] = template

#This will run when the module is imported so that each process has its own copy of this data
settings, headers, numGoals, target, MIN_VALUE, MAX_VALUE, toolbox = setup(sys.argv[1])