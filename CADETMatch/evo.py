import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
import pandas
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
from deap import creator
from deap import tools

import os
import shutil

import spea2
import nsga2
import nsga3
from cadet import Cadet

#parallelization
from scoop import futures

import scipy.signal

from cache import cache

ERROR = {'scores': None,
         'path': None,
         'simulation' : None,
         'error': None,
         'cadetValues':None,
         'cadetValuesKEQ': None}

def fitness(individual, json_path):
    if json_path != cache.json_path:
        cache.setup(json_path)
    
    if not(util.feasible(individual)):
        return cache.WORST

    

    scores = []
    error = 0.0

    results = {}
    for experiment in cache.settings['experiments']:
        result = runExperiment(individual, experiment, cache.settings, cache.target)
        if result is not None:
            results[experiment['name']] = result
            scores.extend(results[experiment['name']]['scores'])
            error += results[experiment['name']]['error']
        else:
            return cache.WORST

    #need

    #human scores
    humanScores = numpy.array( [functools.reduce(operator.mul, scores, 1)**(1.0/len(scores)), 
                                min(scores), sum(scores)/len(scores), 
                                numpy.linalg.norm(scores)/numpy.sqrt(len(scores)), 
                                -error] )

    #best
    cache.target['bestHumanScores'] = numpy.max(numpy.vstack([cache.target['bestHumanScores'], humanScores]), 0)

    #save
    keepTop = cache.settings['keepTop']

    keep_result = 0
    if any(humanScores >= (keepTop * cache.target['bestHumanScores'])):
        keep_result = 1
        
    #flip sign of SSE for writing out to file
    humanScores[-1] = -1 * humanScores[-1]

    #generate save name
    save_name_base = hashlib.md5(str(individual).encode('utf-8','ignore')).hexdigest()

    for result in results.values():
        if result['cadetValuesKEQ']:
            cadetValuesKEQ = result['cadetValuesKEQ']
            break

    #generate csv
    path = Path(cache.settings['resultsDirBase'], cache.settings['CSV'])
    with path.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerow([time.ctime(), save_name_base, 'EVO', 'NA'] + 
                        ["%.5g" % i for i in cadetValuesKEQ] + 
                        ["%.5g" % i for i in scores] + 
                        list(humanScores)) 

    #print('keep_result', keep_result)
    if keep_result:
        notDuplicate = saveExperiments(save_name_base, cache.settings, cache.target, results)
        #print('notDuplicate', notDuplicate)
        if notDuplicate:
            plotExperiments(save_name_base, cache.settings, cache.target, results)

    #cleanup
    for result in results.values():
        if result['path']:
            os.remove(result['path'])
            
    return scores

def saveExperiments(save_name_base, settings,target, results):
    return util.saveExperiments(save_name_base, settings,target, results, settings['resultsDirEvo'], '%s_%s_EVO.h5')

def plotExperiments(save_name_base, settings, target, results):
    util.plotExperiments(save_name_base, settings, target, results, settings['resultsDirEvo'], '%s_%s_EVO.png')

def runExperiment(individual, experiment, settings, target):
    if 'simulation' not in experiment:
        templatePath = Path(settings['resultsDirMisc'], "template_%s.h5" % experiment['name'])
        templateSim = Cadet()
        templateSim.filename = templatePath
        templateSim.load()
        experiment['simulation'] = templateSim

    return util.runExperiment(individual, experiment, settings, target, experiment['simulation'], float(experiment['timeout']))

def createExperiment(experiment):
    temp = {}

    sim = Cadet()
    sim.filename = Path(experiment['HDF5'])
    sim.load()

    abstol = sim.root.input.solver.time_integrator.abstol

    #CV needs to be based on superficial velocity not interstitial velocity
    length = float(sim.root.input.model.unit_001.col_length)

    velocity = sim.root.input.model.unit_001.velocity
    if velocity == {}:
        velocity = 1.0
    velocity = float(velocity)

    area = sim.root.input.model.uni_001.cross_section_area
    if area == {}:
        area = 1.0
    area = float(area)

    porosity = sim.root.input.model.unit_001.col_porosity
    if porosity == {}:
        porosity = sim.root.input.model.unit_001.total_porosity
    if porosity == {}:
        porosity = 1.0
    porosity = float(porosity)

    conn = sim.root.input.model.connections.switch_000.connections

    conn = numpy.array(conn)
    conn = numpy.reshape(conn, [-1, 5])

    #find all the entries that connect to the column
    filter = conn[:,1] == 1

    #flow is the sum of all flow rates that connect to this column which is in the last column
    flow = sum(conn[filter, -1])

    if area == 1 and abs(velocity) != 1:
        CV_time = length / velocity
    else:
        CV_time = (area * length) / flow

    if 'CSV' in experiment:
        data = numpy.genfromtxt(experiment['CSV'], delimiter=',')

        temp['time'] = data[:,0]
        temp['value'] = data[:,1]

    for feature in experiment['features']:
        featureName = feature['name']
        featureType = feature['type']
        featureStart = float(feature['start'])
        featureStop = float(feature['stop'])

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

        if featureType in ('similarity', 'similarityCross', 'similarityHybrid'):
            temp[featureName]['peak'] = util.find_peak(selectedTimes, selectedValues)[0]
            temp[featureName]['time_function'] = score.time_function(CV_time, temp[featureName]['peak'][0], diff_input = True if featureType in ('similarityCross', 'similarityHybrid') else False)
            temp[featureName]['value_function'] = score.value_function(temp[featureName]['peak'][1], abstol)

        if featureType in ('similarityDecay', 'similarityCrossDecay', 'similarityHybridDecay'):
            temp[featureName]['peak'] = util.find_peak(selectedTimes, selectedValues)[0]
            temp[featureName]['time_function'] = score.time_function_decay(CV_time, temp[featureName]['peak'][0], diff_input = True if featureType in ('similarityCrossDecay', 'similarityHybridDecay') else False)
            temp[featureName]['value_function'] = score.value_function(temp[featureName]['peak'][1], abstol)

        if featureType == 'breakthrough':
            temp[featureName]['break'] = util.find_breakthrough(selectedTimes, selectedValues)
            temp[featureName]['time_function_start'] = score.time_function(CV_time, temp[featureName]['break'][0][0])
            temp[featureName]['time_function_stop'] = score.time_function(CV_time, temp[featureName]['break'][1][0])
            temp[featureName]['value_function'] = score.value_function(temp[featureName]['break'][0][1], abstol)

        if featureType == 'breakthroughCross':
            temp[featureName]['break'] = util.find_breakthrough(selectedTimes, selectedValues)
            temp[featureName]['time_function'] = score.time_function(CV_time, temp[featureName]['break'][0][0], diff_input=True)
            temp[featureName]['value_function'] = score.value_function(temp[featureName]['break'][0][1], abstol)

        if featureType == 'derivative_similarity':
            exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, util.smoothing(selectedTimes, selectedValues), s=util.smoothing_factor(selectedValues)).derivative(1)

            [high, low] = util.find_peak(selectedTimes, exp_spline(selectedTimes))

            temp[featureName]['peak_high'] = high
            temp[featureName]['peak_low'] = low

            temp[featureName]['time_function_high'] = score.time_function(CV_time, high[0])
            temp[featureName]['value_function_high'] = score.value_function(high[1], abstol, 0.1)
            temp[featureName]['time_function_low'] = score.time_function(CV_time, low[0])
            temp[featureName]['value_function_low'] = score.value_function(low[1], abstol, 0.1)

        if featureType in ('derivative_similarity_hybrid', 'derivative_similarity_cross'):
            exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, util.smoothing(selectedTimes, selectedValues), s=util.smoothing_factor(selectedValues)).derivative(1)

            [high, low] = util.find_peak(selectedTimes, exp_spline(selectedTimes))

            temp[featureName]['peak_high'] = high
            temp[featureName]['peak_low'] = low

            temp[featureName]['time_function'] = score.time_function(CV_time,high[0], diff_input = True)
            temp[featureName]['value_function_high'] = score.value_function(high[1], abstol, 0.1)
            temp[featureName]['value_function_low'] = score.value_function(low[1], abstol, 0.1)

        if featureType == 'derivative_similarity_cross_alt':
            exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, util.smoothing(selectedTimes, selectedValues), s=util.smoothing_factor(selectedValues)).derivative(1)

            [high, low] = util.find_peak(selectedTimes, exp_spline(selectedTimes))

            temp[featureName]['peak_high'] = high
            temp[featureName]['peak_low'] = low

            temp[featureName]['time_function'] = score.time_function(CV_time,high[0], diff_input = True)

        if featureType == "dextran":
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
            temp[featureName]['maxTimeFunction'] = score.time_function_decay(CV_time/10.0, max_time, diff_input=True)

        if featureType == "dextranHybrid":
            #change the stop point to be where the max positive slope is along the searched interval
            exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, selectedValues, s=util.smoothing_factor(selectedValues), k=1).derivative(1)
            values = exp_spline(selectedTimes)
            max_index = numpy.argmax(values)
            max_time = selectedTimes[max_index]
            
            temp[featureName]['origSelected'] = temp[featureName]['selected']
            temp[featureName]['selected'] = temp[featureName]['selected'] & (temp[featureName]['time'] <= max_time)
            temp[featureName]['max_time'] = max_time
            temp[featureName]['offsetTimeFunction'] = score.time_function_decay(CV_time/10.0, max_time, diff_input=True)

        if featureType == 'fractionation':
            data = pandas.read_csv(feature['csv'])
            rows, cols = data.shape

            flow = sim.root.input.model.connections.switch_000.connections[9]
            smallestTime = min(data['Stop'] - data['Start'])
            abstolFraction = flow * abstol * smallestTime

            print('abstolFraction', abstolFraction)

            headers = data.columns.values.tolist()

            funcs = []

            for sample in range(rows):
                for component in headers[2:]:
                    start = data['Start'][sample]
                    stop = data['Stop'][sample]
                    value = data[component][sample]
                    func = score.value_function(value, abstolFraction)

                    funcs.append( (start, stop, int(component), value, func) )
            temp[featureName]['funcs'] = funcs

        if featureType == 'fractionationCombine':
            data = pandas.read_csv(feature['csv'])
            rows, cols = data.shape

            headers = data.columns.values.tolist()

            flow = sim.root.input.model.connections.switch_000.connections[9]
            smallestTime = min(data['Stop'] - data['Start'])
            abstolFraction = flow * abstol * smallestTime

            print('abstolFraction', abstolFraction)

            funcs = []

            for sample in range(rows):
                for component in headers[2:]:
                    start = data['Start'][sample]
                    stop = data['Stop'][sample]
                    value = data[component][sample]
                    func = score.value_function(value, abstolFraction)

                    funcs.append( (start, stop, int(component), value, func) )
            temp[featureName]['funcs'] = funcs
            temp[featureName]['components'] = [int(i) for i in headers[2:]]
            temp[featureName]['samplesPerComponent'] = rows
            
    return temp

def run(cache):
    "run the parameter estimation"
    searchMethod = cache.settings.get('searchMethod', 'SPEA2')
    if searchMethod == 'SPEA2':
        return spea2.run(cache, tools, creator)
    if searchMethod == 'NSGA2':
        return nsga2.run(cache, tools, creator)
    if searchMethod == 'NSGA3':
        return nsga3.run(cache, tools, creator)
