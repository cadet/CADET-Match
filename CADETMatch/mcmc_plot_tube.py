import numpy
from sklearn import preprocessing
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import cross_val_score
import scipy
import multiprocessing
import sys

import cadet
import CADETMatch.util as util
import CADETMatch.evo as evo
import pandas
from addict import Dict
from CADETMatch.cache import cache
from pathlib import Path
import warnings
import joblib
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import matplotlib
matplotlib.use('Agg')

from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import matplotlib.pyplot as plt

import matplotlib.cm
cm_plot = matplotlib.cm.gist_rainbow

import logging
import CADETMatch.loggerwriter as loggerwriter

import CADETMatch.kde_generator as kde_generator

def get_color(idx, max_colors, cmap):
    return cmap(1.*float(idx)/max_colors)

saltIsotherms = {b'STERIC_MASS_ACTION', b'SELF_ASSOCIATION', b'MULTISTATE_STERIC_MASS_ACTION', 
                 b'SIMPLE_MULTISTATE_STERIC_MASS_ACTION', b'BI_STERIC_MASS_ACTION'}

size = 20

plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title
plt.rc('figure', autolayout=True)

def plotTube(cache, chain, kde, scaler):
    results, combinations = processChainForPlots(cache, chain, kde, scaler)

    output_mcmc = cache.settings['resultsDirSpace'] / "mcmc"
    output_mcmc.mkdir(parents=True, exist_ok=True)

    mcmc_h5 = output_mcmc / "mcmc_plots.h5"

    h5 = cadet.H5()
    h5.filename = mcmc_h5.as_posix()

    for expName,value in combinations.items():
        exp_name = expName.split('_')[0]
        plot_mcmc(output_mcmc, value, expName, "combine", cache.target[exp_name]['time'], cache.target[exp_name]['value'])
        h5.root[expName] = value['data']
        h5.root['exp_%s_time' % expName] = cache.target[exp_name]['time']
        h5.root['exp_%s_value' % expName] = cache.target[exp_name]['value']

    for exp, units in results.items():
        for unitName, unit in units.items():
            for comp, data in unit.items():
                expName = '%s_%s' % (exp, unitName)
                plot_mcmc(output_mcmc, data, expName, comp, cache.target[exp]['time'], cache.target[exp]['value'])
                h5.root['%s_%s' % (expName, comp)] = data['data']
                h5.root['exp_%s_%s_time' % (expName, comp)] = cache.target[exp_name]['time']
                h5.root['exp_%s_%s_value' % (expName, comp)] = cache.target[exp_name]['value']
    h5.save()

def processChainForPlots(cache, chain, kde, scaler):
    mcmc_selected, mcmc_selected_transformed, mcmc_selected_score, results, times, mcmc_score = genRandomChoice(cache, chain, kde, scaler)

    writeSelected(cache, mcmc_selected, mcmc_selected_transformed, mcmc_selected_score, mcmc_score)

    results, combinations = processResultsForPlotting(results, times)

    return results, combinations

def processResultsForPlotting(results, times):
    for expName, units in list(results.items()):
        for unitName, unit in list(units.items()):
            for compIdx, compValue in list(unit.items()):
                data = numpy.array(compValue)
                results[expName][unitName][compIdx] = {}
                results[expName][unitName][compIdx]['data'] = data
                results[expName][unitName][compIdx]["mean"] = numpy.mean(data, 0)
                results[expName][unitName][compIdx]["std"] = numpy.std(data, 0)
                results[expName][unitName][compIdx]["min"] = numpy.min(data, 0)
                results[expName][unitName][compIdx]["max"] = numpy.max(data, 0)
                results[expName][unitName][compIdx]['time'] = times[expName]

    combinations = {}
    for expName, units in results.items():
        for unitName, unit in list(units.items()):
            data = numpy.zeros(unit[0]['data'].shape)
            times = unit[0]['time']
            for compIdx, compValue in list(unit.items()):
                data = data + compValue['data']
            temp = {}
            temp['data'] = data
            temp['time'] = times
            temp["mean"] = numpy.mean(data, 0)
            temp["std"] = numpy.std(data, 0)
            temp["min"] = numpy.min(data, 0)
            temp["max"] = numpy.max(data, 0)
            comb_name = '%s_%s' % (expName, unitName)
            combinations[comb_name] = temp
    return results, combinations

def fitness(individual):
    return evo.fitness(individual, sys.argv[1])

def genRandomChoice(cache, chain, kde, scaler):
    "want about 1000 items and will be removing about 10% of them"
    size = 1100
    chain = chain[~numpy.all(chain == 0, axis=1)]
    if len(chain) > size:
        indexes = numpy.random.choice(chain.shape[0], size, replace=False)
        chain = chain[indexes,:]

    lb, ub = numpy.percentile(chain, [5, 95], 0)
    selected = (chain >= lb) & (chain <= ub)
    bools = numpy.all(selected, 1)
    chain = chain[bools, :]

    individuals = []

    for idx in range(len(chain)):
        individuals.append(chain[idx,:])

    map_function = util.getMapFunction()
    fitnesses = map_function(fitness, individuals)

    results = {}
    times = {}

    mcmc_selected = []
    mcmc_selected_transformed = []
    mcmc_selected_score = []

    for (fit, csv_line, result) in fitnesses:        
        if result is not None:
            mcmc_selected_score.append(tuple(fit))
            for value in result.values():
                mcmc_selected_transformed.append(tuple(value['cadetValues']))
                mcmc_selected.append(tuple(value['individual']))
                break
  
            for key,value in result.items():
                sims = results.get(key, {})

                sim = value['simulation']

                outlets = [(name, sim.root.input.model[name].ncomp) for name in cache.target[key]['units_used']]

                for outlet, ncomp in outlets:
                    units = sims.get(outlet, {})

                    for i in range(ncomp):
                        comps = units.get(i, [])
                        comps.append(sim.root.output.solution[outlet]["solution_outlet_comp_%03d" % i])
                        units[i] = comps

                    sims[outlet] = units

                    if key not in times:
                        times[key] = sim.root.output.solution.solution_times

                results[key] = sims
        else:
            multiprocessing.get_logger().info("Failure in random choice: fit: %s  csv_line: %s   result:%s", fit, csv_line, result)

    mcmc_selected = numpy.array(mcmc_selected)
    mcmc_selected_transformed = numpy.array(mcmc_selected_transformed)
    mcmc_selected_score = numpy.array(mcmc_selected_score)

    mcmc_score = kde.score_samples(scaler.transform(mcmc_selected_score)) + numpy.log(2)

    return mcmc_selected, mcmc_selected_transformed, mcmc_selected_score, results, times, mcmc_score

def writeSelected(cache, mcmc_selected, mcmc_selected_transformed, mcmc_selected_score, mcmc_score):
    mcmc_h5 = Path(cache.settings['resultsDirMCMC']) / "mcmc_selected.h5"
    h5 = cadet.H5()
    h5.filename = mcmc_h5.as_posix()
    h5.root.mcmc_selected = numpy.array(mcmc_selected)
    h5.root.mcmc_selected_transformed = numpy.array(mcmc_selected_transformed)
    h5.root.mcmc_selected_score = numpy.array(mcmc_selected_score)
    h5.root.mcmc_selected_kdescore = numpy.array(mcmc_score)
    h5.save()

def plot_mcmc(output_mcmc, value, expName, name, expTime, expValue):
    data = value['data']
    times = value['time']
    mean = value["mean"]
    std = value["std"]
    minValues = value["min"]
    maxValues = value["max"]

    plt.figure(figsize=[10,10])
    plt.plot(times, mean, label='mean')
    plt.fill_between(times, mean - std, mean + std,
                color='green', alpha=0.2, label='high prob')
    plt.fill_between(times, minValues, maxValues,
                color='red', alpha=0.2, label='low prob')
    plt.plot(expTime, expValue, 'r', label='exp')
    plt.xlabel('time(s)')
    plt.ylabel('conc(mM)')
    plt.legend()
    plt.savefig(str(output_mcmc / ("%s_%s.png" % (expName, name) ) ))
    plt.close()

    row, col = data.shape
    alpha = 0.01
    plt.figure(figsize=[10,10])
    plt.plot(times, data[0,:], 'g', alpha=0.5, label='prob')
    plt.plot(times, data.transpose(), 'g', alpha=alpha)
    plt.plot(times, mean, 'k', label='mean')
    plt.plot(expTime, expValue, 'r', label='exp')
    plt.xlabel('time(s)')
    plt.ylabel('conc(mM)')
    plt.legend()

    plt.savefig(str(output_mcmc / ("%s_%s_lines.png" % (expName, name) ) ))
    plt.close()

def main():
    cache.setup_dir(sys.argv[1])
    util.setupLog(cache.settings['resultsDirLog'], "mcmc_plot_tube.log")
    cache.setup(sys.argv[1])

    kde, kde_scaler = kde_generator.getKDE(cache)

    mcmcDir = Path(cache.settings['resultsDirMCMC'])
    mcmc_h5 = mcmcDir / "mcmc.h5"

    with h5py.File(mcmc_h5, 'r') as h5:
        flat_chain = h5['/flat_chain'][()]
        plotTube(cache, flat_chain, kde, kde_scaler)

if __name__ == "__main__":
    main()

