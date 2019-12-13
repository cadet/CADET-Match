#This module simulates noise (systemic and random) to use for kernel density estimation based on experimental data

#Currently the randomness is

#Pump flow rate
#Pump delay
#Base noise
#Signal noise

import itertools
import CADETMatch.score as score
from cadet import Cadet, H5
from pathlib import Path
import numpy
import multiprocessing
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import copy

bw_tol = 1e-4

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import scipy.optimize

import CADETMatch.util as util
import CADETMatch.cache as cache

import CADETMatch.synthetic_error as synthetic_error

import joblib
import subprocess
import sys

def bandwidth_score(bw, data, store):
    bandwidth = 10**bw[0]
    kde_bw = KernelDensity(kernel='gaussian', bandwidth=bandwidth, atol=bw_tol)
    scores = cross_val_score(kde_bw, data, cv=3)
    store.append( [bandwidth, -max(scores)] )
    return -max(scores)

def get_bandwidth(scores, cache):
    store = []

    result = scipy.optimize.differential_evolution(bandwidth_score, bounds = [(-4, 1),], 
                    args = (scores,store,), updating='deferred', workers=cache.toolbox.map, disp=True,
                    popsize=100)
    bandwidth = 10**result.x[0]
    multiprocessing.get_logger().info("selected bandwidth %s", bandwidth)

    store = numpy.array(store)
    return bandwidth, store

def mirror(data):
    data_min = numpy.max(data,0) - data
    data_mask = numpy.ma.masked_equal(data_min, 0.0, copy=False)
    min_value = data_mask.min(axis=0)
    
    data_mirror = 1 + 1 - numpy.copy(data) + min_value
    full_data = numpy.vstack([data_mirror, data])
    
    return full_data

def setupKDE(cache):
    scores, simulations = generate_synthetic_error(cache)

    mcmcDir = Path(cache.settings['resultsDirMCMC'])

    scores_mirror = mirror(scores)

    scaler = getScaler(scores_mirror)

    scores_scaler = scaler.transform(scores_mirror)

    bandwidth, store = get_bandwidth(scores_scaler, cache)

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, atol=bw_tol).fit(scores_scaler)

    joblib.dump(scaler, mcmcDir / 'kde_scaler.joblib')

    joblib.dump(kde, mcmcDir / 'kde_score.joblib')

    h5_data = H5()
    h5_data.filename = (mcmcDir / "kde_settings.h5").as_posix()
    h5_data.root.bandwidth = bandwidth
    h5_data.root.store = numpy.array(store)
    h5_data.root.scores = scores
    h5_data.root.scores_mirror = scores_mirror
    h5_data.root.scores_mirror_scaled = scores_scaler
    h5_data.save()

    cwd = str(Path(__file__).parent)
    ret = subprocess.run([sys.executable, 'graph_kde.py', str(cache.json_path), str(util.getCoreCounts())], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    util.log_subprocess('graph_kde.py', ret)

    return kde, scaler

def getScaler(data):
    scaler = preprocessing.StandardScaler().fit(data)
    return scaler

def getKDE(cache):
    mcmcDir = Path(cache.settings['resultsDirMCMC'])

    kde = joblib.load(mcmcDir / 'kde_score.joblib')

    scaler = joblib.load(mcmcDir / 'kde_scaler.joblib')

    return kde, scaler

def getKDEPrevious(cache):
    if 'mcmc_h5' in cache.settings:
        mcmc_h5 = Path(cache.settings['mcmc_h5'])
        mcmcDir = mcmc_h5.parent

        if mcmcDir.exists():
            kde = joblib.load(mcmcDir / 'kde_prior.joblib')

            scaler = joblib.load(mcmcDir / 'kde_prior_scaler.joblib')

            return kde, scaler
    return None, None

def generate_data(cache):
    scores, simulations = generate_synthetic_error(cache)

    mcmcDir = Path(cache.settings['resultsDirMCMC'])
    save_scores = mcmcDir / 'scores_used.npy'

    numpy.save(save_scores, scores)

    scores_mirror = mirror(scores)

    scaler = getScaler(scores_mirror)

    scores_scaler = scaler.transform(scores_mirror)

    bandwidth, store = get_bandwidth(scores_scaler, cache)

    return scores, bandwidth

def synthetic_error_simulation(json_path):
    if json_path != cache.cache.json_path:
        cache.cache.setup_dir(json_path)
        util.setupLog(cache.cache.settings['resultsDirLog'], "main.log")
        cache.cache.setup(json_path)

    scores = []
    outputs = {}
    simulations = {}
    experiment_failed = False

    for experiment in cache.cache.settings['kde_synthetic']:
        file_path = experiment['file_path']
        delay_settings = experiment['delay']
        flow_settings = experiment['flow']
        load_settings = experiment['load']
        count_settings = experiment['count']
        experimental_csv = experiment['experimental_csv']
        uv_noise = experiment['uv_noise']
        units = experiment['units']
        name = experiment['name']

        dir_base = Path(cache.cache.settings.get('baseDir'))
        file = dir_base / file_path

        data = numpy.loadtxt(experimental_csv, delimiter=',')
        times = data[:,0]

        temp = Cadet()
        temp.filename = file.as_posix()
        temp.load()

        util.setupSimulation(temp, times, cache.cache.target[name]['smallest_peak'], cache.cache)

        nsec = temp.root.input.solver.sections.nsec

        def post_function(simulation):
            #baseline drift need to redo this
            #error_slope = numpy.random.normal(error_slope_settings[0], error_slope_settings[1], 1)[0]

            for unit in units:
                unit_name = 'unit_%03d' % unit
                for comp in range(simulation.root.input.model[unit_name].ncomp):
                    comp_name = 'solution_outlet_comp_%03d' % comp
                    error_uv = numpy.random.normal(uv_noise[0], uv_noise[1], len(simulation.root.output.solution[unit_name][comp_name]))
                    simulation.root.output.solution[unit_name][comp_name] = simulation.root.output.solution[unit_name][comp_name] + error_uv
    
        error_delay = Cadet(temp.root)

        delays = numpy.random.uniform(delay_settings[0], delay_settings[1], nsec)
    
        synthetic_error.pump_delay(error_delay, delays)

        flow = numpy.random.normal(flow_settings[0], flow_settings[1], error_delay.root.input.solver.sections.nsec)
    
        synthetic_error.error_flow(error_delay, flow)
    
        load = numpy.random.normal(load_settings[0], load_settings[1], error_delay.root.input.solver.sections.nsec)
    
        synthetic_error.error_load(error_delay, load)


        exp_info = None
        for exp in cache.cache.settings['experiments']:
            if exp['name'] == name:
                exp_info = exp
                break

        result = util.runExperiment(None, exp_info, cache.cache.settings, cache.cache.target, error_delay, error_delay.root.timeout, cache.cache, post_function=post_function)

        if result is not None:
            scores.extend(result['scores'])

            simulations[name] = result['simulation']

            for unit in units:
                outputs['%s_unit_%03d' % (name, int(unit))] = result['simulation'].root.output.solution['unit_%03d' % int(unit)].solution_outlet_comp_000
        else:
            experiment_failed = True

    if experiment_failed:
        return None, None, None
    return scores, simulations, outputs

def generate_synthetic_error(cache):
    if 'kde_synthetic' in cache.settings:
        count_settings = int(cache.settings['kde_synthetic'][0]['count'])
        
        scores_all = []
        simulations_all = {}
        outputs_all = {}

        for scores, simulations, outputs in cache.toolbox.map(synthetic_error_simulation, [cache.json_path] * count_settings):
            if scores and simulations and outputs:

                scores_all.append(scores)

                for key,value in outputs.items():
                    temp = outputs_all.get(key, [])
                    temp.append(value)
                    outputs_all[key] = temp
                                        
        scores = numpy.array(scores_all)

        times = {}
        for simulation_name, simulation_values in simulations_all.items():
            times[simulation_name] = simulation_values.root.input.solver.user_solution_times
            #Only the first item is needed
            break

        dir_base = cache.settings.get('resultsDirBase')
        file = dir_base / 'kde_data.h5'

        with h5py.File(file, 'w') as hf:
            hf.create_dataset('scores', data=scores)

            for output_name, output in outputs_all.items():
                hf.create_dataset(output_name, data=numpy.array(output))

            for time_name, time in times.items():
                hf.create_dataset(time_name, data=times)
                               
        return scores, simulations_all

    return None, None