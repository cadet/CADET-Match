# This module simulates noise (systemic and random) to use for kernel density estimation based on experimental data

# Currently the randomness is

# Pump flow rate
# Pump delay
# Base noise
# Signal noise

import copy
import itertools
import multiprocessing
from pathlib import Path

import numpy
from cadet import H5, Cadet
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity

import CADETMatch.kde_util as kde_util
import CADETMatch.score as score

bw_tol = 1e-4

import sys
import warnings

import joblib
import SALib.sample.sobol_sequence
import scipy.optimize
import scipy.stats

import CADETMatch.cache as cache
import CADETMatch.synthetic_error as synthetic_error
import CADETMatch.util as util
import CADETMatch.sub as sub


def mirror(data):
    data_max = numpy.max(data, 0)

    mirror_index = data_max <= 1.0
    keep_index = data_max > 1.0

    data_min = data_max - data
    data_mask = numpy.ma.masked_equal(data_min, 0.0, copy=False)
    min_value = data_mask.min(axis=0)

    data_mirror = numpy.zeros(data.shape)

    data_mirror[:, mirror_index] = (
        data_max[mirror_index]
        + data_max[mirror_index]
        - numpy.copy(data[:, mirror_index])
        + min_value[mirror_index]
    )
    data_mirror[:, keep_index] = data[:, keep_index]
    full_data = numpy.vstack([data_mirror, data])

    return full_data


def setupKDE(cache):
    scores = generate_synthetic_error(cache)

    mcmcDir = Path(cache.settings["resultsDirMCMC"])

    scores_mirror = mirror(scores)

    scaler = getScaler(scores_mirror)

    scores_scaler = scaler.transform(scores_mirror)

    kde = KernelDensity(kernel="gaussian")

    kde, bandwidth, store = kde_util.get_bandwidth(kde, scores_scaler)

    kde.fit(scores_scaler)

    probability = kde.score_samples(scores_scaler)

    joblib.dump(scaler, mcmcDir / "kde_scaler.joblib")

    joblib.dump(kde, mcmcDir / "kde_score.joblib")

    h5_data = H5()
    h5_data.filename = (mcmcDir / "kde_settings.h5").as_posix()
    h5_data.root.bandwidth = bandwidth
    h5_data.root.store = numpy.array(store)
    h5_data.root.scores = scores
    h5_data.root.scores_mirror = scores_mirror
    h5_data.root.scores_mirror_scaled = scores_scaler
    h5_data.root.probability = probability

    h5_data.save(lock=True)

    sub.graph_kde(cache)

    return kde, scaler


def getScaler(data):
    scaler = preprocessing.RobustScaler().fit(data)
    return scaler


def getKDE(cache):
    mcmcDir = Path(cache.settings["resultsDirMCMC"])

    kde = joblib.load(mcmcDir / "kde_score.joblib")

    scaler = joblib.load(mcmcDir / "kde_scaler.joblib")

    return kde, scaler


def getKDEPrevious(cache):
    if "mcmc_h5" in cache.settings:
        mcmc_h5 = Path(cache.settings["mcmc_h5"])
        mcmcDir = mcmc_h5.parent

        if mcmcDir.exists():
            kde = joblib.load(mcmcDir / "kde_prior.joblib")

            scaler = joblib.load(mcmcDir / "kde_prior_scaler.joblib")

            return kde, scaler
    return None, None


def generate_data(cache):
    scores = generate_synthetic_error(cache)

    mcmcDir = Path(cache.settings["resultsDirMCMC"])
    save_scores = mcmcDir / "scores_used.npy"

    numpy.save(save_scores, scores)

    scores_mirror = mirror(scores)

    scaler = getScaler(scores_mirror)

    scores_scaler = scaler.transform(scores_mirror)

    bandwidth, store = get_bandwidth(scores_scaler)

    return scores, bandwidth


def synthetic_error_simulation(x):
    json_path, error = x
    if json_path != cache.cache.json_path:
        cache.cache.setup_dir(json_path)
        util.setupLog(cache.cache.settings["resultsDirLog"], "main.log")
        cache.cache.setup(json_path)

    scores = []
    outputs = {}
    simulations = {}
    experiment_failed = False

    delays_store = []
    flows_store = []
    load_store = []
    uv_store = {}
    uv_store_norm = {}

    for experiment in cache.cache.settings["errorModel"]:
        delay_settings = experiment["delay"]
        flow_settings = experiment["flow"]
        load_settings = experiment["load"]
        experimental_csv = experiment["experimental_csv"]
        uv_noise = experiment.get("uv_noise", None)
        uv_noise_norm = experiment.get("uv_noise_norm", None)
        units = experiment["units"]
        name = experiment["name"]

        data = numpy.loadtxt(experimental_csv, delimiter=",")
        times = data[:, 0]

        if "file_path" in experiment:
            template_path = Path(experiment["file_path"])
        else:
            resultsDir = cache.cache.settings["resultsDir"]
            if "resultsDirOriginal" in cache.cache.settings:
                resultsDir = Path(cache.cache.settings["resultsDirOriginal"])

            template_path = resultsDir / "misc" / ("template_%s_base.h5" % name)

        temp = Cadet()
        temp.filename = template_path.as_posix()
        temp.load()

        util.setupSimulation(temp, times, name, cache.cache)

        nsec = temp.root.input.solver.sections.nsec

        for unit in units:
            unit_name = "unit_%03d" % unit
            for comp in range(temp.root.input.model[unit_name].ncomp):
                comp_name = "solution_outlet_comp_%03d" % comp

                if uv_noise is None:
                    error_uv = numpy.zeros(
                        len(temp.root.input.solver.user_solution_times)
                    )
                else:
                    error_uv = numpy.random.normal(
                        uv_noise[0],
                        uv_noise[1],
                        len(temp.root.input.solver.user_solution_times),
                    )
                uv_store["%s_%s_%s" % (name, unit, comp)] = error_uv

                if uv_noise_norm is None:
                    error_uv = numpy.ones(
                        len(temp.root.input.solver.user_solution_times)
                    )
                else:
                    error_uv = numpy.random.normal(
                        uv_noise_norm[0],
                        uv_noise_norm[1],
                        len(temp.root.input.solver.user_solution_times),
                    )
                uv_store_norm["%s_%s_%s" % (name, unit, comp)] = error_uv

        def post_function(simulation):
            # baseline drift need to redo this
            # error_slope = numpy.random.normal(error_slope_settings[0], error_slope_settings[1], 1)[0]

            for unit in units:
                unit_name = "unit_%03d" % unit
                for comp in range(simulation.root.input.model[unit_name].ncomp):
                    comp_name = "solution_outlet_comp_%03d" % comp
                    error_uv = uv_store["%s_%s_%s" % (name, unit, comp)]
                    error_norm = uv_store_norm["%s_%s_%s" % (name, unit, comp)]
                    simulation.root.output.solution[unit_name][comp_name] = (
                        simulation.root.output.solution[unit_name][comp_name]
                        * error_norm
                        + error_uv
                    )

        error_delay = Cadet(temp.root)

        # delays = numpy.random.uniform(delay_settings[0], delay_settings[1], nsec)
        delays = error[name]["pump_delays"]
        delays_store.extend(delays)

        synthetic_error.pump_delay(error_delay, delays)

        # flow = numpy.random.normal(flow_settings[0], flow_settings[1], error_delay.root.input.solver.sections.nsec)
        flow = error[name]["flow_rates"]
        flows_store.extend(flow)

        synthetic_error.error_flow(error_delay, flow)

        # load = numpy.random.normal(load_settings[0], load_settings[1], error_delay.root.input.solver.sections.nsec)
        load = error[name]["loading_concentrations"]
        load_store.extend(load)

        synthetic_error.error_load(error_delay, load)

        exp_info = None
        for exp in cache.cache.settings["experiments"]:
            if exp["name"] == name:
                exp_info = exp
                break

        result = util.runExperiment(
            None,
            exp_info,
            cache.cache.settings,
            cache.cache.target,
            error_delay,
            error_delay.root.timeout,
            cache.cache,
            post_function=post_function,
        )

        if result is not None:
            scores.extend(result["scores"])

            simulations[name] = result["simulation"]

            for unit in units:
                unit_name = "unit_%03d" % unit
                for comp in range(
                    result["simulation"].root.input.model[unit_name].ncomp
                ):
                    outputs[
                        "%s_unit_%03d_comp_%03d" % (name, int(unit), comp)
                    ] = result["simulation"].root.output.solution[
                        "unit_%03d" % int(unit)
                    ][
                        "solution_outlet_comp_%03d" % comp
                    ]
        else:
            experiment_failed = True

    if experiment_failed:
        return None, None, None, None

    errors = {}
    errors["delays"] = delays_store
    errors["flows"] = flows_store
    errors["load"] = load_store
    errors["uv_store"] = uv_store

    return scores, simulations, outputs, errors


def get_section_counts(cache):
    sections = []

    for experiment in cache.cache.settings["errorModel"]:
        name = experiment["name"]

        if "file_path" in experiment:
            template_path = Path(experiment["file_path"])
        else:
            resultsDir = cache.cache.settings["resultsDir"]
            if "resultsDirOriginal" in cache.cache.settings:
                resultsDir = Path(cache.cache.settings["resultsDirOriginal"])

            template_path = resultsDir / "misc" / ("template_%s_base.h5" % name)

        temp = Cadet()
        temp.filename = template_path.as_posix()
        temp.load()

        nsec = temp.root.input.solver.sections.nsec

        sections.append(nsec)

    return sections


def generate_error_sequence(cache):
    errors = {}
    count = cache.settings["errorModelCount"]
    for experiment in cache.settings["errorModel"]:
        delay_settings = experiment["delay"]
        flow_settings = experiment["flow"]
        load_settings = experiment["load"]
        name = experiment["name"]

        if "file_path" in experiment:
            template_path = Path(experiment["file_path"])
        else:
            resultsDir = cache.cache.settings["resultsDir"]
            if "resultsDirOriginal" in cache.settings:
                resultsDir = Path(cache.settings["resultsDirOriginal"])

            template_path = resultsDir / "misc" / ("template_%s_base.h5" % name)

        temp = Cadet()
        temp.filename = template_path.as_posix()
        temp.load()

        nsec = temp.root.input.solver.sections.nsec

        error = {}

        # total space is 1 delay per section 1 flow rate per new section and 1 load per new section
        delay_size = nsec
        flow_size = 2 * nsec
        load_size = 2 * nsec
        total_size = delay_size + flow_size + load_size

        errors_exp = SALib.sample.sobol_sequence.sample(count, total_size)

        # set delays
        block = errors_exp[:, :delay_size]
        if delay_settings[0] != delay_settings[1]:
            block = scipy.stats.uniform.ppf(
                block, delay_settings[0], delay_settings[1] - delay_settings[0]
            )
        else:
            block = numpy.ones(block.shape) * delay_settings[0]

        error["pump_delays"] = block

        # set flow range changes
        lb = delay_size
        ub = delay_size + flow_size

        # grab the block to work with easier
        block = errors_exp[:, lb:ub]

        # sobol sequence is [0, 1) and 0 needs to be removed and set to the same distance as the closest distance to 1
        min_value = 1 - numpy.max(block)

        block[block < min_value] = min_value

        if flow_settings[1] != 0:
            block = scipy.stats.norm.ppf(block, flow_settings[0], flow_settings[1])
        else:
            block = numpy.ones(block.shape) * flow_settings[0]

        error["flow_rates"] = block

        # set loading changes
        lb = ub
        ub = lb + load_size

        # grab the block to work with easier
        block = errors_exp[:, lb:ub]

        # sobol sequence is [0, 1) and 0 needs to be removed and set to the same distance as the closest distance to 1
        min_value = 1 - numpy.max(block)

        block[block < min_value] = min_value

        if load_settings[1] != 0:
            block = scipy.stats.norm.ppf(block, load_settings[0], load_settings[1])
        else:
            block = numpy.ones(block.shape) * load_settings[0]

        error["loading_concentrations"] = block

        errors[name] = error

    return errors


def split_errors(errors_all):
    errors = []
    for name, experiment in errors_all.items():
        length = len(experiment["flow_rates"])
        break

    temp = {}
    for key in errors_all:
        temp[key] = {}

    for i in range(length):
        errors.append({key: {} for key in errors_all})

    for name, experiment in errors_all.items():
        for error_name, error_value in experiment.items():
            for idx in range(length):
                errors[idx][name][error_name] = error_value[idx, :]

    return errors


def keep_data(data, lb=0.5, ub=99.5):
    cutoffs_lb, cutoffs_ub = numpy.percentile(data, [lb, ub], axis=0)
    remove = numpy.zeros(data.shape)

    for i in range(data.shape[1] - 2):
        remove[:, i] = data[:, i] > cutoffs_lb[i]
    remove[:, -2] = (data[:, -2] > cutoffs_lb[-2]) & (data[:, -2] < cutoffs_ub[-2])
    remove[:, -1] = (data[:, -1] > cutoffs_lb[-1]) & (data[:, -1] < cutoffs_ub[-1])

    remove = numpy.all(remove, axis=1)

    return remove


def generate_synthetic_error(cache):
    count_settings = int(cache.settings["errorModelCount"])
    if "errorModel" in cache.settings:
        scores_all = []
        times = {}
        outputs_all = {}

        uv_store_all = {}

        errors_all = generate_error_sequence(cache)

        errors_split = split_errors(errors_all)
        json_path = itertools.repeat(cache.json_path)

        indexes = []

        for idx, (scores, simulations, outputs, errors) in enumerate(
            cache.toolbox.map(synthetic_error_simulation, zip(json_path, errors_split))
        ):
            if scores and simulations and outputs:
                indexes.append(idx)

                for key, value in errors["uv_store"].items():
                    uv = uv_store_all.get(key, [])
                    uv.append(value)
                    uv_store_all[key] = uv

                scores_all.append(scores)

                for key, value in outputs.items():
                    temp = outputs_all.get(key, [])
                    temp.append(value)
                    outputs_all[key] = temp

                for key, sim in simulations.items():
                    if key not in times:
                        times[key] = sim.root.output.solution.solution_times

        indexes = numpy.array(indexes)

        for name, experiment in errors_all.items():
            for error_name, error_value in experiment.items():
                errors_all[name][error_name] = error_value[indexes, :]

        scores = numpy.array(scores_all)

        keep_idx = keep_data(scores)

        kept = int(numpy.sum(keep_idx))
        removed = int(len(scores) - kept)

        dir_base = cache.settings.get("resultsDirBase")
        file = dir_base / "kde_data.h5"

        kde_data = H5()
        kde_data.filename = file.as_posix()

        kde_data.root.kept = kept
        kde_data.root.removed = removed

        kde_data.root.scores = scores[keep_idx, :]

        kde_data.root.original.scores = scores

        for output_name, output in outputs_all.items():
            kde_data.root[output_name] = numpy.array(output)[keep_idx, :]
            kde_data.root.original[output_name] = numpy.array(output)[keep_idx, :]

        for time_name, time in times.items():
            kde_data.root["%s_time" % time_name] = time
            kde_data.root.original["%s_time" % time_name] = time

        for name, experiment in errors_all.items():
            for error_name, error_value in experiment.items():
                kde_data.root.errors[name][error_name] = error_value[keep_idx, :]
                kde_data.root.original.errors[name][error_name] = error_value[
                    keep_idx, :
                ]

        for key, value in uv_store_all.items():
            kde_data.root.uv_store[key] = numpy.array(value)[keep_idx, :]
            kde_data.root.original.uv_store[key] = numpy.array(value)[keep_idx, :]

        kde_data.save(lock=True)

        return scores[keep_idx, :]

    return None


def convert_to_array(seq):
    "this converts a sequence of arrays to a single numpy array and pads uneven rows with 0"
    max_len = numpy.max([len(a) for a in seq])
    return numpy.asarray(
        [
            numpy.pad(a, (0, max_len - len(a)), "constant", constant_values=0)
            for a in seq
        ]
    )
