import copy
import csv
import decimal as decim
import hashlib
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import jstyleson
import numpy
import pandas
import psutil
import scipy.signal
from addict import Dict
from cadet import H5, Cadet
from deap import tools

import CADETMatch.calc_coeff as calc_coeff
import CADETMatch.sub as sub

decim.getcontext().prec = 64
__logBase10of2_decim = decim.Decimal(2).log10()
__logBase10of2 = float(__logBase10of2_decim)

import logging
import os

import SALib.sample.sobol_sequence
import scipy.interpolate

import CADETMatch.loggerwriter as loggerwriter
import CADETMatch.pareto as pareto
import CADETMatch.synthetic_error as synthetic_error

# numpy.warnings.filterwarnings('error', category=numpy.VisibleDeprecationWarning)


def get_times_values(simulation, target, selected=None):

    try:
        times = simulation.root.output.solution.solution_times

        isotherm = target["isotherm"]

        if isinstance(isotherm, list):
            values = numpy.sum([simulation[i] for i in isotherm], 0)
        else:
            values = simulation[isotherm]
    except (AttributeError, KeyError):
        times = simulation[:, 0]
        values = simulation[:, 1]

    if selected is None:
        selected = target["selected"]

    return times[selected], values[selected] * target["factor"]


def sse(data1, data2):
    return numpy.sum((data1 - data2) ** 2)


def find_peak(times, data):
    "Return tuples of (times,data) for the peak we need"
    minIdx = numpy.argmin(data)
    maxIdx = numpy.argmax(data)

    return (times[maxIdx], data[maxIdx]), (times[minIdx], data[minIdx])


def find_breakthrough(times, data):
    "return tuple of time,value for the start breakthrough and end breakthrough"
    selected = data > 0.999 * max(data)
    selected_times = times[selected]
    return (selected_times[0], max(data)), (selected_times[-1], max(data))


def generateIndividual(icls, size, imin, imax, cache):
    return icls(numpy.random.uniform(imin, imax))


def initIndividual(icls, cache, content):
    return icls(content)


def sobolGenerator(icls, cache, n):
    if n > 0:
        populationDimension = len(cache.MIN_VALUE)
        populationSize = n
        sobol = SALib.sample.sobol_sequence.sample(populationSize, populationDimension)
        lb = numpy.array(cache.MIN_VALUE)
        ub = numpy.array(cache.MAX_VALUE)
        sobol = sobol * (ub - lb) + lb
        data = numpy.apply_along_axis(list, 1, sobol)
        data = list(map(icls, data))
        return data
    else:
        return []


def calcMetaScores(scores, cache):
    scores = numpy.array(scores)[cache.meta_mask]
    if cache.allScoreNorm:
        prod_score = product_score(scores)
        min_score = min(scores)
        mean_score = sum(scores) / len(scores)
        human = [prod_score, min_score, mean_score]
    elif cache.allScoreSSE:
        if cache.MultiObjectiveSSE:
            # scores = numpy.abs(scores)
            prod_score = product_score(scores)
            min_score = min(scores)
            mean_score = sum(scores) / len(scores)
            human = [prod_score, min_score, mean_score]
        else:
            sse = numpy.sum(numpy.array(scores))
            human = [sse, sse, sse]
    return human


def product_score(values):
    values = numpy.array(values)
    if numpy.all(values >= 0.0):
        return numpy.prod(values) ** (1.0 / len(values))
    else:
        return -numpy.prod(numpy.abs(values)) ** (1.0 / len(values))


def saveExperiments(save_name_base, settings, target, results, directory, file_pattern):
    for experiment in settings["experiments"]:
        experimentName = experiment["name"]
        simulation = results[experimentName]["simulation"]

        dst = Path(directory, file_pattern % (save_name_base, experimentName))

        same = False
        if dst.is_file():
            # load tolerance to see if the files are actually the same
            sim_dst = Cadet()
            sim_dst.filename = dst.as_posix()
            sim_dst.load(
                paths=[
                    "/input/solver/time_integrator",
                ]
            )

            dst_tol = sim_dst.root.input.solver.time_integrator.abstol
            sim_tol = simulation.root.input.solver.time_integrator.abstol
            same = sim_tol == dst_tol

        if same:
            return False
        else:
            simulation.filename = dst.as_posix()

            for (header, score) in zip(
                experiment["headers"], results[experimentName]["scores"]
            ):
                simulation.root.score[header] = score
            simulation.save()

    return True


def convert_individual_inputorder(individual, cache):
    cadetValues = []

    idx = 0
    for parameter in cache.parameters:
        count = parameter.count
        if count:
            seq = individual[idx : idx + count]
            values = parameter.untransform_inputorder(seq)
            cadetValues.extend(values)
            idx += count

    return cadetValues


def convert_individual(individual, cache):
    cadetValues = []
    cadetValuesExtended = []

    idx = 0
    for parameter in cache.parameters:
        count = parameter.count
        if count:
            seq = individual[idx : idx + count]
            values, headerValues = parameter.untransform(seq)
            cadetValues.extend(values)
            cadetValuesExtended.extend(headerValues)
            idx += count

    return cadetValues, cadetValuesExtended


def convert_individual_grad(individual, cache):
    cadetValues = []

    idx = 0
    for parameter in cache.parameters:
        count = parameter.count
        if count:
            seq = individual[idx : idx + count]
            values = parameter.grad_untransform(seq)
            cadetValues.extend(values)
            idx += count

    return cadetValues


def convert_population(population, cache):
    cadetValues = numpy.zeros(population.shape)

    idx = 0
    for parameter in cache.parameters:
        count = parameter.count
        if count:
            matrix = population[:, idx : idx + count]
            values = parameter.untransform_matrix(matrix)
            cadetValues[:, idx : idx + count] = values
            idx += count

    return cadetValues


def convert_population_inputorder(population, cache):
    if len(population):
        cadetValues = numpy.zeros(population.shape)

        idx = 0
        for parameter in cache.parameters:
            count = parameter.count
            if count:
                matrix = population[:, idx : idx + count]
                values = parameter.untransform_matrix_inputorder(matrix)
                cadetValues[:, idx : idx + count] = values
                idx += count

        return cadetValues
    else:
        return numpy.array([])


def convert_individual_inverse(individual, cache):
    return numpy.array([f(v) for f, v in zip(cache.settings["transform"], individual)])


def convert_individual_inverse_grad(individual, cache):
    return numpy.array(
        [f(v) for f, v in zip(cache.settings["grad_transform"], individual)]
    )


def set_simulation(individual, simulation, settings, cache, experiment):
    multiprocessing.get_logger().debug("individual %s", individual)

    cadetValues = []
    cadetValuesKEQ = []

    idx = 0
    for parameter in cache.parameters:
        count = parameter.count
        # even if count is 0 this needs to be run so that setSimulation will be run for things like calculations
        seq = individual[idx : idx + count]
        values, headerValues = parameter.setSimulation(simulation, seq, experiment)
        cadetValues.extend(values)
        cadetValuesKEQ.extend(headerValues)
        idx += count

    multiprocessing.get_logger().debug("finished setting hdf5")
    return cadetValues, cadetValuesKEQ


def getBoundOffset(unit):
    if unit.unit_type == b"CSTR":
        NBOUND = unit.nbound
    else:
        NBOUND = unit.discretization.nbound

    if not len(NBOUND):
        "If NBOUND is empty it is all zero"
        NBOUND = [0.0] * unit.ncomp

    boundOffset = numpy.cumsum(
        numpy.concatenate(
            [
                [
                    0,
                ],
                NBOUND,
            ]
        )
    )
    return boundOffset


def runExperiment(
    individual,
    experiment,
    settings,
    target,
    template_sim,
    timeout,
    cache,
    post_function=None,
):
    handle, path = tempfile.mkstemp(suffix=".h5", dir=cache.tempDir)
    os.close(handle)

    simulation = Cadet(template_sim.root)
    simulation.filename = path

    simulation.root.input.solver.nthreads = int(settings.get("nThreads", 1))

    if individual is not None:
        cadetValues, cadetValuesKEQ = set_simulation(
            individual, simulation, settings, cache, experiment
        )
    else:
        cadetValues = []
        cadetValuesKEQ = []

    simulation.save()

    try:
        simulation.run(timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        multiprocessing.get_logger().warn("Simulation Timed Out")
        os.remove(path)
        return None

    except subprocess.CalledProcessError as error:
        multiprocessing.get_logger().error("The simulation failed %s", individual)
        logError(cache, cadetValuesKEQ, error)
        return None

    # read sim data
    simulation.load(paths=["/meta", "/output"], update=True)
    os.remove(path)

    simulationFailed = isinstance(simulation.root.output.solution.solution_times, Dict)
    if simulationFailed:
        multiprocessing.get_logger().error(
            "%s sim must have failed %s", individual, path
        )
        return None
    multiprocessing.get_logger().debug("Everything ran fine")

    if post_function:
        post_function(simulation)

    temp = {}
    temp["simulation"] = simulation
    temp["path"] = path
    temp["scores"] = []
    temp["error"] = 0.0
    temp["error_count"] = 0.0
    temp["cadetValues"] = cadetValues
    temp["cadetValuesKEQ"] = cadetValuesKEQ

    if individual is not None:
        temp["individual"] = tuple(individual)
    temp["diff"] = []
    temp["minimize"] = []
    temp["sim_time"] = []
    temp["sim_value"] = []
    temp["exp_value"] = []

    for feature in experiment["features"]:
        featureType = feature["type"]
        featureName = feature["name"]

        if featureType in cache.scores:
            try:
                (
                    scores,
                    sse,
                    sse_count,
                    sim_time,
                    sim_value,
                    exp_value,
                    minimize,
                ) = cache.scores[featureType].run(
                    temp, target[experiment["name"]][featureName]
                )
            except TypeError:
                return None

            diff = sim_value - exp_value

            temp["scores"].extend(scores)
            temp["error"] += sse
            temp["error_count"] += sse_count
            temp["diff"].extend(diff)
            temp["minimize"].extend(minimize)
            temp["sim_time"].append(sim_time)
            temp["sim_value"].append(sim_value)
            temp["exp_value"].append(exp_value)

    return temp


def logError(cache, values, error):
    with cache.error_path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)

        row = list(values)
        row.append(error.returncode)
        row.append(error.stdout)
        row.append(error.stderr)

        writer.writerow(row)


def repeatSimulation(idx):
    "read the original json file and copy it to a subdirectory for each repeat and change where the target data is written"
    settings_file = Path(sys.argv[1])
    with settings_file.open() as json_data:
        settings = jstyleson.load(json_data)

        baseDir = settings.get("baseDir", None)
        if baseDir is not None:
            baseDir = Path(baseDir)
            settings["resultsDir"] = baseDir / settings["resultsDir"]

        resultDir = Path(settings["resultsDir"]) / str(idx)
        resultDir.mkdir(parents=True, exist_ok=True)

        settings["resultsDirOriginal"] = settings["resultsDir"].as_posix()
        settings["resultsDir"] = resultDir.as_posix()

        if baseDir is not None:
            settings["baseDir"] = baseDir.as_posix()

        new_settings_file = resultDir / settings_file.name
        with new_settings_file.open(mode="w") as json_data:
            jstyleson.dump(settings, json_data, indent=4, sort_keys=True)
        return new_settings_file


def setupMCMC(cache):
    "read the original json file and make an mcmc file based on it with new boundaries"
    settings_file = Path(sys.argv[1])
    with settings_file.open() as json_data:
        settings = jstyleson.load(json_data)
        settings["continueMCMC"] = 0

        baseDir = settings.get("baseDir", None)
        if baseDir is not None:
            baseDir = Path(baseDir)
            settings["resultsDir"] = baseDir / settings["resultsDir"]

        resultDirOriginal = Path(settings["resultsDir"])
        resultsOriginal = resultDirOriginal / "result.h5"
        resultDir = resultDirOriginal / "mcmc_refine"
        resultDir.mkdir(parents=True, exist_ok=True)

        settings["resultsDirOriginal"] = resultDirOriginal.as_posix()
        settings["resultsDir"] = resultDir.as_posix()
        settings["PreviousResults"] = resultsOriginal.as_posix()

        if baseDir is not None:
            settings["baseDir"] = baseDir.as_posix()

        if "mcmc_h5" in settings:
            update_json_mcmc(settings)

        settings["searchMethod"] = "MCMC"
        settings["graphSpearman"] = 0

        for experiment in settings["experiments"]:
            foundAbsoluteTime = False
            foundAbsoluteHeight = False
            for feature in experiment["features"]:
                if feature["type"] == "AbsoluteTime":
                    foundAbsoluteTime = True
                if feature["type"] == "AbsoluteHeight":
                    foundAbsoluteHeight = True
            if foundAbsoluteTime is False:
                experiment["features"].append(
                    {"name": "AbsoluteTime", "type": "AbsoluteTime"}
                )
            if foundAbsoluteHeight is False:
                experiment["features"].append(
                    {"name": "AbsoluteHeight", "type": "AbsoluteHeight"}
                )

        if "kde_synthetic" in settings:
            individual = getBestIndividual(cache)
            found = createSimulationBestIndividual(individual, cache)
            for experiment in settings["kde_synthetic"]:
                experiment["file_path"] = found[experiment["name"]]

        new_settings_file = resultDir / settings_file.name
        with new_settings_file.open(mode="w") as json_data:
            jstyleson.dump(settings, json_data, indent=4, sort_keys=False)
        return new_settings_file


def update_json_mcmc(settings):
    data = H5()
    data.filename = settings["mcmc_h5"]
    data.load(paths=["/bounds_change/json"])
    json_data = jstyleson.loads(str(data.root.bounds_change.json, "ascii"))

    if "parameters_mcmc" in settings:
        new_parameters = settings["parameters_mcmc"]

        for a, b in zip(new_parameters, json_data):
            if (
                a["transform"] == b["transform"]
                and a["location"].split("/")[-1] == b["location"].split("/")[-1]
            ):
                a["min"] = b["min"]
                a["max"] = b["max"]
            else:
                multiprocessing.get_logger().info(
                    "parameters_mcmc does not have the same transform and variables in the same order as the prior, MCMC cannot continue until this is fixed"
                )
                sys.exit()

        settings["parameters"].extend(new_parameters)
    else:
        settings["parameters"].extend(json_data)


def setupAltFeature(cache, name):
    "read the original json file and make an mcmc file based on it with new boundaries"
    settings_file = Path(sys.argv[1])
    with settings_file.open() as json_data:
        settings = jstyleson.load(json_data)

        baseDir = settings.get("baseDir", None)
        if baseDir is not None:
            baseDir = Path(baseDir)
            settings["resultsDir"] = baseDir / settings["resultsDir"]

        resultDirOriginal = Path(settings["resultsDir"])
        resultsOriginal = resultDirOriginal / "result.h5"
        resultDir = resultDirOriginal / "altScore" / name
        resultDir.mkdir(parents=True, exist_ok=True)

        settings["resultsDirOriginal"] = resultDirOriginal.as_posix()
        settings["resultsDir"] = resultDir.as_posix()
        settings["PreviousResults"] = resultsOriginal.as_posix()

        if baseDir is not None:
            settings["baseDir"] = baseDir.as_posix()

        settings["searchMethod"] = "AltScore"

        for experiment in settings["experiments"]:
            for feature in experiment["featuresAlt"]:
                if feature["name"] == name:
                    experiment["features"] = feature["features"]
            del experiment["featuresAlt"]

        new_settings_file = resultDir / settings_file.name
        with new_settings_file.open(mode="w") as json_data:
            jstyleson.dump(settings, json_data, indent=4, sort_keys=True)
        return new_settings_file


def createSimulationBestIndividual(individual, cache):
    temp = {}
    for experiment in cache.settings["experiments"]:
        name = experiment["name"]
        templatePath = Path(
            cache.settings["resultsDirMisc"], "template_%s_base.h5" % name
        )
        templateSim = Cadet()
        templateSim.filename = templatePath.as_posix()
        templateSim.load()

        cadetValues, cadetValuesKEQ = set_simulation(
            individual, templateSim, cache.settings, cache, experiment
        )

        bestPath = Path(cache.settings["resultsDirMisc"], "best_%s_base.h5" % name)
        templateSim.filename = bestPath.as_posix()
        templateSim.save()
        temp[name] = bestPath.as_posix()
    return temp


def getBestIndividual(cache):
    "return the path to the best item based on meta min score"
    progress_path = Path(cache.settings["resultsDirBase"]) / "result.h5"
    results = H5()
    results.filename = progress_path.as_posix()
    results.load(paths=["/meta_score", "/meta_population"])

    idx = numpy.argmax(results.root.meta_score[:, 1])
    individual = results.root.meta_population[idx, :]
    return individual


def copyCSVWithNoise(idx, center, noise):
    "read the original json file and create a new set of simulation data and simulation file in a subdirectory with noise"
    settings_file = Path(sys.argv[1])
    with settings_file.open() as json_data:
        settings = jstyleson.load(json_data)

        baseDir = Path(settings["resultsDir"]) / str(idx)
        baseDir.mkdir(parents=True, exist_ok=True)

        settings["resultsDir"] = str(baseDir)

        del settings["bootstrap"]

        # find CSV files
        for experiment in settings["experiments"]:
            if "csv" in experiment:
                data = numpy.genfromtxt(experiment["csv"], delimiter=",")
                addNoise(data, center, noise)
                csv_path = Path(experiment["csv"])
                new_csv_path = baseDir / csv_path.name
                numpy.savetxt(str(new_csv_path), data, delimiter=",")
                experiment["csv"] = str(new_csv_path)
            for feature in experiment["features"]:
                if "csv" in feature:
                    data = numpy.genfromtxt(feature["csv"], delimiter=",")
                    addNoise(data, center, noise)
                    csv_path = Path(feature["csv"])
                    new_csv_path = baseDir / csv_path.name
                    numpy.savetxt(str(new_csv_path), data, delimiter=",")
                    feature["csv"] = str(new_csv_path)

        new_settings_file = baseDir / settings_file.name
        with new_settings_file.open(mode="w") as json_data:
            jstyleson.dump(settings, json_data)
        return new_settings_file


def addNoise(array, center, noise):
    "add noise to an array"
    maxValue = numpy.max(array[:, 1])
    randomNoise = numpy.random.normal(center, noise * maxValue, len(array))
    array[:, 1] += randomNoise


def bestMinScore(hof):
    "find the best score based on the minimum of the scores"
    idxMax = numpy.argmax([min(i.fitness.values) for i in hof])
    return hof[idxMax]


def fracStat(time_center, value):
    mean_time = numpy.sum(time_center * value) / numpy.sum(value)
    variance_time = numpy.sum((time_center - mean_time) ** 2 * value) / numpy.sum(value)
    skew_time = numpy.sum((time_center - mean_time) ** 3 * value) / numpy.sum(value)

    mean_value = numpy.sum(time_center * value) / numpy.sum(time_center)
    variance_value = numpy.sum((value - mean_value) ** 2 * time_center) / numpy.sum(
        time_center
    )
    skew_value = numpy.sum((value - mean_value) ** 3 * time_center) / numpy.sum(
        time_center
    )

    return mean_time, variance_time, skew_time, mean_value, variance_value, skew_value


def fractionate_spline(start_seq, stop_seq, spline):
    temp = []
    for (start, stop) in zip(start_seq, stop_seq):
        temp.append(spline.integral(start, stop) / (stop - start))
    return numpy.array(temp)


def find_opt_poly(x, y, index):
    if index == 0:
        indexes = numpy.array([index, index + 1, index + 2])
    elif index == (len(x) - 1):
        indexes = numpy.array([index - 2, index - 1, index])
    else:
        indexes = numpy.array([index - 1, index, index + 1])

    x = x[indexes]
    y = y[indexes]

    if x[0] > x[-1]:  # need to invert order
        x = x[::-1]
        y = y[::-1]

    poly, res = numpy.polynomial.Polynomial.fit(x, y, 2, full=True)
    try:
        root = poly.deriv().roots()[0]
    except IndexError:
        # this happens if all y values are the same in which case just take the center value
        root = x[indexes[1]]
    root = numpy.clip(root, x[0], x[1])
    return root, x, y


def fractionate(start_seq, stop_seq, times, values):
    temp = []
    for (start, stop) in zip(start_seq, stop_seq):
        selected = (times >= start) & (times <= stop)
        local_times = times[selected]
        local_values = values[selected]

        # need to use the actual start and stop times from the data no what we want to cut at
        stop = local_times[-1]
        start = local_times[0]

        temp.append(numpy.trapz(local_values, local_times) / (stop - start))
    return numpy.array(temp)


def metaCSV(cache):
    repeat = int(cache.settings["repeat"])

    generations = []
    timeToComplete = []
    timePerGeneration = []
    totalCPUTime = []
    paretoFront = []
    avergeScore = []
    minimumScore = []
    productScore = []
    paretoAverageScore = []
    paretoMinimumScore = []
    paretoProductScore = []
    population = None
    dimensionIn = None
    dimensionOut = None
    searchMethod = None

    # read base progress csv and append each score
    base_dir = Path(cache.settings["resultsDirOriginal"])
    paths = [
        base_dir / "progress.csv",
    ]
    for idx in range(repeat):
        paths.append(base_dir / str(idx) / "progress.csv")

    for idx, path in enumerate(paths):
        data = pandas.read_csv(path)

        if idx == 0:
            population = data["Population"].iloc[-1]
            dimensionIn = data["Dimension In"].iloc[-1]
            dimensionOut = data["Dimension Out"].iloc[-1]
            searchMethod = data["Search Method"].iloc[-1]

        generations.append(data["Generation"].iloc[-1])
        timeToComplete.append(data["Elapsed Time"].iloc[-1])
        timePerGeneration.append(numpy.mean(data["Generation Time"]))
        paretoFront.append(data["Pareto Front"].iloc[-1])
        avergeScore.append(data["Average Score"].iloc[-1])
        minimumScore.append(data["Minimum Score"].iloc[-1])
        productScore.append(data["Product Score"].iloc[-1])
        totalCPUTime.append(data["Total CPU Time"].iloc[-1])
        paretoAverageScore = [data["Pareto Mean Average Score"].iloc[-1]]
        paretoMinimumScore = [data["Pareto Mean Minimum Score"].iloc[-1]]
        paretoProductScore = [data["Pareto Mean Product Score"].iloc[-1]]
        paretoProductScore = [data["Pareto Mean Product Score"].iloc[-1]]

    meta_progress = base_dir / "meta_progress.csv"
    with meta_progress.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)

        writer.writerow(
            [
                "Population",
                "Dimension In",
                "Dimension Out",
                "Search Method",
                "Generation",
                "Generation STDDEV",
                "Elapsed Time",
                "Elapsed Time STDDEV",
                "Generation Time",
                "Geneation Time STDDEV",
                "Pareto Front",
                "Paret Front STDDEV",
                "Average Score",
                "Average Score STDDEV",
                "Minimum Score",
                "Minimum Score STDDEV",
                "Product Score",
                "Product Score STDDEV",
                "Pareto Mean Average Score",
                "Pareto Mean Average Score STDDEV",
                "Pareto Mean Minimum Score",
                "Pareto Mean Minimum Score STDDEV",
                "Pareto Mean Product Score",
                "Pareto Mean Product Score STDDEV",
                "Total CPU Time",
                "Total CPU Time STDDEV",
            ]
        )

        writer.writerow(
            [
                population,
                dimensionIn,
                dimensionOut,
                searchMethod,
                numpy.mean(generations),
                numpy.std(generations),
                numpy.mean(timeToComplete),
                numpy.std(timeToComplete),
                numpy.mean(timePerGeneration),
                numpy.std(timePerGeneration),
                numpy.mean(paretoFront),
                numpy.std(paretoFront),
                numpy.mean(avergeScore),
                numpy.std(avergeScore),
                numpy.mean(minimumScore),
                numpy.std(minimumScore),
                numpy.mean(productScore),
                numpy.std(productScore),
                numpy.mean(paretoAverageScore),
                numpy.std(paretoAverageScore),
                numpy.mean(paretoMinimumScore),
                numpy.std(paretoMinimumScore),
                numpy.mean(paretoProductScore),
                numpy.std(paretoProductScore),
                numpy.mean(totalCPUTime),
                numpy.std(totalCPUTime),
            ]
        )


def update_result_data(cache, ind, fit, result_data, results, meta_scores):
    if result_data is not None and results is not None:
        result_data["input"].append(tuple(ind))

        if ind.strategy is not None:
            result_data["strategy"].append(tuple(ind.strategy))
        if ind.mean is not None:
            result_data["mean"].append(tuple(ind.mean))
        if ind.confidence is not None:
            result_data["confidence"].append(tuple(ind.confidence))
        result_data["output"].append(tuple(fit))
        result_data["output_meta"].append(tuple(meta_scores))

        for result in results.values():
            result_data["input_transform"].append(tuple(result["cadetValues"]))
            result_data["input_transform_extended"].append(
                tuple(result["cadetValuesKEQ"])
            )

            # All results have the same parameter set so we only need the first one
            break

        if cache.fullTrainingData:
            if "results" not in result_data:
                result_data["results"] = {}

            if "times" not in result_data:
                result_data["times"] = {}

            for experimentName, experiment in results.items():
                units_used = cache.target[experimentName]["units_used"]
                sim = experiment["simulation"]
                times = sim.root.output.solution.solution_times

                timeName = "%s_time" % experimentName

                if timeName not in result_data["times"]:
                    result_data["times"][timeName] = times

                for unitName in units_used:
                    unit = sim.root.output.solution[unitName]
                    for solutionName, solution in unit.items():
                        if solutionName.startswith("solution_outlet_comp"):
                            comp = solutionName.replace("solution_outlet_comp_", "")

                            name = "%s_%s_%s" % (experimentName, unitName, comp)

                            if name not in result_data["results"]:
                                result_data["results"][name] = []

                            result_data["results"][name].append(tuple(solution))


def calcFitness(scores_orig, cache):
    scores = numpy.array(scores_orig)[cache.meta_mask]

    if cache.allScoreSSE and cache.MultiObjectiveSSE is False:
        scores = (numpy.sum(scores),)
    return tuple(scores)


def process_population(
    toolbox,
    cache,
    population,
    fitnesses,
    writer,
    csvfile,
    halloffame,
    meta_hof,
    progress_hof,
    generation,
    result_data=None,
):
    csv_lines = []
    meta_csv_lines = []

    made_progress = False

    if meta_hof:
        best_min = max([i.fitness.values[2] for i in meta_hof.items])
    else:
        # just in case if the SSE score is used this will make sure that it has to be greater
        best_min = -1e308

    lookup = create_lookup(population)

    last_time = time.time()
    elapsed = cache.progress_elapsed_time

    for idx, result in enumerate(fitnesses):

        if (time.time() - last_time) > elapsed:
            percent = idx / len(population)
            multiprocessing.get_logger().info(
                "Generation %s approximately %.1f %% complete with %s/%s done",
                generation,
                percent * 100,
                idx,
                len(population),
            )
            last_time = time.time()

        fit, csv_line, meta_score, results, individual = result

        ind = pop_lookup(lookup, individual)

        save_name_base = hashlib.md5(
            str(list(ind)).encode("utf-8", "ignore")
        ).hexdigest()

        ind.fitness.values = calcFitness(fit, cache)
        ind.csv_line = [time.ctime(), save_name_base] + csv_line

        ind_meta = toolbox.individualMeta(ind)

        ind_meta.fitness.values = meta_score
        ind_meta.csv_line = [time.ctime(), save_name_base] + csv_line

        update_result_data(cache, ind, fit, result_data, results, meta_score)

        if csv_line:
            csv_lines.append([time.ctime(), save_name_base] + csv_line)

            if halloffame is not None:
                onFront, significant = pareto.updateParetoFront(halloffame, ind, cache)
                if onFront and not cache.metaResultsOnly:
                    processResults(save_name_base, ind, cache, results)

            onFrontMeta, significant_meta = pareto.updateParetoFront(
                meta_hof, ind_meta, cache
            )
            if onFrontMeta:
                meta_csv_lines.append([time.ctime(), save_name_base] + csv_line)
                processResultsMeta(save_name_base, ind, cache, results)

            # If this is None progress can never be made and the algorithm will terminate, it should only be
            # None if used with algorithms like multistart or scoretest which don't need progress
            if progress_hof is not None:
                onFrontProgress, significant_progress = pareto.updateParetoFront(
                    copy.deepcopy(progress_hof), ind_meta, cache
                )
                if significant_progress:
                    made_progress = True
                    pareto.updateParetoFront(progress_hof, ind_meta, cache)

    writer.writerows(csv_lines)

    new_best_min = max([i.fitness.values[2] for i in meta_hof.items])

    # if the min value is zero then use the old method to determine progress of adding to the pareto front
    # this should catch later stages where real progress is not being made but you have a variable that is not identifiable
    # if new_best_min > 0:
    #    if new_best_min > best_min:
    #        made_progress = True
    #    else:
    #        made_progress = False

    # flush before returning
    csvfile.flush()

    if made_progress:
        if generation != cache.lastProgressGeneration:
            cache.generationsOfProgress += 1
            cache.lastProgressGeneration = generation
    elif generation != cache.lastProgressGeneration:
        cache.generationsOfProgress = 0

    path_meta_csv = cache.settings["resultsDirMeta"] / "results.csv"
    with path_meta_csv.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)
        writer.writerows(meta_csv_lines)

    cleanupFront(cache, halloffame, meta_hof)
    writeMetaFront(cache, meta_hof, path_meta_csv)

    stalled = (generation - cache.lastProgressGeneration) >= cache.stallGenerations
    stallWarn = (generation - cache.lastProgressGeneration) >= cache.stallCorrect
    progressWarn = cache.generationsOfProgress >= cache.progressCorrect

    return stalled, stallWarn, progressWarn


def eval_population(
    toolbox,
    cache,
    invalid_ind,
    writer,
    csvfile,
    halloffame,
    meta_hof,
    progress_hof,
    generation,
    result_data=None,
):
    return eval_population_base(
        toolbox.evaluate,
        toolbox,
        cache,
        invalid_ind,
        writer,
        csvfile,
        halloffame,
        meta_hof,
        progress_hof,
        generation,
        result_data,
    )


def eval_population_final(
    toolbox,
    cache,
    invalid_ind,
    writer,
    csvfile,
    halloffame,
    meta_hof,
    progress_hof,
    generation,
    result_data=None,
):
    return eval_population_base(
        toolbox.evaluate_final,
        toolbox,
        cache,
        invalid_ind,
        writer,
        csvfile,
        halloffame,
        meta_hof,
        progress_hof,
        generation,
        result_data,
    )


def eval_population_base(
    evaluate,
    toolbox,
    cache,
    invalid_ind,
    writer,
    csvfile,
    halloffame,
    meta_hof,
    progress_hof,
    generation,
    result_data=None,
):
    fitnesses = toolbox.map(evaluate, map(list, invalid_ind))

    return process_population(
        toolbox,
        cache,
        invalid_ind,
        fitnesses,
        writer,
        csvfile,
        halloffame,
        meta_hof,
        progress_hof,
        generation,
        result_data,
    )


def get_grad_tolerance(cache, name):
    smallest_peak = cache.target[name]["smallest_peak"]
    largest_peak = cache.target[name]["largest_peak"]

    grad_tol = cache.abstolFactorGrad * smallest_peak
    grad_tol_min = cache.abstolFactorGradMax * largest_peak

    return max([grad_tol, grad_tol_min])


def get_evo_tolerance(cache, name):
    smallest_peak = cache.target[name]["smallest_peak"]
    largest_peak = cache.target[name]["largest_peak"]

    evo_tol = cache.abstolFactor * smallest_peak
    evo_tol_min = cache.abstolFactorGradMax * largest_peak

    return max([evo_tol, evo_tol_min])


def writeMetaFront(cache, meta_hof, path_meta_csv):
    new_data = [individual.csv_line for individual in meta_hof.items]
    new_data = pandas.DataFrame(new_data, columns=cache.headers)

    new_data.to_csv(path_meta_csv, quoting=csv.QUOTE_ALL, index=False)
    new_data.to_excel(cache.settings["resultsDirMeta"] / "results.xlsx", index=False)


def processResultsGrad(save_name_base, individual, cache, results):
    saveExperiments(
        save_name_base,
        cache.settings,
        cache.target,
        results,
        cache.settings["resultsDirGrad"],
        "%s_%s_GRAD.h5",
    )


def processResults(save_name_base, individual, cache, results):
    saveExperiments(
        save_name_base,
        cache.settings,
        cache.target,
        results,
        cache.settings["resultsDirEvo"],
        "%s_%s_EVO.h5",
    )


def processResultsMeta(save_name_base, individual, cache, results):
    saveExperiments(
        save_name_base,
        cache.settings,
        cache.target,
        results,
        cache.settings["resultsDirMeta"],
        "%s_%s_meta.h5",
    )


def cleanupFront(cache, halloffame=None, meta_hof=None, grad_hof=None):
    if halloffame is not None:
        cleanDir(Path(cache.settings["resultsDirEvo"]), halloffame)

    if meta_hof is not None:
        cleanDir(Path(cache.settings["resultsDirMeta"]), meta_hof)

    if grad_hof is not None:
        cleanDir(Path(cache.settings["resultsDirGrad"]), grad_hof)


def cleanDir(dir, hof):
    # find all items in directory
    paths = dir.glob("*.h5")

    # make set of items based on removing everything after _
    exists = {str(path.name).split("_", 1)[0] for path in paths}

    # make set of allowed keys based on hall of hame
    allowed = {
        hashlib.md5(str(list(individual)).encode("utf-8", "ignore")).hexdigest()
        for individual in hof.items
    }

    # remove everything not in hall of fame
    remove = exists - allowed

    for save_name_base in remove:
        for path in dir.glob("%s*" % save_name_base):
            path.unlink()


def finish(cache):
    sub.graph_process(cache, "Last", last=True)
    sub.graph_spearman(cache)


def find_outliers(data, lower_percent=10, upper_percent=90):
    lb, ub = numpy.percentile(data, [lower_percent, upper_percent], 0)
    selected = (data >= lb) & (data <= ub)
    bools = numpy.all(selected, 1)
    return selected, bools


def get_confidence(data, lb=5, ub=95):
    lb, ub = numpy.percentile(data, [lb, ub], 0)
    selected = (data >= lb) & (data <= ub)
    bools = numpy.all(selected, 1)
    return data[bools, :]


def test_eta(eta, xl, xu, size):
    "return delta_q log10 power for each eta to use with distribution"
    x = (xu - xl) / 2.0

    delta_1 = (x - xl) / (xu - xl)
    delta_2 = (xu - x) / (xu - xl)

    rand = numpy.random.rand(size)

    mut_pow = 1.0 / (eta + 1.0)

    xy = numpy.zeros(size)
    val = numpy.zeros(size)
    delta_q = numpy.zeros(size)

    xy[rand < 0.5] = 1.0 - delta_1
    val[rand < 0.5] = 2.0 * rand[rand < 0.5] + (1.0 - 2.0 * rand[rand < 0.5]) * xy[
        rand < 0.5
    ] ** (eta + 1)
    delta_q[rand < 0.5] = val[rand < 0.5] ** mut_pow - 1.0

    xy[rand >= 0.5] = 1.0 - delta_2
    val[rand >= 0.5] = 2.0 * (1.0 - rand[rand >= 0.5]) + 2.0 * (
        rand[rand >= 0.5] - 0.5
    ) * xy[rand >= 0.5] ** (eta + 1)
    delta_q[rand >= 0.5] = 1.0 - val[rand >= 0.5] ** mut_pow

    delta_q = delta_q * (xu - xl)

    return delta_q


def confidence_eta(eta, xl, xu):
    seq = test_eta(eta, xl, xu, 100000)
    confidence = numpy.max(numpy.abs(numpy.percentile(seq, [5, 95])))
    mean = numpy.mean(numpy.abs(seq))

    return mean, confidence


def setupSimulation(sim, times, name, cache):
    "set the user solution times to match the times vector and adjust other sim parameters to required settings"

    try:
        del sim.root.input.solver.user_solution_times
    except KeyError:
        pass

    try:
        del sim.root.output
    except KeyError:
        pass

    sim.root.input.solver.user_solution_times = times
    sim.root.input.solver.sections.section_times[-1] = times[-1]
    sim.root.input["return"].split_components_data = 1

    if cache.dynamicTolerance:
        sim.root.input.solver.time_integrator.abstol = get_evo_tolerance(cache, name)
        sim.root.input.solver.time_integrator.reltol = 0.0

    multiprocessing.get_logger().info(
        "%s abstol=%.3g  reltol=%.3g",
        sim.filename,
        sim.root.input.solver.time_integrator.abstol,
        sim.root.input.solver.time_integrator.reltol,
    )

    experiment_name = sim.root.experiment_name
    if isinstance(experiment_name, bytes):
        experiment_name = experiment_name.decode()
    units_used = cache.target[experiment_name]["units_used"]
    for unit in sim.root.input.model.keys():
        if "unit_" in unit:
            sim.root.input["return"][unit].write_solution_particle = 0
            sim.root.input["return"][unit].write_solution_solid = 0
            sim.root.input["return"][unit].write_solution_column_inlet = 0
            sim.root.input["return"][unit].write_solution_inlet = 0

            sim.root.input["return"][unit].write_sens_bulk = 0
            sim.root.input["return"][unit].write_sens_inlet = 0
            sim.root.input["return"][unit].write_sens_particle = 0
            sim.root.input["return"][unit].write_sens_solid = 0
            sim.root.input["return"][unit].write_sens_flux = 0
            sim.root.input["return"][unit].write_sens_volume = 0

            sim.root.input["return"][unit].write_sensdot_bulk = 0
            sim.root.input["return"][unit].write_sensdot_inlet = 0
            sim.root.input["return"][unit].write_sensdot_outlet = 0
            sim.root.input["return"][unit].write_sensdot_particle = 0
            sim.root.input["return"][unit].write_sensdot_solid = 0
            sim.root.input["return"][unit].write_sensdot_flux = 0
            sim.root.input["return"][unit].write_sensdot_volume = 0

            if unit in units_used:
                sim.root.input["return"][unit].write_solution_column_outlet = 1
                sim.root.input["return"][unit].write_solution_outlet = 1
                sim.root.input["return"][unit].write_sens_outlet = 1
            else:
                sim.root.input["return"][unit].write_solution_column_outlet = 0
                sim.root.input["return"][unit].write_solution_outlet = 0
                sim.root.input["return"][unit].write_sens_outlet = 0
    sim.root.input.solver.nthreads = 1


def biasSimulation(simulation, experiment, cache):

    bias_simulation = Cadet(simulation.root)

    if cache.errorBias:
        name = experiment["name"]

        error_model = None
        for error in cache.settings["kde_synthetic"]:
            if error["name"] == name:
                error_model = error
                break

        delay_settings = error_model["delay"]
        flow_settings = error_model["flow"]
        load_settings = error_model["load"]

        nsec = bias_simulation.root.input.solver.sections.nsec

        delays = numpy.ones(nsec) * sum(delay_settings) / 2.0

        synthetic_error.pump_delay(bias_simulation, delays)

        flow = numpy.ones(nsec) * flow_settings[0]

        synthetic_error.error_flow(bias_simulation, flow)

        load = numpy.ones(nsec) * load_settings[0]

        synthetic_error.error_load(bias_simulation, load)

    return bias_simulation


def setupLog(log_directory, log_name):
    logger = multiprocessing.get_logger()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logFormatter = logging.Formatter(
        "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(message)s"
    )

    if not logger.handlers:
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(logging.INFO)
        stream.setFormatter(logFormatter)

        logger.addHandler(stream)

        sys.stdout = loggerwriter.LoggerWriter(logger.info)
        sys.stderr = loggerwriter.LoggerWriter(logger.warning)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_directory / log_name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logFormatter)

    # add the handlers to the logger
    logger.addHandler(fh)


def getCoreCounts():
    try:
        cpus = int(sys.argv[-1])
    except ValueError:
        # This happens when running in jupyter notebook or if the number can't be found, in either way default to just use availab cpu
        cpus = None
    if cpus:
        return cpus
    else:
        return multiprocessing.cpu_count()


def getMapFunction():
    cores = getCoreCounts()
    if cores == 1:

        multiprocessing.get_logger().info("CADETMatch startup: running single threaded")

        return map
    else:
        if "pool" not in getMapFunction.__dict__:
            pool = multiprocessing.Pool(cores)
            getMapFunction.pool = pool
            multiprocessing.get_logger().info(
                "CADETMatch startup: created a parallel pool of %s workers", cores
            )

        return getMapFunction.pool.imap_unordered


def create_lookup(seq):
    temp = {}
    for i in seq:
        key = tuple(i)
        if key not in temp:
            temp[key] = [
                i,
            ]
        else:
            temp[key].append(i)
    return temp


def pop_lookup(lookup, key):
    temp = lookup[tuple(key)]
    data = temp.pop(0)
    if not temp:
        del lookup[tuple(key)]
    return data


def process_fraction_csv(csv_file):
    data = pandas.read_csv(csv_file)
    rows, cols = data.shape

    data_headers = data.columns.values.tolist()

    start_times = numpy.array(data.iloc[:, 0])
    stop_times = numpy.array(data.iloc[:, 1])
    components = [int(i) for i in data_headers[2:]]
    fractions = data.iloc[:, 2:]

    return start_times, stop_times, components, fractions


def fractionate_sim(start_times, stop_times, components, simulation, unit):
    times = simulation.root.output.solution.solution_times

    fracs = {}
    for component in components:
        sim_value = simulation.root.output.solution[unit][
            "solution_outlet_comp_%03d" % component
        ]
        spline = scipy.interpolate.InterpolatedUnivariateSpline(times, sim_value, ext=1)

        fracs[component] = fractionate_spline(start_times, stop_times, spline)

    return fracs


def translate_meta_min(score, cache):
    temp = numpy.array(score)
    if cache.allScoreSSE:
        temp[:3] = -temp[:3]
    else:
        temp[:3] = 1 - temp[:3]

    return temp
