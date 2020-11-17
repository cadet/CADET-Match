import multiprocessing
from pathlib import Path

import numpy
from cadet import Cadet
from deap import creator, tools

import CADETMatch.cache as cache
import CADETMatch.progress as progress
import CADETMatch.score_calc as score_calc
import CADETMatch.util as util

ERROR = {
    "scores": None,
    "path": None,
    "simulation": None,
    "error": None,
    "cadetValues": None,
    "cadetValuesKEQ": None,
}


def fitness_final(individual, json_path, run_experiment=None):
    return fitness_base(
        runExperimentFinal, "simulation_final", individual, json_path, run_experiment
    )


def fitness(individual, json_path, run_experiment=None):
    return fitness_base(
        runExperiment, "simulation", individual, json_path, run_experiment
    )


def meta_score_trans(cache, score):
    temp = numpy.copy(score)

    if cache.allScoreSSE:
        temp[:3] = -temp[:3]
    else:
        temp[:3] = 1 - temp[:3]

    return temp


def fitness_base(runner, template_name, individual, json_path, run_experiment):
    if json_path != cache.cache.json_path:
        cache.cache.setup_dir(json_path)
        util.setupLog(cache.cache.settings["resultsDirLog"], "main.log")
        cache.cache.setup(json_path)

    if run_experiment is None:
        run_experiment = runExperiment

    scores = []
    error = 0.0
    exp_values = []
    sim_values = []

    results = {}
    for experiment in cache.cache.settings["experiments"]:
        result = runner(
            individual,
            template_name,
            experiment,
            cache.cache.settings,
            cache.cache.target,
            cache.cache,
        )
        if result is not None:
            results[experiment["name"]] = result
            scores.extend(results[experiment["name"]]["scores"])
            error += results[experiment["name"]]["error"]

            sim_values.extend(result["sim_value"])
            exp_values.extend(result["exp_value"])
        else:
            return cache.cache.WORST, [], cache.cache.WORST_META, None, individual

    rmse = score_calc.rmse_combine(exp_values, sim_values)

    if numpy.any(numpy.isnan(scores)):
        multiprocessing.get_logger().info("NaN found for %s %s", individual, scores)

    # human scores
    meta_score = numpy.concatenate(
        [util.calcMetaScores(scores, cache.cache), [error, rmse]]
    )

    for result in results.values():
        if result["cadetValuesKEQ"]:
            cadetValuesKEQ = result["cadetValuesKEQ"]
            break

    # generate csv
    csv_record = []
    csv_record.extend(["EVO", "NA"])
    csv_record.extend(cadetValuesKEQ)
    csv_record.extend(progress.score_trans(cache.cache, scores))
    csv_record.extend(meta_score_trans(cache.cache, meta_score))

    return scores, csv_record, meta_score, results, tuple(individual)


def saveExperiments(save_name_base, settings, target, results):
    return util.saveExperiments(
        save_name_base,
        settings,
        target,
        results,
        settings["resultsDirEvo"],
        "%s_%s_EVO.h5",
    )


def plotExperiments(save_name_base, settings, target, results):
    util.plotExperiments(
        save_name_base,
        settings,
        target,
        results,
        settings["resultsDirEvo"],
        "%s_%s_EVO.png",
    )


def runExperimentFinal(individual, template_name, experiment, settings, target, cache):
    sim_name = "template_%s_final.h5" % experiment["name"]
    return runExperimentBase(
        sim_name, template_name, individual, experiment, settings, target, cache
    )


def runExperiment(individual, template_name, experiment, settings, target, cache):
    sim_name = "template_%s.h5" % experiment["name"]
    return runExperimentBase(
        sim_name, template_name, individual, experiment, settings, target, cache
    )


def runExperimentBase(
    sim_name, template_name, individual, experiment, settings, target, cache
):
    if template_name not in experiment:
        templatePath = Path(settings["resultsDirMisc"], sim_name)
        templateSim = Cadet()
        templateSim.filename = templatePath.as_posix()
        templateSim.load()
        experiment[template_name] = templateSim

    return util.runExperiment(
        individual,
        experiment,
        settings,
        target,
        experiment[template_name],
        experiment[template_name].root.timeout,
        cache,
    )


def run(cache):
    "run the parameter estimation"
    searchMethod = cache.settings.get("searchMethod", "NSGA3")
    return cache.search[searchMethod].run(cache, tools, creator)
