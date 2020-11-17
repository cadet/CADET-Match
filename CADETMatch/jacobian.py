import multiprocessing
from pathlib import Path

import numpy
from scipy.optimize._numdiff import approx_derivative

import CADETMatch.util as util
from CADETMatch.cache import cache


class GradientException(Exception):
    pass


def jac(individual, cache):
    J = approx_derivative(
        fitness_grad, individual, method="3-point", kwargs={"cache": cache}
    )
    cond = numpy.linalg.cond(J)
    u, s, v = numpy.linalg.svd(J)
    minSing = numpy.min(s)
    maxSing = numpy.max(s)
    multiprocessing.get_logger().info(
        "%s has condition %s  min sing:  %s   max sing: %s",
        individual,
        cond,
        minSing,
        maxSing,
    )
    return J


def fitness_grad(individual, cache):
    scores = []
    error = 0.0

    results = {}
    for experiment in cache.settings["experiments"]:
        result = runExperiment(
            individual, experiment, cache.settings, cache.target, cache
        )
        if result is not None:
            results[experiment["name"]] = result
            scores.extend(results[experiment["name"]]["scores"])
            error += results[experiment["name"]]["error"]
        else:
            raise GradientException("Gradient caused simulation failure, aborting")

    return scores


def runExperiment(individual, experiment, settings, target, cache):
    if "simulation" not in experiment:
        templatePath = Path(
            settings["resultsDirMisc"], "template_%s.h5" % experiment["name"]
        )
        templateSim = Cadet()
        templateSim.filename = templatePath.as_posix()
        templateSim.load()
        experiment["simulation"] = templateSim

    return util.runExperiment(
        individual,
        experiment,
        settings,
        target,
        experiment["simulation"],
        experiment["simulation"].root.timeout,
        cache,
    )
