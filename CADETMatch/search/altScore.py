import array
import csv
import multiprocessing
import time
from pathlib import Path

import numpy
from cadet import H5

import CADETMatch.jacobian as jacobian
import CADETMatch.pareto as pareto
import CADETMatch.progress as progress
import CADETMatch.sub as sub
import CADETMatch.util as util
import CADETMatch.pop as pop

name = "AltScore"


def run(cache, tools, creator):
    "run the parameter estimation"
    path = Path(cache.settings["resultsDirBase"], cache.settings["csv"])
    with path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)
        pop = cache.toolbox.population(n=0)
        sim_start = generation_start = time.time()
        result_data = {
            "input": [],
            "output": [],
            "output_meta": [],
            "results": {},
            "times": {},
            "input_transform": [],
            "input_transform_extended": [],
            "strategy": [],
            "mean": [],
            "confidence": [],
        }

        previousResults = cache.settings["PreviousResults"]
        data = H5()
        data.filename = previousResults
        data.load(paths=["/meta_population", "/meta_population_transform"], lock=True)

        meta_population = data.root.meta_population
        meta_population_transform = data.root.meta_population_transform

        for line in meta_population:
            pop.append(cache.toolbox.individual_guess(line))

        multiprocessing.get_logger().info(
            "altScore starting population %s %s", len(pop), pop
        )
        multiprocessing.get_logger().info(
            "altScore starting population (transform) %s %s",
            len(meta_population_transform),
            meta_population_transform,
        )

        convert = util.convert_population_inputorder(numpy.array(pop), cache)

        multiprocessing.get_logger().info(
            "altScore starting population (convert) %s %s", len(convert), convert
        )

        if cache.metaResultsOnly:
            hof = pareto.DummyFront()
        else:
            hof = pareto.ParetoFront(dimensions=len(cache.WORST),
                similar=pareto.similar, similar_fit=pareto.similar_fit(cache)
            )
        meta_hof = pareto.ParetoFront(dimensions=len(cache.WORST_META),
            similar=pareto.similar, similar_fit=pareto.similar_fit_meta(cache)
        )
        grad_hof = pareto.ParetoFront(dimensions=len(cache.WORST), 
            similar=pareto.similar, similar_fit=pareto.similar_fit(cache)
        )
        progress_hof = pareto.ParetoFront(dimensions=len(cache.WORST_META),
            similar=pareto.similar, similar_fit=pareto.similar_fit_meta(cache)
        )

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        stalled, stallWarn, progressWarn = util.eval_population(
            cache,
            invalid_ind,
            writer,
            csvfile,
            hof,
            meta_hof,
            progress_hof,
            -1,
            result_data,
        )
        progress.writeProgress(
            cache,
            -1,
            pop,
            hof,
            meta_hof,
            grad_hof,
            progress_hof,
            sim_start,
            generation_start,
            result_data,
        )

        if cache.settings.get("condTest", None):
            for ind in invalid_ind:
                J = jacobian.jac(ind, cache)
                multiprocessing.get_logger().info("%s %s", ind, J)

        population = [cache.toolbox.individual_guess(i) for i in meta_hof]
        stalled, stallWarn, progressWarn = util.eval_population_final(
            cache,
            population,
            writer,
            csvfile,
            hof,
            meta_hof,
            progress_hof,
            0,
            result_data,
        )
        progress.writeProgress(
            cache,
            0,
            population,
            hof,
            meta_hof,
            grad_hof,
            progress_hof,
            sim_start,
            generation_start,
            result_data,
        )

        util.finish(cache)
        sub.graph_corner_process(cache, last=True)
        return hof


def setupDEAP(
    cache,
    creator,
    base,
    tools,
):
    "setup the DEAP variables"
    creator.create("FitnessMin", base.Fitness, weights=[-1.0] * cache.numGoals)
    creator.create(
        "Individual",
        pop.Individual,
        fitness=creator.FitnessMin
    )

    creator.create("FitnessMinMeta", base.Fitness, weights=[-1.0, -1.0, -1.0, -1.0, -1.0])
    creator.create(
        "IndividualMeta",
        pop.Individual,
        fitness=creator.FitnessMinMeta
    )
    cache.toolbox.register(
        "individualMeta", util.initIndividual, creator.IndividualMeta, cache
    )

    cache.toolbox.register(
        "individual",
        util.generateIndividual,
        creator.Individual,
        len(cache.MIN_VALUE),
        cache.MIN_VALUE,
        cache.MAX_VALUE,
        cache,
    )
    cache.toolbox.register(
        "population", tools.initRepeat, list, cache.toolbox.individual
    )

    cache.toolbox.register(
        "individual_guess", util.initIndividual, creator.Individual, cache
    )