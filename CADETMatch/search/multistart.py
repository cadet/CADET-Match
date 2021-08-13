import array
import csv
import multiprocessing
import random
import time
from pathlib import Path

import CADETMatch.pareto as pareto
import CADETMatch.progress as progress
import CADETMatch.sub as sub
import CADETMatch.util as util
import CADETMatch.pop as pop

name = "Multistart"


def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()

    parameters = len(cache.MIN_VALUE)

    LAMBDA = parameters * cache.settings["population"]
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

    pop = cache.toolbox.population(n=LAMBDA)

    if "seeds" in cache.settings:
        seed_pop = [
            cache.toolbox.individual_guess(
                [f(v) for f, v in zip(cache.settings["transform"], sublist)]
            )
            for sublist in cache.settings["seeds"]
        ]
        pop.extend(seed_pop)

    gradCheck = cache.badScore

    if cache.metaResultsOnly:
        hof = pareto.DummyFront()
    else:
        hof = pareto.ParetoFront(dimensions=len(cache.WORST),
            similar=pareto.similar, similar_fit=pareto.similar_fit(cache)
        )
    meta_hof = pareto.ParetoFront(dimensions=len(cache.WORST_META),
        similar=pareto.similar,
        similar_fit=pareto.similar_fit_meta(cache),
        slice_object=cache.meta_slice,
    )
    grad_hof = pareto.ParetoFront(dimensions=len(cache.WORST),
        similar=pareto.similar, similar_fit=pareto.similar_fit(cache)
    )
    progress_hof = pareto.DummyFront()

    path = Path(cache.settings["resultsDirBase"], cache.settings["csv"])
    with path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)

        multiprocessing.get_logger().info(
            "Population %s", [util.convert_individual_inputorder(i, cache) for i in pop]
        )

        gradCheck, newChildren = cache.eval.grad_search(
            gradCheck, pop, cache, writer, csvfile, hof, meta_hof, -1, check_all=True
        )

        stalled, stallWarn, progressWarn = util.eval_population(
            cache,
            newChildren,
            writer,
            csvfile,
            hof,
            meta_hof,
            None,
            -1,
            result_data,
        )

        progress.writeProgress(
            cache,
            -1,
            newChildren,
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
    creator.create(
        "Individual",
        pop.Individual
    )

    creator.create(
        "IndividualMeta",
        pop.Individual
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

    if cache.sobolGeneration:
        cache.toolbox.register(
            "population", util.sobolGenerator, creator.Individual, cache
        )
    else:
        cache.toolbox.register(
            "population", tools.initRepeat, list, cache.toolbox.individual
        )
    cache.toolbox.register(
        "randomPopulation", tools.initRepeat, list, cache.toolbox.individual
    )

    cache.toolbox.register(
        "individual_guess", util.initIndividual, creator.Individual, cache
    )
