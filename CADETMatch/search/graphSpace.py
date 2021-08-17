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
import numpy

name = "GraphSpace"

"This is not actually an optimization method. It is used to sample a space and generate graphs to see what the space looks like."


def run(cache):
    "run the parameter estimation"
    random.seed()

    parameters = len(cache.MIN_VALUE)

    populationSize = parameters * cache.settings["population"]
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

    init_pop = util.sobolPopulation(populationSize, parameters, numpy.array(cache.MIN_VALUE), numpy.array(cache.MAX_VALUE))
    init_pop = [pop.Individual(row) for row in init_pop]

    if "seeds" in cache.settings:
        seed_pop = [
            pop.Individual(
                [f(v) for f, v in zip(cache.settings["transform"], sublist)]
            )
            for sublist in cache.settings["seeds"]
        ]
        init_pop.extend(seed_pop)

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

        multiprocessing.get_logger().info("Population %s", init_pop)

        stalled, stallWarn, progressWarn = util.eval_population(
            cache,
            init_pop,
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
            init_pop,
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

