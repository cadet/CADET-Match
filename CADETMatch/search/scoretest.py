import array
import csv
import multiprocessing
import time
from pathlib import Path

import CADETMatch.jacobian as jacobian
import CADETMatch.pareto as pareto
import CADETMatch.progress as progress
import CADETMatch.sub as sub
import CADETMatch.util as util
import CADETMatch.pop as pop

name = "ScoreTest"


def run(cache, tools, creator):
    "run the parameter estimation"
    path = Path(cache.settings["resultsDirBase"], cache.settings["csv"])
    with path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)
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

        population = []
        if "seeds" in cache.settings:
            seed_pop = [
                pop.Individual(
                    [f(v) for f, v in zip(cache.settings["transform"], sublist)]
                )
                for sublist in cache.settings["seeds"]
            ]
            population.extend(seed_pop)

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

        stalled, stallWarn, progressWarn = util.eval_population(
            cache,
            population,
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
            population,
            hof,
            meta_hof,
            grad_hof,
            progress_hof,
            sim_start,
            generation_start,
            result_data,
        )

        if cache.settings.get("condTest", None):
            for ind in population:
                J = jacobian.jac(ind, cache)
                multiprocessing.get_logger().info("%s %s", ind, J)

        population = [pop.Individual(i) for i in meta_hof]
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
