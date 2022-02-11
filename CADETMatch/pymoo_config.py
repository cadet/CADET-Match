import csv
import time
import numpy
import random
import pickle
from pathlib import Path

import CADETMatch.pareto as pareto
import CADETMatch.progress as progress
import CADETMatch.sub as sub
import CADETMatch.util as util
import CADETMatch.pop as pop

from pymoo.factory import get_algorithm, get_reference_directions
from pymoo.core.problem import Problem
import attr


@attr.s
class ProblemState:
    cache = attr.ib()
    writer = attr.ib()
    csvfile = attr.ib()
    halloffame = attr.ib()
    meta_hof = attr.ib()
    progress_hof = attr.ib()
    result_data = attr.ib()


name = "pymoo"

class MyProblem(Problem):

    def __init__(self, n_var, n_obj, lb, ub, evaluate_function, problem_state):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=lb, xu=ub)
        self.evaluate_function = evaluate_function
        self.problem_state = problem_state

    def _evaluate(self, population, out, *args, **kwargs):
        population = [pop.Individual(row) for row in numpy.array(population)]

        stalled, stallWarn, progressWarn = util.eval_population(
                    self.problem_state.cache,
                    population,
                    self.problem_state.writer,
                    self.problem_state.csvfile,
                    self.problem_state.halloffame,
                    self.problem_state.meta_hof,
                    self.problem_state.progress_hof,
                    kwargs['algorithm'].n_gen,
                    self.problem_state.result_data,
                )

        self.stalled = stalled
        self.stallWarn = stallWarn
        self.progressWarn = progressWarn
        self.eval_population = population
        self.ngen = kwargs['algorithm'].n_gen

        error = numpy.array([ind.fitness.values for ind in population])
        out["F"] = error


def stop_iteration(best, stalled, cache):
    stopAverage = cache.settings.get("stopAverage", 0.0)
    stopBest = cache.settings.get("stopBest", 0.0)
    stopRMSE = cache.settings.get("stopRMSE", 0.0)
    if best[2] <= stopAverage or best[1] <= stopBest or stalled or best[-1] <= stopRMSE:
        return True
    else:
        return False


def run(cache, alg="unsga3"):
    "run the parameter estimation"
    random.seed()
    parameters = len(cache.MIN_VALUE)

    populationSize = parameters * cache.settings.get("population", 100)

    totalGenerations = parameters * cache.settings.get("generations", 1000)

    init_pop = util.sobolPopulation(populationSize, parameters, numpy.array(cache.MIN_VALUE), numpy.array(cache.MAX_VALUE))

    if "seeds" in cache.settings:
        seed_pop = [
            [f(v) for f, v in zip(cache.settings["transform"], sublist)]
            for sublist in cache.settings["seeds"]
        ]
        init_pop = numpy.concatenate([init_pop, numpy.array(seed_pop)], axis=0)

    
    path = Path(cache.settings["resultsDirBase"], cache.settings["csv"])
    with path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)

        checkpointFile = Path(
            cache.settings["resultsDirMisc"], cache.settings.get("checkpointFile", "check")
        )

        pymoo_checkpointFile = Path(
            cache.settings["resultsDirMisc"], "pymoo_check.npy"
        )

        if checkpointFile.exists():
            with checkpointFile.open("rb") as cp_file:
                cp = pickle.load(cp_file)
            hof = cp["halloffame"]
            meta_hof = cp["meta_halloffame"]
            grad_hof = cp["grad_halloffame"]
            progress_hof = cp["progress_halloffame"]
            cache.generationsOfProgress = cp["generationsOfProgress"]
            cache.lastProgressGeneration = cp["lastProgressGeneration"]

            if cp["gradCheck"] > cache.settings.get("gradCheck", 0.0):
                gradCheck = cp["gradCheck"]
            else:
                gradCheck = cache.settings.get("gradCheck", 0.0)

        else:
            if cache.metaResultsOnly:
                hof = pareto.DummyFront()
            else:
                hof = pareto.ParetoFront(dimensions=len(cache.WORST),
                    similar=pareto.similar, similar_fit=pareto.similar_fit(cache)
                )
            meta_hof = pareto.ParetoFront(dimensions=len(cache.WORST_META),
                similar=pareto.similar, similar_fit=pareto.similar_fit_meta(cache),
                slice_object=cache.meta_slice
            )
            grad_hof = pareto.ParetoFront(dimensions=len(cache.WORST),
                similar=pareto.similar, similar_fit=pareto.similar_fit(cache)
            )
            progress_hof = pareto.ParetoFront(dimensions=len(cache.WORST_META),
                similar=pareto.similar, similar_fit=pareto.similar_fit_meta(cache),
                slice_object=cache.meta_slice
            )

            gradCheck = cache.settings.get("gradCheck", 0.0)

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
        
        problem_state = ProblemState(cache=cache, writer=writer, csvfile=csvfile, halloffame=hof,
                                     meta_hof=meta_hof, progress_hof=progress_hof, result_data=result_data)

        problem = MyProblem(parameters, cache.numGoals, cache.MIN_VALUE, cache.MAX_VALUE, cache.eval.evaluate, problem_state)

        numRefDirs = min(populationSize, 1000)

        ref_dirs = get_reference_directions("energy", cache.numGoals, numRefDirs, seed=1)


        if pymoo_checkpointFile.exists():
            algorithm, = numpy.load(pymoo_checkpointFile, allow_pickle=True).flatten()
            algorithm.problem = problem

            algorithm.n_offsprings = populationSize
            algorithm.pop_size = populationSize
        else:
            algorithm = get_algorithm(alg, ref_dirs=ref_dirs, sampling=init_pop, pop_size=populationSize )
            algorithm.setup(problem, termination=('n_gen', totalGenerations), seed=1)

        while algorithm.has_next():
            algorithm.next()

            stalled = problem.stalled
            stallWarn = problem.stallWarn
            progressWarn = problem.progressWarn
            population = problem.eval_population

            progress.writeProgress(
                cache,
                problem.ngen,
                population,
                hof,
                meta_hof,
                grad_hof,
                progress_hof,
                sim_start,
                generation_start,
                problem.problem_state.result_data,
            )

            sub.graph_process(cache, problem.ngen)
            sub.graph_corner_process(cache, last=False)

            best = meta_hof.getBestScores()
            cp = dict(
                population=population,
                halloffame=hof,
                rndstate=random.getstate(),
                meta_halloffame=meta_hof,
                grad_halloffame=grad_hof,
                gradCheck=gradCheck,
                generationsOfProgress=cache.generationsOfProgress,
                lastProgressGeneration=cache.lastProgressGeneration,
                progress_halloffame=progress_hof,
            )

            with checkpointFile.open("wb") as cp_file:
                pickle.dump(cp, cp_file)

            #can't pickle the pool and other things so all of that is just stored in problem_state
            #just temporarily unhook it from the problem and then add it back
            del algorithm.problem.problem_state
            numpy.save(pymoo_checkpointFile, algorithm)
            algorithm.problem.problem_state = problem_state

            if stop_iteration(best, stalled, cache):
                break

        gen = problem.ngen + 1
        if cache.finalGradRefinement:            
            best_individuals = [pop.Individual(i) for i in meta_hof]
            gradCheck, newChildren = cache.eval.grad_search(
                gradCheck,
                best_individuals,
                cache,
                writer,
                csvfile,
                grad_hof,
                meta_hof,
                gen,
                check_all=True,
                result_data=result_data,
            )
            if newChildren:
                progress.writeProgress(
                    cache,
                    gen,
                    newChildren,
                    hof,
                    meta_hof,
                    grad_hof,
                    progress_hof,
                    sim_start,
                    generation_start,
                    result_data,
                )

        population = [pop.Individual(i) for i in meta_hof]
        stalled, stallWarn, progressWarn = util.eval_population_final(
            cache,
            population,
            writer,
            csvfile,
            hof,
            meta_hof,
            progress_hof,
            gen + 1,
            result_data,
        )
        progress.writeProgress(
            cache,
            gen + 1,
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
