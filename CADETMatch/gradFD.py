import csv
import hashlib
import multiprocessing
import time
from pathlib import Path

import numpy
import numpy.linalg
import scipy.optimize
from cadet import H5, Cadet

import CADETMatch.cache as cache
import CADETMatch.evo as evo
import CADETMatch.pareto as pareto
import CADETMatch.util as util
import CADETMatch.pop as pop


class GradientException(Exception):
    pass


def create_template(
    experiment, name, cache, settings, template_name, store_name
):
    simulationGrad = Cadet(experiment["simulation"].root)
    template_path_grad = Path(settings["resultsDirMisc"], template_name % name)
    simulationGrad.filename = template_path_grad.as_posix()

    if cache.dynamicTolerance:
        simulationGrad.root.input.solver.time_integrator.abstol = util.get_grad_tolerance(cache, name)
        simulationGrad.root.input.solver.time_integrator.reltol = 0.0

    start = time.time()
    util.runExperiment(
        None,
        experiment,
        cache.settings,
        cache.target,
        simulationGrad,
        experiment.get("timeout", 1800),
        cache,
    )
    elapsed = time.time() - start

    # timeout needs to be stored in the template so all processes have it without calculating it
    simulationGrad.root.timeout = max(10, elapsed * 50)
    simulationGrad.save()

    multiprocessing.get_logger().info("grad simulation took %s", elapsed)

    multiprocessing.get_logger().info(
        "grad %s abstol=%.3g  reltol=%.3g",
        simulationGrad.filename,
        simulationGrad.root.input.solver.time_integrator.abstol,
        simulationGrad.root.input.solver.time_integrator.reltol,
    )

    experiment[store_name] = simulationGrad


def setupTemplates(cache):
    settings = cache.settings
    target = cache.target

    for experiment in settings["experiments"]:
        name = experiment["name"]

        create_template(
            experiment,
            name,
            cache,
            settings,
            "template_%s_grad.h5",
            "simulationSens",
        )


def grad_score(values, cache):
    values = numpy.array(values)

    prod = util.product_score(values)

    return prod


def search(
    gradCheck,
    offspring,
    cache,
    writer,
    csvfile,
    grad_hof,
    meta_hof,
    generation,
    check_all=False,
    result_data=None,
    filterOverlap=True,
):
    if check_all:
        checkOffspring = offspring
    else:
        checkOffspring = (
            ind
            for ind in offspring
            if grad_score(ind.fitness.values, cache) <= gradCheck
        )
    if filterOverlap:
        checkOffspring = filterOverlapArea(cache, checkOffspring)
    newOffspring = cache.map_function(
        cache.eval.evaluate_grad, map(tuple, checkOffspring)
    )

    temp = []
    csv_lines = []
    meta_csv_lines = []

    gradient_results = []

    multiprocessing.get_logger().info("starting coarse refine")
    new_results = processOffspring(
        newOffspring,
        temp,
        csv_lines,
        meta_csv_lines,
        gradient_results,
        grad_hof,
        meta_hof,
        generation,
        result_data,
        cache,
    )
    multiprocessing.get_logger().info("ending coarse refine")

    if new_results:
        multiprocessing.get_logger().info("starting fine refine")
        fineOffspring = cache.map_function(
            cache.eval.evaluate_grad_fine, map(tuple, meta_hof)
        )
        processOffspring(
            fineOffspring,
            temp,
            csv_lines,
            meta_csv_lines,
            gradient_results,
            grad_hof,
            meta_hof,
            generation,
            result_data,
            cache,
        )
        multiprocessing.get_logger().info("ending fine refine")

    if temp:
        best = meta_hof.getBestScores()
        prod = best[0]
        if prod < gradCheck:
            gradCheck = prod
            multiprocessing.get_logger().info("updating gradCheck %s", gradCheck)

    writer.writerows(csv_lines)

    # flush before returning
    csvfile.flush()

    path_meta_csv = cache.settings["resultsDirMeta"] / "results.csv"
    with path_meta_csv.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)
        writer.writerows(meta_csv_lines)

    if gradient_results:
        write_gradient_results(gradient_results, cache)

    util.cleanupFront(cache, None, meta_hof, grad_hof)
    util.writeMetaFront(cache, meta_hof, path_meta_csv)

    return gradCheck, temp


def processOffspring(
    offspring,
    temp,
    csv_lines,
    meta_csv_lines,
    gradient_results,
    grad_hof,
    meta_hof,
    generation,
    result_data,
    cache,
):
    new_meta = []
    for i in offspring:
        if i is None:
            pass
        elif i.x is not None:
            gradient_results.append(i)

            ind = pop.Individual(i.x)

            (
                fit,
                csv_line,
                meta_score,
                results,
                individual,
            ) = cache.eval.evaluate_final(ind, run_experiment=runExperimentSens)

            ind.fitness.values = fit

            csv_line[0] = "GRAD"

            try:
                csv_line[1] = numpy.linalg.cond(i.jac)
            except (AttributeError, numpy.linalg.LinAlgError):
                csv_line[1] = ""

            save_name_base = hashlib.md5(
                str(list(ind)).encode("utf-8", "ignore")
            ).hexdigest()
            ind.csv_line = [time.ctime(), save_name_base] + csv_line

            ind_meta = pop.Individual(ind)
            ind_meta.fitness.values = meta_score
            ind_meta.csv_line = [time.ctime(), save_name_base] + csv_line

            util.update_result_data(cache, ind, fit, result_data, results, meta_score)

            if csv_line:
                csv_lines.append([time.ctime(), save_name_base] + csv_line)
                onFront, significant = pareto.updateParetoFront(grad_hof, ind)
                if onFront and not cache.metaResultsOnly:
                    util.processResultsGrad(save_name_base, ind, cache, results)

                onFrontMeta, significant = pareto.updateParetoFront(meta_hof, ind_meta)
                new_meta.append(onFrontMeta)
                if onFrontMeta:
                    meta_csv_lines.append([time.ctime(), save_name_base] + csv_line)
                    util.processResultsMeta(save_name_base, ind, cache, results)

                    if significant:
                        cache.lastProgressGeneration = generation

            temp.append(ind)
    return any(new_meta)


def write_gradient_results(gradient_results, cache):
    grad_path = Path(cache.settings["resultsDir"]) / "grad_result.h5"
    results = H5()
    results.filename = grad_path.as_posix()

    if grad_path.exists():
        results.load(lock=True)

    for i in gradient_results:
        grad_key = " ".join(["%.16f" % val for val in i.cadetValuesInput])

        for key, value in i.items():
            results[grad_key][key] = value
    results.save(lock=True)


def filterOverlapArea(cache, checkOffspring, cutoff=0.01):
    """if there is no overlap between the simulation and the data there is no gradient to follow and these entries need to be skipped
    This function also sorts from highest to lowest overlap and keeps the top multiStartPercent"""
    checkOffspring = list(checkOffspring)
    temp = cache.map_function(cache.eval.evaluate, map(list, checkOffspring))

    temp_offspring = []

    lookup = util.create_lookup(checkOffspring)

    for fit, csv_line, meta_score, results, individual in temp:
        ind = util.pop_lookup(lookup, individual)
        temp_area_total = 0.0
        temp_area_overlap = 0.0
        if results is not None:
            for exp in results.values():
                sim_times = exp["sim_time"]
                sim_values = exp["sim_value"]
                exp_values = exp["exp_value"]

                for sim_time, sim_value, exp_value in zip(
                    sim_times, sim_values, exp_values
                ):
                    if (
                        len(sim_time) < len(exp_value)
                        and len(exp_value) % len(sim_time) == 0
                    ):
                        # dealing with fractionation data, there are multiple data sets of len(sim_time)
                        exp_value = exp_value.reshape(-1, len(sim_time))
                        sim_value = sim_value.reshape(-1, len(sim_time))
                        for exp_row, sim_row in zip(exp_value, sim_value):
                            temp_area_total += numpy.trapz(exp_row, sim_time)
                            temp_area_overlap += numpy.trapz(
                                numpy.min([sim_row, exp_row], 0), sim_time
                            )
                    else:
                        temp_area_total += numpy.trapz(exp_value, sim_time)
                        temp_area_overlap += numpy.trapz(
                            numpy.min([sim_value, exp_value], 0), sim_time
                        )

            percent = temp_area_overlap / temp_area_total
            if percent > cutoff:
                temp_offspring.append((percent, ind))
                multiprocessing.get_logger().info(
                    "kept %s with overlap (%s) in gradient descent", ind, percent
                )
            else:
                multiprocessing.get_logger().info(
                    "removed %s for insufficient overlap (%s) in gradient descent",
                    ind,
                    percent,
                )
        else:
            multiprocessing.get_logger().info("removed %s for failure", ind)

    # sort in descending order, this has the best chance of converging so we can quick abort
    temp_offspring.sort(reverse=True)

    if cache.multiStartPercent < 1.0 and temp_offspring:
        # cut to the top multiStartPercent items with a minimum of 1 item
        temp_offspring = temp_offspring[
            : max(int(cache.multiStartPercent * len(checkOffspring)), 1)
        ]
        multiprocessing.get_logger().info(
            "gradient overlap cutoff %.2g", temp_offspring[-1][0]
        )

    temp_offspring = [ind for (percent, ind) in temp_offspring]

    if checkOffspring:
        multiprocessing.get_logger().info("overlap okay offspring %s", temp_offspring)

    return temp_offspring


def gradSearchFine(x, json_path):
    if json_path != cache.cache.json_path:
        cache.cache.setup_dir(json_path)
        util.setupLog(cache.cache.settings["resultsDirLog"], "main.log")
        cache.cache.setup(json_path)
    return refine(x, cache.cache.gradFineStop)


def gradSearch(x, json_path):
    if json_path != cache.cache.json_path:
        cache.cache.setup_dir(json_path)
        util.setupLog(cache.cache.settings["resultsDirLog"], "main.log")
        cache.cache.setup(json_path)
    return refine(x, 1e-4)


def refine(x, xtol):
    localRefine = cache.cache.settings.get("localRefine", "gradient")
    if localRefine == "gradient":
        try:
            x = numpy.clip(x, cache.cache.MIN_VALUE, cache.cache.MAX_VALUE)
            val = scipy.optimize.least_squares(
                fitness_sens_grad,
                x,
                jac="3-point",
                method="trf",
                bounds=(cache.cache.MIN_VALUE, cache.cache.MAX_VALUE),
                xtol=xtol,
                ftol=None,
                gtol=None,
                loss="linear",
                diff_step=1e-4,
            )

            cadetValues, cadetValuesExtended = util.convert_individual(
                val.x, cache.cache
            )
            cadetValuesInput, cadetValuesExtendedInput = util.convert_individual(
                x, cache.cache
            )

            val.cadetValues = cadetValues
            val.cadetValuesExtended = cadetValuesExtended
            val.cadetValuesInput = cadetValuesInput
            val.cadetValuesExtendedInput = cadetValuesExtendedInput
            val.cond = numpy.linalg.cond(val.jac)

            multiprocessing.get_logger().info(
                "gradient optimization result start: %s (%s) result: %s",
                x,
                cadetValues,
                val,
            )

            return val
        except GradientException:
            # If the gradient fails return None as the point so the optimizer can adapt
            return None

    if localRefine == "powell":

        def goal(x):
            diff = fitness_grad(x)
            return float(numpy.sum(diff ** 2.0))

        try:
            x = numpy.clip(x, cache.cache.MIN_VALUE, cache.cache.MAX_VALUE)
            val = scipy.optimize.minimize(
                goal,
                x,
                method="powell",
                options={"xtol": xtol},
                bounds=list(zip(cache.cache.MIN_VALUE, cache.cache.MAX_VALUE)),
            )

            cadetValues, cadetValuesExtended = util.convert_individual(
                val.x, cache.cache
            )
            cadetValuesInput, cadetValuesExtendedInput = util.convert_individual(
                x, cache.cache
            )

            val.cadetValues = cadetValues
            val.cadetValuesExtended = cadetValuesExtended
            val.cadetValuesInput = cadetValuesInput
            val.cadetValuesExtendedInput = cadetValuesExtendedInput

            multiprocessing.get_logger().info(
                "gradient optimization result start: %s (%s) result: %s",
                x,
                cadetValues,
                val,
            )

            return val
        except GradientException:
            # If the gradient fails return None as the point so the optimizer can adapt
            return None


def fitness_grad(individual, finished=0):
    return fitness_base(evo.runExperiment, individual, finished)


def fitness_sens_grad(individual, finished=0):
    return fitness_sens(individual, finished)


def fitness(individual, finished=1):
    return fitness_base(evo.runExperiment, individual, finished)


def fitness_sens(individual, finished=1):
    return fitness_base(runExperimentSens, individual, finished)


def fitness_base(fit, individual, finished):
    scores = []
    error = 0.0

    diff = []

    results = {}
    for experiment in cache.cache.settings["experiments"]:
        setup_experiments(cache.cache, experiment)
        result = fit(
            individual,
            "simulationSens",
            experiment,
            cache.cache.settings,
            cache.cache.target,
            cache.cache,
        )

        if result is not None:
            results[experiment["name"]] = result
            scores.extend(results[experiment["name"]]["scores"])
            error += results[experiment["name"]]["error"]
            diff.extend(results[experiment["name"]]["diff"])
        else:
            raise GradientException("Gradient caused simulation failure, aborting")

    # need to minimize
    if cache.cache.gradVector:
        return numpy.array(diff)
    else:
        scores = numpy.array(scores)
        if cache.cache.allScoreNorm:
            return scores
        elif cache.cache.allScoreSSE:
            return numpy.array(diff)
        else:
            multiprocessing.get_logger().info(
                "Norm 1 and SSE based scores can't be mixed %s", cache.cache.badScores
            )
            sys.exit()


def saveExperimentsSens(save_name_base, settings, target, results):
    return util.saveExperiments(
        save_name_base,
        settings,
        target,
        results,
        settings["resultsDirGrad"],
        "%s_%s_GRAD.h5",
    )


def setup_experiments(cache, experiment):
    if "simulationSens" not in experiment:
        templatePath = Path(
            cache.settings["resultsDirMisc"], "template_%s_grad.h5" % experiment["name"]
        )
        templateSim = Cadet()
        templateSim.filename = templatePath.as_posix()
        templateSim.load()
        experiment["simulationSens"] = templateSim


def runExperimentSens(individual, template_name, experiment, settings, target, cache):
    setup_experiments(cache, experiment)

    return util.runExperiment(
        individual,
        experiment,
        settings,
        target,
        experiment[template_name],
        experiment[template_name].root.timeout,
        cache,
    )
