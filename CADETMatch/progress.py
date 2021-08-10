import csv
import multiprocessing
import time
import warnings
from pathlib import Path

import numpy
import psutil
from cadet import H5

import CADETMatch.util as util
import filelock

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


def process_pareto(cache, hof):
    data = numpy.array([i.fitness.values for i in hof])
    data_param = numpy.array(hof)
    data_param_transform = util.convert_population_inputorder(data_param, cache)

    return data, data_param, data_param_transform


def get_population_information(cache, population, generation):
    if cache.debugWrite:
        population_input = []
        population_output = []
        for ind in population:
            temp = [generation]
            temp.extend(ind)
            population_input.append(temp)

            temp = [generation]
            temp.extend(ind.fitness.values)
            population_output.append(temp)

        population_input = numpy.array(population_input)
        population_output = numpy.array(population_output)

        return population_input, population_output
    else:
        return None, None


def print_progress(
    cache,
    line_log,
    best_product,
    best_min,
    best_mean,
    best_sse,
    best_rmse,
    generation,
    population,
):
    if best_product is not None:
        alt_line_format = "Generation: %s \tPopulation: %s \tAverage Best: %.1e \tMinimum Best: %.1e \tProduct Best: %.1e\tSSE Best: %.3e \tRMSE Best: %.3e"

        sse_line_format = (
            "Generation: %s \tPopulation: %s \tSSE Best: %.3e \tRMSE Best: %.3e"
        )

        if line_log:
            if cache.allScoreSSE and not cache.MultiObjectiveSSE:
                multiprocessing.get_logger().info(
                    sse_line_format, generation, len(population), best_sse, best_rmse
                )
            else:
                multiprocessing.get_logger().info(
                    alt_line_format,
                    generation,
                    len(population),
                    best_mean,
                    best_min,
                    best_product,
                    best_sse,
                    best_rmse,
                )
    else:
        if line_log:
            multiprocessing.get_logger().info(
                "Generation: %s \tPopulation: %s \t No Stats Avaialable",
                generation,
                len(population),
            )


def get_best(cache, data_meta):
    if len(data_meta) and data_meta.ndim > 1:
        best_product = numpy.min(data_meta[:, 0])
        best_min = numpy.min(data_meta[:, 1])
        best_mean = numpy.min(data_meta[:, 2])
        best_sse = numpy.min(data_meta[:, 3])
        best_rmse = numpy.min(data_meta[:, 4])

        return best_product, best_min, best_mean, best_sse, best_rmse
    else:
        return None, None, None, None, None


def clear_result_data(result_data):
    result_data["input"] = []
    result_data["strategy"] = []
    result_data["mean"] = []
    result_data["confidence"] = []
    result_data["output"] = []
    result_data["output_meta"] = []
    result_data["input_transform"] = []
    result_data["input_transform_extended"] = []
    result_data["results"] = {}
    result_data["times"] = {}

    if "mcmc_score" in result_data:
        result_data["mcmc_score"] = []


def write_progress_csv(
    cache,
    data_meta,
    generation,
    population,
    now,
    sim_start,
    generation_start,
    cpu_time,
    line_log,
    meta_length,
):
    with cache.progress_path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)

        best_product, best_min, best_mean, best_sse, best_rmse = get_best(
            cache, data_meta
        )

        print_progress(
            cache,
            line_log,
            best_product,
            best_min,
            best_mean,
            best_sse,
            best_rmse,
            generation,
            population,
        )

        # ['Generation', 'Population', 'Dimension In', 'Dimension Out', 'Search Method',
        #'Meta Front', 'Meta Min', 'Meta Product', 'Meta Mean', 'Meta SSE',
        #'Elapsed Time', 'Generation Time', 'Total CPU Time', 'Last Progress Generation',
        #'Generations of Progress']

        writer.writerow(
            [
                generation,
                len(population),
                len(cache.MIN_VALUE),
                cache.numGoals,
                cache.settings.get("searchMethod", "NSGA3"),
                meta_length,
                best_min,
                best_product,
                best_mean,
                best_sse,
                best_rmse,
                now - sim_start,
                now - generation_start,
                cpu_time.user + cpu_time.system,
                cache.lastProgressGeneration,
                cache.generationsOfProgress,
            ]
        )


def write_results(
    cache,
    result_h5,
    result_data,
    gen_data,
    now,
    sim_start,
    population_input,
    population_output,
    data,
    hof_param,
    hof_param_transform,
    data_meta,
    meta_param,
    meta_param_transform,
    data_grad,
    grad_param,
    grad_param_transform,
    probability,
):
    with h5py.File(result_h5, "w") as hf:
        hf.create_dataset(
            "input",
            data=result_data["input"],
            maxshape=(None, len(result_data["input"][0])),
        )

        if len(result_data["strategy"]):
            hf.create_dataset(
                "strategy",
                data=result_data["strategy"],
                maxshape=(None, len(result_data["strategy"][0])),
            )

        if len(result_data["mean"]):
            hf.create_dataset(
                "mean",
                data=result_data["mean"],
                maxshape=(None, len(result_data["mean"][0])),
            )

        if len(result_data["confidence"]):
            hf.create_dataset(
                "confidence",
                data=result_data["confidence"],
                maxshape=(None, len(result_data["confidence"][0])),
            )

        if cache.correct is not None:
            distance = cache.correct - result_data["input"]
            hf.create_dataset(
                "distance_correct",
                data=distance,
                maxshape=(None, len(result_data["input"][0])),
            )

            distance_transform = (
                cache.correct_transform - result_data["input_transform"]
            )
            hf.create_dataset(
                "distance_correct_transform",
                data=distance_transform,
                maxshape=(None, len(result_data["input_transform"][0])),
            )

        hf.create_dataset(
            "output",
            data=result_data["output"],
            maxshape=(None, len(result_data["output"][0])),
        )
        hf.create_dataset(
            "output_meta",
            data=result_data["output_meta"],
            maxshape=(None, len(result_data["output_meta"][0])),
        )

        hf.create_dataset(
            "input_transform",
            data=result_data["input_transform"],
            maxshape=(None, len(result_data["input_transform"][0])),
        )

        if numpy.array_equal(
            result_data["input_transform"], result_data["input_transform_extended"]
        ):
            hf.create_dataset("is_extended_input", data=False)
        else:
            hf.create_dataset("is_extended_input", data=True)
            hf.create_dataset(
                "input_transform_extended",
                data=result_data["input_transform_extended"],
                maxshape=(None, len(result_data["input_transform_extended"][0])),
            )

        hf.create_dataset("generation", data=gen_data, maxshape=(None, 2))
        hf.create_dataset("total_time", data=now - sim_start)

        if population_input is not None:
            hf.create_dataset(
                "population_input",
                data=population_input,
                maxshape=(None, population_input.shape[1]),
            )

        if population_output is not None:
            hf.create_dataset(
                "population_output",
                data=population_output,
                maxshape=(None, population_output.shape[1]),
            )

        if probability is not None:
            hf.create_dataset("probability", data=probability, maxshape=(None, 1))

        if cache.debugWrite:
            mcmc_score = result_data.get("mcmc_score", None)
            if mcmc_score is not None:
                hf.create_dataset(
                    "mcmc_score", data=mcmc_score, maxshape=(None, len(mcmc_score[0]))
                )

        if len(hof_param):
            hf.create_dataset(
                "hof_population", data=hof_param, maxshape=(None, hof_param.shape[1])
            )
            hf.create_dataset(
                "hof_population_transform",
                data=hof_param_transform,
                maxshape=(None, hof_param_transform.shape[1]),
            )
            hf.create_dataset(
                "hof_score_original", data=data, maxshape=(None, data.shape[1])
            )
            hf.create_dataset(
                "hof_score",
                data=data,
                maxshape=(None, data.shape[1]),
            )

        if len(meta_param):
            hf.create_dataset(
                "meta_population", data=meta_param, maxshape=(None, meta_param.shape[1])
            )
            hf.create_dataset(
                "meta_population_transform",
                data=meta_param_transform,
                maxshape=(None, meta_param_transform.shape[1]),
            )
            hf.create_dataset(
                "meta_score",
                data=data_meta,
                maxshape=(None, data_meta.shape[1]),
            )

        if len(grad_param):
            hf.create_dataset(
                "grad_population", data=grad_param, maxshape=(None, grad_param.shape[1])
            )
            hf.create_dataset(
                "grad_population_transform",
                data=grad_param_transform,
                maxshape=(None, grad_param_transform.shape[1]),
            )
            hf.create_dataset(
                "grad_score_original",
                data=data_grad,
                maxshape=(None, data_grad.shape[1]),
            )
            hf.create_dataset(
                "grad_score",
                data=data_grad,
                maxshape=(None, data_grad.shape[1]),
            )

        if cache.fullTrainingData:

            for filename, chroma in result_data["results"].items():
                hf.create_dataset(
                    filename, data=chroma, maxshape=(None, len(chroma[0]))
                )

            for filename, chroma in result_data["times"].items():
                hf.create_dataset(filename, data=chroma)


def update_results(
    cache,
    result_h5,
    result_data,
    gen_data,
    now,
    sim_start,
    population_input,
    population_output,
    data,
    hof_param,
    hof_param_transform,
    data_meta,
    meta_param,
    meta_param_transform,
    data_grad,
    grad_param,
    grad_param_transform,
    probability,
):

    with h5py.File(result_h5, "a") as hf:
        hf["input"].resize((hf["input"].shape[0] + len(result_data["input"])), axis=0)
        hf["input"][-len(result_data["input"]) :] = result_data["input"]

        hf["total_time"][()] = now - sim_start
        hf["generation"][()] = gen_data

        if len(result_data["strategy"]):
            hf["strategy"].resize(
                (hf["strategy"].shape[0] + len(result_data["strategy"])), axis=0
            )
            hf["strategy"][-len(result_data["strategy"]) :] = result_data["strategy"]

        if len(result_data["mean"]):
            hf["mean"].resize((hf["mean"].shape[0] + len(result_data["mean"])), axis=0)
            hf["mean"][-len(result_data["mean"]) :] = result_data["mean"]

        if len(result_data["confidence"]):
            hf["confidence"].resize(
                (hf["confidence"].shape[0] + len(result_data["confidence"])), axis=0
            )
            hf["confidence"][-len(result_data["confidence"]) :] = result_data[
                "confidence"
            ]

        if cache.correct is not None:
            distance = cache.correct - result_data["input"]
            hf["distance_correct"].resize(
                (hf["distance_correct"].shape[0] + len(result_data["input"])), axis=0
            )
            hf["distance_correct"][-len(result_data["input"]) :] = distance

            distance_transform = (
                cache.correct_transform - result_data["input_transform"]
            )
            hf["distance_correct_transform"].resize(
                (
                    hf["distance_correct_transform"].shape[0]
                    + len(result_data["input_transform"])
                ),
                axis=0,
            )
            hf["distance_correct_transform"][
                -len(result_data["input_transform"]) :
            ] = distance_transform

        if population_input is not None:
            hf["population_input"].resize(
                (hf["population_input"].shape[0] + population_input.shape[0]), axis=0
            )
            hf["population_input"][-population_input.shape[0] :] = population_input

        if population_output is not None:
            hf["population_output"].resize(
                (hf["population_output"].shape[0] + population_output.shape[0]), axis=0
            )
            hf["population_output"][-population_output.shape[0] :] = population_output

        if probability is not None:
            hf["probability"].resize(
                (hf["probability"].shape[0] + probability.shape[0]), axis=0
            )
            hf["probability"][-probability.shape[0] :] = probability

        if cache.debugWrite:
            mcmc_score = result_data.get("mcmc_score", None)
            if mcmc_score is not None:
                hf["mcmc_score"].resize(
                    (hf["mcmc_score"].shape[0] + len(mcmc_score)), axis=0
                )
                hf["mcmc_score"][-len(mcmc_score) :] = mcmc_score

        hf["output"].resize(
            (hf["output"].shape[0] + len(result_data["output"])), axis=0
        )
        hf["output"][-len(result_data["output"]) :] = result_data["output"]

        hf["output_meta"].resize(
            (hf["output_meta"].shape[0] + len(result_data["output_meta"])), axis=0
        )
        hf["output_meta"][-len(result_data["output_meta"]) :] = result_data["output_meta"]

        hf["input_transform"].resize(
            (hf["input_transform"].shape[0] + len(result_data["input_transform"])),
            axis=0,
        )
        hf["input_transform"][-len(result_data["input_transform"]) :] = result_data[
            "input_transform"
        ]

        if not numpy.array_equal(
            result_data["input_transform"], result_data["input_transform_extended"]
        ):
            hf["input_transform_extended"].resize(
                (
                    hf["input_transform_extended"].shape[0]
                    + len(result_data["input_transform_extended"])
                ),
                axis=0,
            )
            hf["input_transform_extended"][
                -len(result_data["input_transform_extended"]) :
            ] = result_data["input_transform_extended"]

        if len(hof_param):
            hf["hof_population"].resize((hof_param.shape[0]), axis=0)
            hf["hof_population"][:] = hof_param

            hf["hof_population_transform"].resize(
                (hof_param_transform.shape[0]), axis=0
            )
            hf["hof_population_transform"][:] = hof_param_transform

            hf["hof_score_original"].resize((data.shape[0]), axis=0)
            hf["hof_score_original"][:] = data

            hf["hof_score"].resize((data.shape[0]), axis=0)
            hf["hof_score"][:] = data

        if len(meta_param):
            hf["meta_population"].resize((meta_param.shape[0]), axis=0)
            hf["meta_population"][:] = meta_param

            hf["meta_population_transform"].resize(
                (meta_param_transform.shape[0]), axis=0
            )
            hf["meta_population_transform"][:] = meta_param_transform

            hf["meta_score"].resize((data_meta.shape[0]), axis=0)
            hf["meta_score"][:] = data_meta

        if len(grad_param):
            if "grad_population" in hf:
                hf["grad_population"].resize((grad_param.shape[0]), axis=0)
                hf["grad_population"][:] = grad_param

                hf["grad_population_transform"].resize(
                    (grad_param_transform.shape[0]), axis=0
                )
                hf["grad_population_transform"][:] = grad_param_transform

                hf["grad_score_original"].resize((data_grad.shape[0]), axis=0)
                hf["grad_score_original"][:] = data_grad

                hf["grad_score"].resize((data_grad.shape[0]), axis=0)
                hf["grad_score"][:] = data_grad
            else:
                hf.create_dataset(
                    "grad_population",
                    data=grad_param,
                    maxshape=(None, grad_param.shape[1]),
                )
                hf.create_dataset(
                    "grad_population_transform",
                    data=grad_param_transform,
                    maxshape=(None, grad_param_transform.shape[1]),
                )
                hf.create_dataset(
                    "grad_score_original",
                    data=data_grad,
                    maxshape=(None, data_grad.shape[1]),
                )

                hf.create_dataset(
                    "grad_score",
                    data=data_grad,
                    maxshape=(None, data_grad.shape[1]),
                )

        if cache.fullTrainingData:

            for filename, chroma in result_data["results"].items():
                hf[filename].resize((hf[filename].shape[0] + len(chroma)), axis=0)
                hf[filename][-len(chroma) :] = chroma


def numpy_result_data(result_data):
    "convert list of lists type structures in result_data into numpy arrays"
    result_data["input"] = numpy.array(result_data["input"])
    result_data["strategy"] = numpy.array(result_data["strategy"])
    result_data["mean"] = numpy.array(result_data["mean"])
    result_data["confidence"] = numpy.array(result_data["confidence"])
    result_data["output"] = numpy.array(result_data["output"])
    result_data["output_meta"] = numpy.array(result_data["output_meta"])
    result_data["input_transform"] = numpy.array(result_data["input_transform"])
    result_data["input_transform_extended"] = numpy.array(
        result_data["input_transform_extended"]
    )

    if result_data["results"]:
        for key, value in result_data["results"].items():
            result_data["results"][key] = numpy.array(value)

    if result_data["times"]:
        for key, value in result_data["times"].items():
            result_data["times"][key] = numpy.array(value)

    if "mcmc_score" in result_data:
        result_data["mcmc_score"] = numpy.array(result_data["mcmc_score"])


def writeProgress(
    cache,
    generation,
    population,
    halloffame,
    meta_halloffame,
    grad_halloffame,
    progress_halloffame,
    sim_start,
    generation_start,
    result_data=None,
    line_log=True,
    probability=None,
):
    cpu_time = psutil.Process().cpu_times()
    now = time.time()

    numpy_result_data(result_data)

    results = Path(cache.settings["resultsDirBase"])

    data, hof_param, hof_param_transform = process_pareto(cache, halloffame)
    data_meta, meta_param, meta_param_transform = process_pareto(cache, meta_halloffame)
    data_grad, grad_param, grad_param_transform = process_pareto(cache, grad_halloffame)

    gen_data = numpy.array([generation, len(result_data["input"])]).reshape(1, 2)

    if probability is not None:
        probability = probability.reshape(-1, 1)

    population_input, population_output = get_population_information(
        cache, population, generation
    )

    if result_data is not None:
        resultDir = Path(cache.settings["resultsDir"])
        result_h5 = resultDir / "result.h5"

        lock = filelock.FileLock(result_h5.as_posix() + '.lock')

        if not result_h5.exists():
            with lock:
                write_results(
                    cache,
                    result_h5,
                    result_data,
                    gen_data,
                    now,
                    sim_start,
                    population_input,
                    population_output,
                    data,
                    hof_param,
                    hof_param_transform,
                    data_meta,
                    meta_param,
                    meta_param_transform,
                    data_grad,
                    grad_param,
                    grad_param_transform,
                    probability,
                )
        else:
            with lock:
                update_results(
                    cache,
                    result_h5,
                    result_data,
                    gen_data,
                    now,
                    sim_start,
                    population_input,
                    population_output,
                    data,
                    hof_param,
                    hof_param_transform,
                    data_meta,
                    meta_param,
                    meta_param_transform,
                    data_grad,
                    grad_param,
                    grad_param_transform,
                    probability,
                )

        clear_result_data(result_data)

    write_progress_csv(
        cache,
        data_meta,
        generation,
        population,
        now,
        sim_start,
        generation_start,
        cpu_time,
        line_log,
        len(meta_halloffame),
    )
