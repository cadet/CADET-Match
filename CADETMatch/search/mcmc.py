import random
import pickle
import CADETMatch.util as util
import CADETMatch.progress as progress
import numpy
import numpy as np
import scipy
from pathlib import Path
import time
import csv
import cadet

import emcee
import SALib.sample.sobol_sequence

import subprocess
import sys

import CADETMatch.evo as evo
import CADETMatch.cache as cache

import CADETMatch.pareto as pareto

import multiprocessing
import pandas
import array
import emcee.autocorr as autocorr

import CADETMatch.kde_generator as kde_generator
from sklearn.neighbors import KernelDensity

name = "MCMC"

from addict import Dict

import joblib

import CADETMatch.de as de
import CADETMatch.de_snooker as de_snooker
import CADETMatch.stretch as stretch

import jstyleson
import shutil

log2 = numpy.log(2)

min_acceptance = 0.2
acceptance_delta = 0.05

def log_previous(cadetValues, kde_previous, kde_previous_scaler):
    # find the right values to use
    col = len(kde_previous_scaler.scale_)
    values = cadetValues[-col:]
    values_shape = numpy.array(values).reshape(1, -1)
    values_scaler = kde_previous_scaler.transform(values_shape)
    score = kde_previous.score_samples(values_scaler)
    return score


def log_likelihood(individual, json_path):
    if json_path != cache.cache.json_path:
        cache.cache.setup_dir(json_path)
        util.setupLog(cache.cache.settings["resultsDirLog"], "main.log")
        cache.cache.setup(json_path, False)

    kde_previous, kde_previous_scaler = kde_generator.getKDEPrevious(cache.cache)

    if "kde" not in log_likelihood.__dict__:
        kde, kde_scaler = kde_generator.getKDE(cache.cache)
        log_likelihood.kde = kde
        log_likelihood.scaler = kde_scaler

    scores, csv_record, meta_score, results, individual = evo.fitness(individual, json_path)

    if results is None:
        return -numpy.inf, scores, csv_record, meta_score, results, individual

    if results is not None and kde_previous is not None:
        logPrevious = log_previous(individual, kde_previous, kde_previous_scaler)
    else:
        logPrevious = 0.0

    scores_shape = numpy.array(scores).reshape(1, -1)

    score_scaler = log_likelihood.scaler.transform(scores_shape)

    score_kde = log_likelihood.kde.score_samples(score_scaler)

    score = (
        score_kde + log2 + logPrevious
    )  # *2 is from mirroring and we need to double the probability to get back to the normalized distribution

    return score, scores, csv_record, meta_score, results, individual


def log_posterior_vectorize(population, json_path, cache, halloffame, meta_hof, grad_hof, progress_hof, result_data, writer, csvfile):
    results = cache.toolbox.map(log_posterior, ((population[i], json_path) for i in range(len(population))))
    results = process(population, cache, halloffame, meta_hof, grad_hof, progress_hof, result_data, results, writer, csvfile)
    return results


def outside_bounds(x, cache):
    for i, lb, ub in zip(x, cache.MIN_VALUE, cache.MAX_VALUE):
        if i < lb or i > ub:
            return True
    return False

def log_posterior(x):
    theta, json_path = x
    if json_path != cache.cache.json_path:
        cache.cache.setup_dir(json_path)
        util.setupLog(cache.cache.settings["resultsDirLog"], "main.log")
        cache.cache.setup(json_path)

    if outside_bounds(theta, cache.cache):
        return -numpy.inf, theta, cache.cache.WORST, [], cache.cache.WORST_META, None, theta

    ll, scores, csv_record, meta_score, results, individual = log_likelihood(theta, json_path)
    if results is None:
        return -numpy.inf, theta, cache.cache.WORST, [], cache.cache.WORST_META, None, individual
    else:
        return ll, theta, scores, csv_record, meta_score, results, individual


def addChain(*args):
    temp = [arg for arg in args if arg is not None]
    if len(temp) > 1:
        return numpy.concatenate(temp, axis=1)
    else:
        return numpy.array(temp[0])


def converged_bounds(chain, length, error_level):
    if chain.shape[1] < (2 * length):
        return False, None, None
    lb = []
    ub = []

    start = chain.shape[1] - length
    stop = chain.shape[1]
    for i in range(start, stop):
        temp_chain = chain[:, :i, :]
        temp_chain_shape = temp_chain.shape
        temp_chain_flat = temp_chain.reshape(temp_chain_shape[0] * temp_chain_shape[1], temp_chain_shape[2])
        lb_5, ub_95 = numpy.percentile(temp_chain_flat, [5, 95], 0)

        lb.append(lb_5)
        ub.append(ub_95)

    lb = numpy.array(lb)
    ub = numpy.array(ub)

    if numpy.all(numpy.std(lb, axis=0) < error_level) and numpy.all(numpy.std(ub, axis=0) < error_level):
        return True, numpy.mean(lb, axis=0), numpy.mean(ub, axis=0)
    else:
        multiprocessing.get_logger().info(
            "bounds have not yet converged lb min: %s max: %s std: %s  ub min %s max: %s std: %s",
            numpy.array2string(numpy.min(lb, axis=0), precision=3, separator=","),
            numpy.array2string(numpy.max(lb, axis=0), precision=3, separator=","),
            numpy.array2string(numpy.std(lb, axis=0), precision=4, separator=","),
            numpy.array2string(numpy.min(ub, axis=0), precision=3, separator=","),
            numpy.array2string(numpy.max(ub, axis=0), precision=3, separator=","),
            numpy.array2string(numpy.std(ub, axis=0), precision=4, separator=","),
        )
        return False, None, None


def rescale(cache, lb, ub, old_lb, old_ub, mcmc_store):
    "give a new lb and ub that will rescale so that the previous lb and ub takes up about 1/2 of the search width"
    new_size = len(lb)
    old_lb_slice = old_lb[:new_size]
    old_ub_slice = old_ub[:new_size]

    center = (ub + lb) / 2.0
    old_center = (old_ub + old_lb) / 2.0

    new_lb = lb - 2 * (center - lb)

    new_lb = numpy.max([new_lb, old_lb_slice], axis=0)

    new_ub = ub + 2 * (ub - center)

    new_ub = numpy.min([new_ub, old_ub_slice], axis=0)

    lb_trans = numpy.ones([3, len(old_lb)]) * old_lb
    ub_trans = numpy.ones([3, len(old_ub)]) * old_ub
    center_trans = numpy.ones([3, len(old_ub)]) * old_center

    lb_trans[1, :new_size] = lb
    lb_trans[2, :new_size] = new_lb

    ub_trans[1, :new_size] = ub
    ub_trans[2, :new_size] = new_ub

    center_trans[1, :new_size] = center
    center_trans[2, :new_size] = center

    lb_trans_conv = util.convert_population(lb_trans, cache)
    center_trans_conv = util.convert_population(center_trans, cache)
    ub_trans_conv = util.convert_population(ub_trans, cache)

    mcmc_store.root.bounds_change.lb_trans = lb_trans
    mcmc_store.root.bounds_change.ub_trans = ub_trans
    mcmc_store.root.bounds_change.center_trans = center_trans
    mcmc_store.root.bounds_change.lb_trans_conv = lb_trans_conv
    mcmc_store.root.bounds_change.center_trans_conv = center_trans_conv
    mcmc_store.root.bounds_change.ub_trans_conv = ub_trans_conv

    multiprocessing.get_logger().info(
        """rescaling bounds (simulator space)  \nold_lb: %s \nold_center: %s \nold_ub: %s
    \nlb5: %s \ncenter: %s \nub5: %s
    \nnew_lb: %s \nnew_center: %s \nnew_ub: %s
    \nbounds (search space)
    \nold_lb: %s \nold_center: %s \nold_ub: %s
    \nlb5: %s \ncenter: %s \nub5: %s
    \nnew_lb: %s \nnew_center: %s \nnew_ub: %s""",
        numpy.array2string(lb_trans_conv[0], precision=3, separator=","),
        numpy.array2string(center_trans_conv[0], precision=3, separator=","),
        numpy.array2string(ub_trans_conv[0], precision=3, separator=","),
        numpy.array2string(lb_trans_conv[1], precision=3, separator=","),
        numpy.array2string(center_trans_conv[1], precision=3, separator=","),
        numpy.array2string(ub_trans_conv[1], precision=3, separator=","),
        numpy.array2string(lb_trans_conv[2], precision=3, separator=","),
        numpy.array2string(center_trans_conv[2], precision=3, separator=","),
        numpy.array2string(ub_trans_conv[2], precision=3, separator=","),
        numpy.array2string(lb_trans[0], precision=3, separator=","),
        numpy.array2string(center_trans[0], precision=3, separator=","),
        numpy.array2string(ub_trans[0], precision=3, separator=","),
        numpy.array2string(lb_trans[1], precision=3, separator=","),
        numpy.array2string(center_trans[1], precision=3, separator=","),
        numpy.array2string(ub_trans[1], precision=3, separator=","),
        numpy.array2string(lb_trans[2], precision=3, separator=","),
        numpy.array2string(center_trans[2], precision=3, separator=","),
        numpy.array2string(ub_trans[2], precision=3, separator=","),
    )

    return lb_trans[2], center_trans[2], ub_trans[2]


def flatten(chain):
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])
    return flat_chain


def change_bounds_json(cache, lb, ub, mcmc_store):
    "change the bounds based on lb and ub and then save it as a new json file and return the path to the new file"
    multiprocessing.get_logger().info("change_bounds_json  lb %s  ub %s", lb, ub)
    lb_trans = util.convert_individual(lb, cache)[1]
    ub_trans = util.convert_individual(ub, cache)[1]

    settings_file = Path(cache.json_path)
    settings_file_backup = settings_file.with_suffix(".json.backup")

    new_name = "%s_bounds%s" % (settings_file.stem, settings_file.suffix)

    new_settings_file = settings_file.with_name(new_name)

    with settings_file.open() as json_data:
        settings = jstyleson.load(json_data)

        idx = 0
        for parameter in settings["parameters"]:
            transform = cache.transforms[parameter["transform"]](parameter, cache)
            count = transform.count_extended
            if count:
                multiprocessing.get_logger().warn("%s %s %s", idx, count, transform)
                lb_local = lb_trans[idx : idx + count]
                ub_local = ub_trans[idx : idx + count]
                transform.setBounds(parameter, lb_local, ub_local)
                idx = idx + count

        with new_settings_file.open(mode="w") as json_data:
            jstyleson.dump(settings, json_data, indent=4, sort_keys=False)

        mcmc_store.root.bounds_change.json = jstyleson.dumps(settings["parameters"], sort_keys=False)

    # copy the original file to a backup name
    shutil.copy(settings_file, settings_file_backup)

    # copy over our new settings file to the original file also
    # this is so that external programs also see the new bounds
    with settings_file.open(mode="w") as json_data:
        jstyleson.dump(settings, json_data, indent=4, sort_keys=False)

    return new_settings_file.as_posix()


def process_sampler_auto_bounds_write(cache, mcmc_store):
    bounds_seq = mcmc_store.root.bounds_acceptance
    bounds_chain = mcmc_store.root.bounds_full_chain

    bounds_chain, bounds_chain_flat, bounds_chain_transform, bounds_chain_flat_transform = process_chain(
        bounds_chain, cache, len(bounds_seq) - 1
    )

    mcmc_store.root.bounds_full_chain_transform = bounds_chain_transform
    mcmc_store.root.bounds_flat_chain = bounds_chain_flat
    mcmc_store.root.bounds_flat_chain_transform = bounds_chain_flat_transform


def sampler_auto_bounds(cache, checkpoint, sampler, checkpointFile, mcmc_store):
    bounds_seq = checkpoint.get("bounds_seq", [])

    bounds_chain = checkpoint.get("bounds_chain", None)

    checkInterval = 25

    parameters = len(cache.MIN_VALUE)

    if "mcmc_h5" in cache.settings:
        data = cadet.H5()
        data.filename = cache.settings["mcmc_h5"]
        data.load(paths=["/bounds_change/center_trans"])
        previous_parameters = data.root.bounds_change.center_trans.shape[1]
    else:
        previous_parameters = 0

    new_parameters = parameters - previous_parameters

    finished = False

    generation = checkpoint["idx_bounds"]

    sampler.iterations = checkpoint["sampler_iterations"]
    sampler.naccepted = checkpoint["sampler_naccepted"]
    sampler._moves[1].n = checkpoint["sampler_n"]

    while not finished:
        state = next(
            sampler.sample(
                checkpoint["p_bounds"], log_prob0=checkpoint["ln_prob_bounds"], rstate0=checkpoint["rstate_bounds"], iterations=1
            )
        )

        p = state.coords
        ln_prob = state.log_prob
        random_state = state.random_state

        accept = numpy.mean(sampler.acceptance_fraction)
        bounds_seq.append(accept)

        bounds_chain = addChain(bounds_chain, p[:, numpy.newaxis, :])

        multiprocessing.get_logger().info("run:  idx: %s accept: %.3f", generation, accept)

        generation += 1

        checkpoint["p_bounds"] = p
        checkpoint["ln_prob_bounds"] = ln_prob
        checkpoint["rstate_bounds"] = random_state
        checkpoint["idx_bounds"] = generation
        checkpoint["bounds_chain"] = bounds_chain
        checkpoint["bounds_seq"] = bounds_seq
        checkpoint["bounds_iterations"] = sampler.iterations
        checkpoint["bounds_naccepted"] = sampler.naccepted

        mcmc_store.root.bounds_acceptance = numpy.array(bounds_seq).reshape(-1, 1)
        mcmc_store.root.bounds_full_chain = bounds_chain

        write_interval(cache.checkpointInterval, cache, checkpoint, checkpointFile, mcmc_store, process_sampler_auto_bounds_write)
        util.graph_corner_process(cache, last=False)

        if generation % checkInterval == 0:
            converged, lb, ub = converged_bounds(bounds_chain[:, :, :new_parameters], 200, 1e-3)

            if converged:
                # sys.exit()
                finished = True

                new_min_value, center, new_max_value = rescale(
                    cache, lb, ub, numpy.array(cache.MIN_VALUE), numpy.array(cache.MAX_VALUE), mcmc_store
                )

                json_path = change_bounds_json(cache, new_min_value, new_max_value, mcmc_store)
                cache.resetTransform(json_path)
                sampler.log_prob_fn.args[0] = json_path
                sampler.log_prob_fn.args[1] = cache
            else:
                multiprocessing.get_logger().info("bounds have not yet converged in gen %s", generation)

    sampler.reset()
    checkpoint["state"] = "burn_in"
    checkpoint["p_burn"] = p

    write_interval(-1, cache, checkpoint, checkpointFile, mcmc_store, process_sampler_auto_bounds_write)


def process_interval(cache, mcmc_store, interval_chain, interval_chain_transform):
    mean = numpy.mean(interval_chain_transform, 0)
    labels = [5, 10, 50, 90, 95]
    percentile = numpy.percentile(interval_chain_transform, labels, 0)

    mcmc_store.root.percentile["mean"] = mean
    for idx, label in enumerate(labels):
        mcmc_store.root.percentile["percentile_%s" % label] = percentile[idx, :]

    flat_interval = interval(interval_chain, cache)
    flat_interval_transform = interval(interval_chain_transform, cache)

    mcmcDir = Path(cache.settings["resultsDirMCMC"])
    flat_interval.to_csv(mcmcDir / "percentile.csv")
    flat_interval_transform.to_csv(mcmcDir / "percentile_transform.csv")


def process_sampler_burn_write(cache, mcmc_store):
    train_chain = mcmc_store.root.train_full_chain
    burn_seq = mcmc_store.root.burn_seq
    train_chain_stat = mcmc_store.root.train_chain_stat

    train_chain, train_chain_flat, train_chain_transform, train_chain_flat_transform = process_chain(train_chain, cache, len(burn_seq) - 1)

    mcmc_store.root.train_full_chain_transform = train_chain_transform
    mcmc_store.root.train_flat_chain = train_chain_flat
    mcmc_store.root.train_flat_chain_transform = train_chain_flat_transform

    train_chain_stat, _, train_chain_stat_transform, _ = process_chain(train_chain_stat, cache, len(burn_seq) - 1)

    mcmc_store.root.train_chain_stat_transform = train_chain_stat_transform

    interval_chain = train_chain_flat
    interval_chain_transform = train_chain_flat_transform
    process_interval(cache, mcmc_store, interval_chain, interval_chain_transform)


def sampler_burn(cache, checkpoint, sampler, checkpointFile, mcmc_store):
    burn_seq = checkpoint.get("burn_seq", [])

    train_chain = checkpoint.get("train_chain", None)

    train_chain_stat = checkpoint.get("train_chain_stat", None)

    converge = checkpoint.get("converge")

    parameters = len(cache.MIN_VALUE)

    tol = 5e-4
    power = checkpoint["sampler_n"]
    distance = 1.0
    # distance_a = sampler.a
    stop_next = False
    finished = False

    generation = checkpoint["idx_burn"]

    sampler.iterations = checkpoint["sampler_iterations"]
    sampler.naccepted = checkpoint["sampler_naccepted"]
    sampler._moves[1].n = checkpoint["sampler_n"]

    while not finished:
        state = next(
            sampler.sample(
                checkpoint["p_burn"], log_prob0=checkpoint["ln_prob_burn"], rstate0=checkpoint["rstate_burn"], iterations=1, tune=False
            )
        )

        p = state.coords
        ln_prob = state.log_prob
        random_state = state.random_state

        accept = numpy.mean(sampler.acceptance_fraction)
        burn_seq.append(accept)
        converge[:-1] = converge[1:]
        converge[-1] = accept

        train_chain = addChain(train_chain, p[:, numpy.newaxis, :])

        train_chain_stat = addChain(train_chain_stat, numpy.percentile(flatten(train_chain), [5, 50, 95], 0)[:, numpy.newaxis, :])

        converge_real = converge[~numpy.isnan(converge)]
        multiprocessing.get_logger().info(
            "burn:  idx: %s accept: %.3g std: %.3g mean: %.3g converge: %.3g",
            generation,
            accept,
            numpy.std(converge_real),
            numpy.mean(converge_real),
            numpy.std(converge_real) / tol,
        )

        generation += 1

        checkpoint["p_burn"] = p
        checkpoint["ln_prob_burn"] = ln_prob
        checkpoint["rstate_burn"] = random_state
        checkpoint["idx_burn"] = generation
        checkpoint["train_chain"] = train_chain
        checkpoint["burn_seq"] = burn_seq
        checkpoint["converge"] = converge
        checkpoint["sampler_iterations"] = sampler.iterations
        checkpoint["sampler_naccepted"] = sampler.naccepted
        checkpoint["train_chain_stat"] = train_chain_stat
        checkpoint["sampler_n"] = sampler._moves[1].n

        mcmc_store.root.train_full_chain = train_chain
        mcmc_store.root.burn_seq = numpy.array(burn_seq).reshape(-1, 1)
        mcmc_store.root.train_chain_stat = train_chain_stat

        write_interval(cache.checkpointInterval, cache, checkpoint, checkpointFile, mcmc_store, process_sampler_burn_write)
        util.graph_corner_process(cache, last=False)

        if numpy.std(converge_real) < tol and len(converge) == len(converge_real):
            average_converge = numpy.mean(converge)
            if average_converge > min_acceptance:
                multiprocessing.get_logger().info("burn in completed at iteration %s", generation)
                finished = True

            if stop_next is True:
                multiprocessing.get_logger().info("burn in completed at iteration %s based on minimum distances", generation)
                finished = True

            if not finished:
                new_distance = min_acceptance - average_converge
                if new_distance < distance:
                    distance_n = sampler._moves[1].n
                    distance = new_distance

                    multiprocessing.get_logger().info(
                        "burn in acceptance is out of tolerance and n must be adjusted while burn in continues"
                    )
                    converge[:] = numpy.nan
                    prev_n = sampler._moves[1].n
                    if average_converge < (min_acceptance - 3 * acceptance_delta):
                        # n must be decreased to increase the acceptance rate (step size)
                        power -= 4
                    elif average_converge < (min_acceptance - 2 * acceptance_delta):
                        # n must be decreased to increase the acceptance rate (step size)
                        power -= 2
                    elif average_converge < (min_acceptance - 1 * acceptance_delta):
                        # n must be decreased to increase the acceptance rate (step size)
                        power -= 1
                    new_n = power
                    sampler._moves[1].n = power

                    mcmc_store.root.train_power = power

                    sampler.reset()
                    checkpoint["p_burn"] = checkpoint["starting_population"]
                    checkpoint["ln_prob_burn"] = None
                    multiprocessing.get_logger().info("previous n: %s    new n: %s", prev_n, new_n)
                else:
                    sampler._moves[1].n = distance_n

                    mcmc_store.root.train_power = distance_n

                    sampler.reset()
                    checkpoint["p_burn"] = checkpoint["starting_population"]
                    checkpoint["ln_prob_burn"] = None
                    stop_next = True

    checkpoint["sampler_iterations"] = sampler.iterations
    checkpoint["sampler_naccepted"] = sampler.naccepted
    checkpoint["state"] = "chain"
    checkpoint["p_chain"] = p
    checkpoint["ln_prob_burn"] = ln_prob
    checkpoint["rstate_burn"] = random_state
    checkpoint["sampler_a"] = sampler._moves[1].n

    write_interval(-1, cache, checkpoint, checkpointFile, mcmc_store, process_sampler_burn_write)


def process_sampler_run_write(cache, mcmc_store):
    chain = mcmc_store.root.full_chain
    chain_seq = mcmc_store.root.mcmc_acceptance
    run_chain_stat = mcmc_store.root.run_chain_stat

    chain, chain_flat, chain_transform, chain_flat_transform = process_chain(chain, cache, len(chain_seq) - 1)

    mcmc_store.root.full_chain_transform = chain_transform
    mcmc_store.root.flat_chain = chain_flat
    mcmc_store.root.flat_chain_transform = chain_flat_transform

    run_chain_stat, _, run_chain_stat_transform, _ = process_chain(run_chain_stat, cache, len(chain_seq) - 1)

    mcmc_store.root.run_chain_stat_transform = run_chain_stat_transform

    interval_chain = chain_flat
    interval_chain_transform = chain_flat_transform
    process_interval(cache, mcmc_store, interval_chain, interval_chain_transform)


def sampler_run(cache, checkpoint, sampler, checkpointFile, mcmc_store):
    chain_seq = checkpoint.get("chain_seq", [])

    run_chain = checkpoint.get("run_chain", None)

    run_chain_stat = checkpoint.get("run_chain_stat", None)

    iat = checkpoint.get("integrated_autocorrelation_time", [])

    checkInterval = 25

    parameters = len(cache.MIN_VALUE)

    finished = False

    generation = checkpoint["idx_chain"]

    sampler.iterations = checkpoint["sampler_iterations"]
    sampler.naccepted = checkpoint["sampler_naccepted"]
    sampler._moves[1].n = checkpoint["sampler_n"]
    tau_percent = None

    while not finished:
        state = next(
            sampler.sample(checkpoint["p_chain"], log_prob0=checkpoint["ln_prob_chain"], rstate0=checkpoint["rstate_chain"], iterations=1)
        )

        p = state.coords
        ln_prob = state.log_prob
        random_state = state.random_state

        accept = numpy.mean(sampler.acceptance_fraction)
        chain_seq.append(accept)

        run_chain = addChain(run_chain, p[:, numpy.newaxis, :])

        run_chain_stat = addChain(run_chain_stat, numpy.percentile(flatten(run_chain), [5, 50, 95], 0)[:, numpy.newaxis, :])

        multiprocessing.get_logger().info("run:  idx: %s accept: %.3f", generation, accept)

        generation += 1

        checkpoint["p_chain"] = p
        checkpoint["ln_prob_chain"] = ln_prob
        checkpoint["rstate_chain"] = random_state
        checkpoint["idx_chain"] = generation
        checkpoint["run_chain"] = run_chain
        checkpoint["chain_seq"] = chain_seq
        checkpoint["sampler_iterations"] = sampler.iterations
        checkpoint["sampler_naccepted"] = sampler.naccepted
        checkpoint["run_chain_stat"] = run_chain_stat

        mcmc_store.root.full_chain = run_chain
        mcmc_store.root.mcmc_acceptance = numpy.array(chain_seq).reshape(-1, 1)
        mcmc_store.root.run_chain_stat = run_chain_stat

        if generation % checkInterval == 0:
            try:
                tau = autocorr.integrated_time(numpy.swapaxes(run_chain, 0, 1), tol=cache.MCMCTauMult)
                multiprocessing.get_logger().info(
                    "Mean acceptance fraction: %s %0.3f tau: %s with shape: %s", generation, accept, tau, run_chain.shape
                )
                if numpy.any(numpy.isnan(tau)):
                    multiprocessing.get_logger().info("tau is NaN and clearly not complete %s", generation)
                else:
                    multiprocessing.get_logger().info("we have run long enough and can quit %s", generation)
                    finished = True
            except autocorr.AutocorrError as err:
                multiprocessing.get_logger().info(str(err))
                tau = err.tau
            multiprocessing.get_logger().info("Mean acceptance fraction: %s %0.3f tau: %s", generation, accept, tau)

            temp_iat = [generation]
            temp_iat.extend(tau)
            iat.append(temp_iat)
            checkpoint["integrated_autocorrelation_time"] = iat

            mcmc_store.root.integrated_autocorrelation_time = numpy.array(iat)

            tau = numpy.array(tau)
            tau_percent = generation / (tau * cache.MCMCTauMult)

            mcmc_store.root.tau_percent = tau_percent.reshape(-1, 1)

        write_interval(cache.checkpointInterval, cache, checkpoint, checkpointFile, mcmc_store, process_sampler_run_write)
        mle_process(last=False)
        util.graph_corner_process(cache, last=False)

    checkpoint["p_chain"] = p
    checkpoint["ln_prob_chain"] = ln_prob
    checkpoint["rstate_chain"] = random_state
    checkpoint["idx_chain"] = generation
    checkpoint["run_chain"] = run_chain
    checkpoint["chain_seq"] = chain_seq
    checkpoint["state"] = "complete"

    write_interval(-1, cache, checkpoint, checkpointFile, mcmc_store, process_sampler_run_write)


def write_interval(interval, cache, checkpoint, checkpointFile, mcmc_store, process_mcmc_store):
    "write the checkpoint and mcmc data at most every n seconds"
    if "last_time" not in write_interval.__dict__:
        write_interval.last_time = time.time()

    if time.time() - write_interval.last_time > interval:
        with checkpointFile.open("wb") as cp_file:
            pickle.dump(checkpoint, cp_file)

        writeMCMC(cache, mcmc_store, process_mcmc_store)

        write_interval.last_time = time.time()


def run(cache, tools, creator):
    "run the parameter estimation"
    random.seed()
    checkpointFile = Path(cache.settings["resultsDirMisc"], cache.settings.get("checkpointFile", "check"))
    checkpoint = getCheckPoint(checkpointFile, cache)

    mcmcDir = Path(cache.settings["resultsDirMCMC"])
    mcmc_h5 = mcmcDir / "mcmc.h5"
    mcmc_store = cadet.H5()
    mcmc_store.filename = mcmc_h5.as_posix()

    if mcmc_h5.exists():
        mcmc_store.load()

    parameters = len(cache.MIN_VALUE)

    MCMCpopulationSet = cache.settings.get("MCMCpopulationSet", None)
    if MCMCpopulationSet is not None:
        populationSize = MCMCpopulationSet
    else:
        populationSize = parameters * cache.settings["MCMCpopulation"]

    # Population must be even
    populationSize = populationSize + populationSize % 2

    # due to emcee 3.0 and RedBlueMove there is a minimum population size to work correctly based on the number of paramters
    populationSize = max(parameters * 2, populationSize)

    if checkpoint["state"] == "start":
        multiprocessing.get_logger().info("setting up kde")
        kde, kde_scaler = kde_generator.setupKDE(cache)
        checkpoint["state"] = "auto_bounds"

        with checkpointFile.open("wb") as cp_file:
            pickle.dump(checkpoint, cp_file)
    else:
        multiprocessing.get_logger().info("loading kde")
        kde, kde_scaler = kde_generator.getKDE(cache)

    path = Path(cache.settings["resultsDirBase"], cache.settings["csv"])
    with path.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_ALL)

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
            "mcmc_score": [],
        }
        halloffame = pareto.DummyFront()
        meta_hof = pareto.ParetoFrontMeta(similar=pareto.similar, similar_fit=pareto.similar_fit_meta(cache), slice_object=cache.meta_slice)
        grad_hof = pareto.DummyFront()
        progress_hof = None

        sampler = emcee.EnsembleSampler(
            populationSize,
            parameters,
            log_posterior_vectorize,
            args=[cache.json_path, cache, halloffame, meta_hof, grad_hof, progress_hof, result_data, writer, csvfile],
            moves=[(de_snooker.DESnookerMove(), 0.1), (de.DEMove(), 0.9 * 0.9), (de.DEMove(gamma0=1.0), 0.9 * 0.1),],
            vectorize=True,
        )

        if "sampler_n" not in checkpoint:
            checkpoint["sampler_n"] = sampler._moves[1].n

        if checkpoint["state"] == "auto_bounds":
            sampler_auto_bounds(cache, checkpoint, sampler, checkpointFile, mcmc_store)

        if checkpoint["state"] == "burn_in":
            sampler_burn(cache, checkpoint, sampler, checkpointFile, mcmc_store)

        if checkpoint["state"] == "chain":
            run_chain = checkpoint.get("run_chain", None)
            if run_chain is not None:
                temp = run_chain[:, : checkpoint["idx_chain"], 0].T
                multiprocessing.get_logger().info("complete shape %s", temp.shape)

                try:
                    tau = autocorr.integrated_time(numpy.swapaxes(run_chain[:, : checkpoint["idx_chain"], :], 0, 1), tol=cache.MCMCTauMult)
                    multiprocessing.get_logger().info("we have previously run long enough and can quit %s", checkpoint["idx_chain"])
                    checkpoint["state"] = "complete"
                except autocorr.AutocorrError as err:
                    multiprocessing.get_logger().info(str(err))
                    tau = err.tau

                multiprocessing.get_logger().info(
                    "Mean acceptance fraction: %s %0.3f tau: %s", checkpoint["idx_chain"], checkpoint["chain_seq"][-1], tau
                )

        if checkpoint["state"] == "chain":
            sampler_run(cache, checkpoint, sampler, checkpointFile, mcmc_store)

    chain = checkpoint["run_chain"]
    chain_shape = chain.shape
    chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])

    if checkpoint["state"] == "complete":
        tube_process(last=True)
        util.finish(cache)
        checkpoint["state"] = "plot_finish"

        with checkpointFile.open("wb") as cp_file:
            pickle.dump(checkpoint, cp_file)

    if checkpoint["state"] == "plot_finish":
        mle_process(last=True)
        util.graph_corner_process(cache, last=True)
    return numpy.mean(chain, 0)


def tube_process(last=False, interval=3600):
    cwd = str(Path(__file__).parent.parent)
    ret = subprocess.run(
        [sys.executable, "mcmc_plot_tube.py", str(cache.cache.json_path), str(util.getCoreCounts())],
        stdin=None,
        stdout=None,
        stderr=None,
        close_fds=True,
        cwd=cwd,
    )


def mle_process(last=False, interval=3600):
    if "last_time" not in mle_process.__dict__:
        mle_process.last_time = time.time()

    if "child" in mle_process.__dict__:
        if mle_process.child.poll() is None:  # This is false if the child has completed
            if last:
                mle_process.child.wait()
            else:
                return

    cwd = str(Path(__file__).parent.parent)

    if last:
        ret = subprocess.run(
            [sys.executable, "mle.py", str(cache.cache.json_path), str(util.getCoreCounts())],
            stdin=None,
            stdout=None,
            stderr=None,
            close_fds=True,
            cwd=cwd,
        )
        mle_process.last_time = time.time()
    elif (time.time() - mle_process.last_time) > interval:
        # mle_process.child = subprocess.Popen([sys.executable, 'mle.py', str(cache.cache.json_path), str(util.getCoreCounts())],
        #    stdin=None, stdout=None, stderr=None, close_fds=True,  cwd=cwd)
        subprocess.run(
            [sys.executable, "mle.py", str(cache.cache.json_path), str(util.getCoreCounts())],
            stdin=None,
            stdout=None,
            stderr=None,
            close_fds=True,
            cwd=cwd,
        )
        mle_process.last_time = time.time()


def get_population(base, size, diff=0.02):
    new_population = base
    row, col = base.shape
    multiprocessing.get_logger().info("%s", base)

    multiprocessing.get_logger().info("row %s size %s", row, size)
    if row < size:
        # create new entries
        indexes = numpy.random.choice(new_population.shape[0], size - row, replace=True)
        temp = new_population[indexes, :]
        rand = numpy.random.uniform(1.0 - diff, 1.0 + diff, size=temp.shape)
        new_population = numpy.concatenate([new_population, temp * rand])
    if row > size:
        # randomly select entries to keep
        indexes = numpy.random.choice(new_population.shape[0], size, replace=False)
        multiprocessing.get_logger().info("indexes: %s", indexes)
        new_population = new_population[indexes, :]
    change = numpy.random.normal(1.0, 0.01, new_population.shape)
    multiprocessing.get_logger().info(
        "Initial population condition number before %s  after %s",
        numpy.linalg.cond(new_population),
        numpy.linalg.cond(new_population * change),
    )
    return new_population * change


def resetPopulation(checkpoint, cache):
    populationSize = checkpoint["populationSize"]
    parameters = len(cache.MIN_VALUE)

    if cache.settings.get("PreviousResults", None) is not None:
        multiprocessing.get_logger().info("running with previous best results")
        previousResultsFile = Path(cache.settings["PreviousResults"])
        results_h5 = cadet.H5()
        results_h5.filename = previousResultsFile.as_posix()
        results_h5.load()
        previousResults = results_h5.root.meta_population_transform

        row, col = previousResults.shape
        multiprocessing.get_logger().info("row: %s col: %s  parameters: %s", row, col, parameters)
        if col < parameters:
            mcmc_h5 = Path(cache.settings.get("mcmc_h5", None))
            mcmcDir = mcmc_h5.parent
            mle_h5 = mcmcDir / "mle.h5"

            data = cadet.H5()
            data.filename = mle_h5.as_posix()
            data.load()
            multiprocessing.get_logger().info("%s", list(data.root.keys()))
            stat_MLE = data.root.stat_MLE.reshape(1, -1)
            previousResults = numpy.hstack([previousResults, numpy.repeat(stat_MLE, row, 0)])
            multiprocessing.get_logger().info("row: %s  col:%s   shape: %s", row, col, previousResults.shape)

        population = get_population(previousResults, populationSize, diff=0.1)
        checkpoint["starting_population"] = [util.convert_individual_inverse(i, cache) for i in population]
        multiprocessing.get_logger().info("p_burn startup population: %s", population)
        multiprocessing.get_logger().info("p_burn startup: %s", checkpoint["starting_population"])
    else:
        checkpoint["starting_population"] = SALib.sample.sobol_sequence.sample(populationSize, parameters)
    checkpoint["p_burn"] = checkpoint["p_bounds"] = checkpoint["starting_population"]


def getCheckPoint(checkpointFile, cache):
    if checkpointFile.exists():
        with checkpointFile.open("rb") as cp_file:
            checkpoint = pickle.load(cp_file)
    else:
        parameters = len(cache.MIN_VALUE)

        MCMCpopulationSet = cache.settings.get("MCMCpopulationSet", None)
        if MCMCpopulationSet is not None:
            populationSize = MCMCpopulationSet
        else:
            populationSize = parameters * cache.settings["MCMCpopulation"]

        # Population must be even
        populationSize = populationSize + populationSize % 2

        # due to emcee 3.0 and RedBlueMove there is a minimum population size to work correctly based on the number of paramters
        populationSize = max(parameters * 2, populationSize)

        checkpoint = {}
        checkpoint["state"] = "start"
        checkpoint["populationSize"] = populationSize
        resetPopulation(checkpoint, cache)

        checkpoint["ln_prob_bounds"] = None
        checkpoint["rstate_bounds"] = None
        checkpoint["idx_bounds"] = 0

        checkpoint["ln_prob_burn"] = None
        checkpoint["rstate_burn"] = None
        checkpoint["idx_burn"] = 0

        checkpoint["p_chain"] = None
        checkpoint["ln_prob_chain"] = None
        checkpoint["rstate_chain"] = None
        checkpoint["idx_chain"] = 0

        checkpoint["sampler_iterations"] = 0
        checkpoint["sampler_naccepted"] = numpy.zeros(populationSize)

        checkpoint["converge"] = numpy.ones(cache.settings.get("burnStable", 50)) * numpy.nan

    checkpoint["length_chain"] = cache.settings.get("chainLength", 50000)
    checkpoint["length_burn"] = cache.settings.get("burnIn", 50000)
    return checkpoint


def setupDEAP(cache, fitness, fitness_final, grad_fitness, grad_search, grad_search_fine, map_function, creator, base, tools):
    "setup the DEAP variables"
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * cache.numGoals)
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None, mean=None, confidence=None)

    creator.create("FitnessMaxMeta", base.Fitness, weights=[1.0, 1.0, 1.0, -1.0, -1.0])
    creator.create("IndividualMeta", array.array, typecode="d", fitness=creator.FitnessMaxMeta, strategy=None, best=None)
    cache.toolbox.register("individualMeta", util.initIndividual, creator.IndividualMeta, cache)

    cache.toolbox.register(
        "individual", util.generateIndividual, creator.Individual, len(cache.MIN_VALUE), cache.MIN_VALUE, cache.MAX_VALUE, cache
    )

    cache.toolbox.register("individual_guess", util.initIndividual, creator.Individual, cache)

    cache.toolbox.register("map", map_function)


def process(population_order, cache, halloffame, meta_hof, grad_hof, progress_hof, result_data, results, writer, csv_file):
    if "gen" not in process.__dict__:
        process.gen = 0

    if "sim_start" not in process.__dict__:
        process.sim_start = time.time()

    if "generation_start" not in process.__dict__:
        process.generation_start = time.time()

    population_lookup = {}
    fitnesses_lookup = {}

    log_likelihoods_lookup = {}

    keep = set()

    for ll, theta, fit, csv_line, meta_score, result, individual in results:
        log_likelihoods_lookup[tuple(individual)] = float(ll)
        if len(csv_line):
            #if csv_line is blank it means a simulation failed or is out of bounds, if this is handed on for
            #processing it will cause problems, failed simulations are arleady recorded and we don't want their
            #failure data in the recorers where other stuff picks it up
            keep.add(tuple(individual))            

            fitnesses_lookup[tuple(individual)] = (fit, csv_line, meta_score, result, tuple(individual))

            ind = cache.toolbox.individual_guess(individual)
            population_lookup[tuple(individual)] = ind

    # everything above is async (unordered) and needs to be reordered based on the population_order
    population = [population_lookup[tuple(row)] for row in population_order if tuple(row) in keep]
    fitnesses = [fitnesses_lookup[tuple(row)] for row in population_order if tuple(row) in keep]
    log_likelihoods = [log_likelihoods_lookup[tuple(row)] for row in population_order if tuple(row) in keep]
    log_likelihoods_all = [log_likelihoods_lookup[tuple(row)] for row in population_order]

    stalled, stallWarn, progressWarn = util.process_population(
        cache.toolbox, cache, population, fitnesses, writer, csv_file, halloffame, meta_hof, progress_hof, process.gen, result_data
    )

    progress.writeProgress(
        cache,
        process.gen,
        population,
        halloffame,
        meta_hof,
        grad_hof,
        progress_hof,
        process.sim_start,
        process.generation_start,
        result_data,
        line_log=False,
        probability=numpy.exp(log_likelihoods),
    )

    util.graph_process(cache, process.gen)

    process.gen += 1
    process.generation_start = time.time()
    return log_likelihoods_all


def process_chain(chain, cache, idx):
    chain_shape = chain.shape
    flat_chain = chain.reshape(chain_shape[0] * chain_shape[1], chain_shape[2])

    flat_chain_transform = util.convert_population(flat_chain, cache)
    chain_transform = flat_chain_transform.reshape(chain_shape)

    return chain, flat_chain, chain_transform, flat_chain_transform


def writeMCMC(cache, mcmc_store, process_mcmc_store):
    "write out the mcmc data so it can be plotted"
    process_mcmc_store(cache, mcmc_store)
    mcmc_store.save()


def interval(flat_chain, cache):
    mean = numpy.mean(flat_chain, 0)

    percentile = numpy.percentile(flat_chain, [5, 10, 50, 90, 95], 0)

    data = numpy.vstack((mean, percentile)).transpose()

    pd = pandas.DataFrame(data, columns=["mean", "5", "10", "50", "90", "95"])
    pd.insert(0, "name", cache.parameter_headers_actual)
    pd.set_index("name")
    return pd
