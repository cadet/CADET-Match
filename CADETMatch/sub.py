import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import filelock

import CADETMatch.util as util

processes = {}
times = {}


def get_lock(cache):
    resultDir = Path(cache.settings["resultsDirBase"])
    result_lock = resultDir / "result.lock"
    lock = filelock.FileLock(result_lock.as_posix())
    return lock


def run_sub(cache, key, line, file_name):
    cwd = str(Path(__file__).parent)

    sub = processes.get(key, None)

    if sub is None:
        multiprocessing.get_logger().info(
            "creating subprocess %s for %s", key, file_name
        )
        sub = subprocess.Popen(
            line,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
        processes[key] = sub

    process_sub(key, file_name)


def process_sub(key, file_name):
    sub = processes.get(key, None)

    if sub is not None:
        finished = sub.poll() is not None
        if finished is not False:
            del processes[key]
            stdout, stderr = sub.communicate()
            multiprocessing.get_logger().info(
                "finished subprocess %s for %s", key, file_name
            )
            log_subprocess(file_name, stdout.decode("utf-8"), stderr.decode("utf-8"))


def wait_sub(key, file_name):
    sub = processes.get(key, None)
    if sub is not None:
        multiprocessing.get_logger().info(
            "waiting for subprocess %s for %s", key, file_name
        )
        stdout, stderr = sub.communicate()
        del processes[key]
        multiprocessing.get_logger().info(
            "finished subprocess %s for %s", key, file_name
        )
        log_subprocess(file_name, stdout.decode("utf-8"), stderr.decode("utf-8"))


def log_subprocess(name, stdout, stderr):
    for line in stdout.splitlines():
        multiprocessing.get_logger().info("%s stdout: %s", name, line)

    for line in stderr.splitlines():
        multiprocessing.get_logger().info("%s stderr: %s", name, line)


def graph_spearman(cache, generation):
    if cache.graphSpearman:
        line = [
            sys.executable,
            "graph_spearman.py",
            str(cache.json_path),
            str(generation),
            str(util.getCoreCounts()),
        ]
        run_sub(cache, "spearman", line, "graph_spearman.py")


def wait_spearman():
    wait_sub("spearman", "graph_spearman.py")


def graph_corner(cache):
    line = [
        sys.executable,
        "generate_corner_graphs.py",
        str(cache.json_path),
        str(util.getCoreCounts()),
    ]
    run_sub(cache, "corner", line, "generate_corner_graphs.py")


def wait_corner():
    wait_sub("corner", "generate_corner_graphs.py")


def graph_autocorr(cache):
    line = [
        sys.executable,
        "generate_autocorr_graphs.py",
        str(cache.json_path),
        str(util.getCoreCounts()),
    ]
    run_sub(cache, "autocorr", line, "generate_autocorr_graphs.py")


def wait_autocorr():
    wait_sub("autocorr", "generate_autocorr_graphs.py")


def graph_mixing(cache):
    line = [
        sys.executable,
        "generate_mixing_graphs.py",
        str(cache.json_path),
        str(util.getCoreCounts()),
    ]
    run_sub(cache, "mixing", line, "generate_mixing_graphs.py")


def wait_mixing():
    wait_sub("mixing", "generate_mixing_graphs.py")


def graph_main(cache, graph_type):
    line = [
        sys.executable,
        "generate_graphs.py",
        str(cache.json_path),
        graph_type,
        str(util.getCoreCounts()),
    ]
    run_sub(cache, "main", line, "generate_graphs.py")


def wait_main():
    wait_sub("main", "generate_graphs.py")


def graph_kde(cache):
    line = [
        sys.executable,
        "graph_kde.py",
        str(cache.json_path),
        str(util.getCoreCounts()),
    ]
    run_sub(cache, "graph_kde", line, "graph_kde.py")


def process_kde():
    process_sub("graph_kde", "graph_kde.py")


def graph_mle(cache):
    line = [sys.executable, "mle.py", str(cache.json_path), str(util.getCoreCounts())]
    run_sub(cache, "graph_mle", line, "mle.py")


def wait_mle():
    wait_sub("graph_mle", "mle.py")


def graph_tube(cache):
    line = [
        sys.executable,
        "mcmc_plot_tube.py",
        str(cache.json_path),
        str(util.getCoreCounts()),
    ]
    run_sub(cache, "graph_tube", line, "mcmc_plot_tube.py")


def wait_tube():
    wait_sub("graph_tube", "mcmc_plot_tube.py")


def graph_spearman_video(cache):
    line = [
        sys.executable,
        "video_spearman.py",
        str(cache.json_path),
        str(getCoreCounts()),
    ]
    run_sub(cache, "spearman_video", line, "video_spearman.py")


def wait_spearman_video():
    wait_sub("spearman_video", "video_spearman.py")


def graph_process(cache, generation, last=False):
    lastGraphTime = times.get("lastGraphTime", time.time())
    lastMetaTime = times.get("lastMetaTime", time.time())

    if last:
        wait_main()

        graph_main(cache, str(cache.graphType))
        wait_main()
        lastGraphTime = time.time()

    elif (time.time() - lastGraphTime) > cache.graphGenerateTime:
        graph_main(cache, str(cache.graphType))
        lastGraphTime = time.time()
    else:
        if (time.time() - lastMetaTime) > cache.graphMetaTime:
            graph_main(cache, "0")
            lastMetaTime = time.time()

    times["lastGraphTime"] = lastGraphTime
    times["lastMetaTime"] = lastMetaTime


def graph_corner_process(cache, last=False, interval=3600):
    last_corner_time = times.get("last_corner_time", time.time())

    if last:
        process_kde()

        # there may be processes already running we need to wait for them and then run a last set on the final data set
        wait_corner()
        wait_autocorr()
        wait_mixing()

        graph_corner(cache)
        graph_autocorr(cache)
        graph_mixing(cache)

        wait_corner()
        wait_autocorr()
        wait_mixing()

        last_corner_time = time.time()
    elif (time.time() - last_corner_time) > interval:
        process_kde()
        graph_corner(cache)
        graph_autocorr(cache)
        graph_mixing(cache)

        last_corner_time = time.time()
    times["last_corner_time"] = last_corner_time


def mle_process(cache, last=False, interval=3600):
    last_mle_time = times.get("last_mle_time", time.time())

    cwd = str(Path(__file__).parent.parent)

    if last:
        wait_mle()

        graph_mle(cache)
        wait_mle()

        last_mle_time = time.time()
    elif (time.time() - last_mle_time) > interval:
        graph_mle(cache)

        last_mle_time = time.time()

    times["last_mle_time"] = last_mle_time


def tube_process(cache, last=False, interval=3600):
    last_tube_time = times.get("last_tube_time", time.time())

    cwd = str(Path(__file__).parent.parent)

    if last:
        wait_tube()

        graph_tube(cache)
        wait_tube()

        last_tube_time = time.time()
    elif (time.time() - last_tube_time) > interval:
        graph_tube(cache)

        last_tube_time = time.time()

    times["last_tube_time"] = last_tube_time


def graph_spearman(cache):
    if cache.graphSpearman:
        graph_spearman_video(cache)
        wait_spearman_video(cache)
