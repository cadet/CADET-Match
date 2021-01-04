import sys

import matplotlib
import matplotlib.style as mplstyle

mplstyle.use("fast")

matplotlib.use("Agg")
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from cadet import H5
from sklearn.preprocessing import MinMaxScaler

from CADETMatch.cache import cache


def main():
    cache.setup(sys.argv[1])

    mcmcDir = Path(cache.settings["resultsDirMCMC"])

    dir_base = Path(cache.settings["resultsDirBase"])
    result_lock = dir_base / "result.lock"

    kde_settings = H5()
    kde_settings.filename = (mcmcDir / "kde_settings.h5").as_posix()
    kde_settings.load(lock=True)

    file = dir_base / "kde_data.h5"

    kde_data = H5()
    kde_data.filename = file.as_posix()
    kde_data.load(lock=True)

    store = kde_settings.root.store

    plt.figure(figsize=[10, 10])
    plt.scatter(store[:, 0], store[:, 1])
    plt.xlabel("bandwidth")
    plt.ylabel("cross_val_score")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(str(mcmcDir / "log_bandwidth.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=[10, 10])
    plt.scatter(store[:, 0], 1 - store[:, 1])
    plt.xlabel("bandwidth")
    plt.ylabel("1 - cross_val_score")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(str(mcmcDir / "1-log_bandwidth.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=[10, 10])
    plt.scatter(store[:, 0], store[:, 1])
    plt.xlabel("bandwidth")
    plt.ylabel("cross_val_score")
    plt.savefig(str(mcmcDir / "bandwidth.png"), bbox_inches="tight")
    plt.close()

    times = {}
    values = {}
    for key in kde_data.root.keys():
        if "_time" in key:
            times[key.replace("_time", "")] = kde_data.root[key]
        if "_unit" in key:
            values[key] = kde_data.root[key]

    error_model = mcmcDir / "error_model"
    error_model.mkdir(parents=True, exist_ok=True)

    scaler = MinMaxScaler()
    prob = numpy.exp(kde_settings.root.probability)
    prob = prob[: kde_settings.root.scores.shape[0]]
    prob = numpy.squeeze(scaler.fit_transform(prob.reshape(-1, 1)))

    sort_index = numpy.argsort(prob)

    colors = plt.cm.rainbow(prob)

    for key, value in values.items():
        time = times[key.split("_unit", 1)[0]]
        plt.figure(figsize=[20, 10])
        for idx in sort_index:
            plt.plot(time, value[idx, :], color=colors[idx])
        plt.xlabel("time")
        plt.ylabel("concentration")
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=1)
        )
        plt.colorbar(sm)
        plt.savefig(
            (error_model / ("%s.png" % key)).as_posix(), bbox_inches="tight", dpi=600
        )
        plt.close()

    for exp_name in kde_data.root.errors:
        temp = kde_data.root.errors[exp_name].pump_delays.reshape(-1, 1)
        if numpy.any(temp > 0):
            plt.figure(figsize=[10, 10])
            plt.hist(temp, bins=40)
            plt.savefig(
                (error_model / ("%s_pump_delays.png" % exp_name)).as_posix(),
                bbox_inches="tight",
            )
            plt.close()

        temp = kde_data.root.errors[exp_name].flow_rates.reshape(-1, 1)
        if numpy.any(temp > 0):
            plt.figure(figsize=[10, 10])
            plt.hist(temp, bins=40)
            plt.savefig(
                (error_model / ("%s_flow_rates.png" % exp_name)).as_posix(),
                bbox_inches="tight",
            )
            plt.close()

        temp = kde_data.root.errors[exp_name].loading_concentrations.reshape(-1, 1)
        if numpy.any(temp > 0):
            plt.figure(figsize=[10, 10])
            plt.hist(temp, bins=40)
            plt.savefig(
                (error_model / ("%s_loading_concentrations.png" % exp_name)).as_posix(),
                bbox_inches="tight",
            )
            plt.close()

    (error_model / "scores").mkdir(parents=True, exist_ok=True)
    for idx in range(kde_settings.root.scores.shape[1]):
        plt.figure(figsize=[10, 10])
        temp = kde_settings.root.scores[:, idx]
        plt.hist(temp[temp > 0], bins=40)
        plt.savefig(
            (error_model / "scores" / ("%s.png" % idx)).as_posix(), bbox_inches="tight"
        )
        plt.close()

    (error_model / "scores_mirror_scaled").mkdir(parents=True, exist_ok=True)
    for idx in range(kde_settings.root.scores_mirror_scaled.shape[1]):
        plt.figure(figsize=[10, 10])
        plt.hist(kde_settings.root.scores_mirror_scaled[:, idx], bins=40)
        plt.savefig(
            (error_model / "scores_mirror_scaled" / ("%s.png" % idx)).as_posix(),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    main()
