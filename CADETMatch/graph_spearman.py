import sys

import matplotlib
import matplotlib.style as mplstyle
mplstyle.use('fast')

matplotlib.use("Agg")
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from CADETMatch.cache import cache

from pathlib import Path
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import numpy
import scipy.stats
import multiprocessing


def main():
    cache.setup(sys.argv[1])
    cache.progress_path = Path(cache.settings["resultsDirBase"]) / "progress.csv"

    generation = sys.argv[2]

    result_path = Path(cache.settings["resultsDirBase"]) / "result.h5"
    output_spear = cache.settings["resultsDirSpace"] / "spearman"

    output_spear.mkdir(parents=True, exist_ok=True)

    with h5py.File(result_path, "r") as h5:
        data_output = h5["/output"][()]
        data_generation = h5["/generation"][()]

        data_generation = h5["/generation"][(-1, -1)]

        data_output = h5["/output"][-data_generation:, :]
        data_in = h5["/input"][-data_generation:, :]
        data_meta = h5["/output_meta"][-data_generation:, :]

        data = numpy.concatenate([data_in, data_output, data_meta], axis=1)

        spear, p_value = scipy.stats.spearmanr(data)

        fig = figure.Figure()
        canvas = FigureCanvas(fig)
        graph = fig.add_subplot(1, 1, 1)
        im = graph.imshow(spear, cmap="RdGy", interpolation="nearest", vmin=-1, vmax=1)
        graph.figure.colorbar(im, ax=graph)
        filename = "generation_%s.png" % (generation)
        fig.savefig(str(output_spear / filename), bbox_inches="tight")


if __name__ == "__main__":
    main()
