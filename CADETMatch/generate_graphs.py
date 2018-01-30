import sys
from matplotlib import figure
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from mpl_toolkits.mplot3d import Axes3D

from cache import cache

from pathlib import Path
import pandas
import numpy

def main():
    cache.setup(sys.argv[1])
    cache.progress_path = Path(cache.settings['resultsDirBase']) / "progress.csv"

    graphProgress(cache)

def graphProgress(cache):
    results = Path(cache.settings['resultsDirBase'])
    
    df = pandas.read_csv(str(cache.progress_path))

    hof = results / "hof.npy"

    with hof.open('rb') as hof_file:
        data = numpy.load(hof_file)

    output = cache.settings['resultsDirProgress']

    x = ['Generation', 'Total CPU Time']
    y = ['Average Score', 'Minimum Score', 'Product Score',
         'Pareto Mean Average Score', 'Pareto Mean Minimum Score', 'Pareto Mean Product Score']

    for i in x:
        for j in y:
            fig = figure.Figure()
            canvas = FigureCanvas(fig)

            graph = fig.add_subplot(1, 1, 1)

            graph.plot(df[i],df[j])
            graph.set_ylim((0,1))
            graph.set_title('%s vs %s' % (i,j))
            graph.set_xlabel(i)
            graph.set_ylabel(j)

            filename = "%s vs %s.png" % (i,j)
            file_path = output / filename
            fig.savefig(str(file_path))

    row, col = data.shape
    x_tick = numpy.array(range(col))
    x = numpy.repeat(x_tick, row, 0)
    x.shape = data.shape

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)

    graph.scatter(x, data)
    headers = [i.replace('_', ' ') for i in cache.score_headers]
    graph.set_xticks(x_tick)
    graph.tick_params(labelrotation = 90)
    graph.set_xticklabels(headers)
    graph.set_ylim((0,1))

    file_path = output / "scores.png"
    fig.savefig(bytes(file_path), bbox_inches='tight')

if __name__ == "__main__":
    main()
