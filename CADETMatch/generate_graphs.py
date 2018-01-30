import sys
from matplotlib import figure
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from mpl_toolkits.mplot3d import Axes3D

from cache import cache

from pathlib import Path
import pandas
import numpy
import itertools

def main():
    cache.setup(sys.argv[1])
    cache.progress_path = Path(cache.settings['resultsDirBase']) / "progress.csv"

    graphProgress(cache)
    graphSpace(cache)

def graphSpace(cache):
    csv_path = Path(cache.settings['resultsDirBase']) / cache.settings['CSV']
    output = cache.settings['resultsDirSpace']

    comp_two = list(itertools.combinations(cache.parameter_indexes, 2))
    comp_one = list(itertools.combinations(cache.parameter_indexes, 1))

    #3d plots
    prod = list(itertools.product(comp_two, cache.score_indexes))
    seq = [(str(output), str(csv_path), i[0][0], i[0][1], i[1]) for i in prod]
    for i in seq:
        plot_3d(i)
    
    #2d plots
    prod = list(itertools.product(comp_one, cache.score_indexes))
    seq = [(str(output), str(csv_path), i[0][0], i[1]) for i in prod]
    for i in seq:
        plot_2d(i)

def plot_3d(arg):
    "This leaks memory and should be disabled for now"
    directory_path, csv_path, c1, c2, score = arg
    dataframe = pandas.read_csv(csv_path)
    directory = Path(directory_path)

    headers = dataframe.columns.values.tolist()
    #print('3d', headers[c1], headers[c2], headers[score])

    scores = numpy.array(dataframe.iloc[:, score])
    scoreName = headers[score]
    if headers[score] == 'SSE':
        scores = -numpy.log(scores)
        scoreName = '-log(%s)' % headers[score]
    
    x = numpy.array(dataframe.iloc[:, c1])
    y = numpy.array(dataframe.iloc[:, c2])

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(numpy.log(x), numpy.log(y), scores, c=scores, cmap=cm.get_cmap('winter'))
    ax.set_xlabel('log(%s)' % headers[c1])
    ax.set_ylabel('log(%s)' % headers[c2])
    ax.set_zlabel(scoreName)
    filename = "%s_%s_%s.png" % (c1, c2, score)
    fig.savefig(str(directory / filename), bbox_inches='tight')
    
def plot_2d(arg):
    directory_path, csv_path, c1, score = arg
    dataframe = pandas.read_csv(csv_path)
    directory = Path(directory_path)
    headers = dataframe.columns.values.tolist()
    #print('2d', headers[c1], headers[score])

    scores = dataframe.iloc[:, score]
    scoreName = headers[score]
    if headers[score] == 'SSE':
        scores = -numpy.log(scores)
        scoreName = '-log(%s)' % headers[score]

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)
    graph.scatter(numpy.log(dataframe.iloc[:, c1]), scores, c=scores, cmap=cm.get_cmap('winter'))
    graph.set_xlabel('log(%s)' % headers[c1])
    graph.set_ylabel(scoreName)
    filename = "%s_%s.png" % (c1, score)
    fig.savefig(str(directory / filename), bbox_inches='tight')

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
