import sys
from matplotlib import figure
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from mpl_toolkits.mplot3d import Axes3D

from cache import cache

from pathlib import Path
import pandas
import numpy
import scipy.interpolate
import itertools

from cadet import Cadet
from addict import Dict

#parallelization
from scoop import futures

saltIsotherms = {b'STERIC_MASS_ACTION', b'SELF_ASSOCIATION', b'MULTISTATE_STERIC_MASS_ACTION', 
                 b'SIMPLE_MULTISTATE_STERIC_MASS_ACTION', b'BI_STERIC_MASS_ACTION'}

def main():
    cache.setup(sys.argv[1])
    cache.progress_path = Path(cache.settings['resultsDirBase']) / "progress.csv"

    graphProgress(cache)
    graphSpace(cache)
    graphExperiments(cache)

def graphExperiments(cache):
    directory = Path(cache.settings['resultsDirEvo'])

    #find all items in directory
    pathsH5 = directory.glob('*.h5')
    pathsPNG = directory.glob('*.png')

    #make set of items based on removing everything after _
    existsH5 = {str(path.name).split('_', 1)[0] for path in pathsH5}
    existsPNG = {str(path.name).split('_', 1)[0] for path in pathsPNG}

    toGenerate = existsH5 - existsPNG

    #for save_name_base in toGenerate:
    #    plotExperiments(save_name_base, cache.settings, cache.target, cache.settings['resultsDirEvo'], '%s_%s_EVO.png')

    list(futures.map(plotExperiments, toGenerate, itertools.repeat(sys.argv[1]), itertools.repeat(cache.settings['resultsDirEvo']), itertools.repeat('%s_%s_EVO.png')))

def graph_simulation(simulation, graph):
    ncomp = int(simulation.root.input.model.unit_001.ncomp)
    isotherm = bytes(simulation.root.input.model.unit_001.adsorption_model)

    hasSalt = isotherm in saltIsotherms

    solution_times = simulation.root.output.solution.solution_times

    comps = []

    hasColumn = isinstance(simulation.root.output.solution.unit_001.solution_outlet_comp_000, Dict)

    if hasColumn:
        for i in range(ncomp):
            comps.append(simulation.root.output.solution.unit_001['solution_column_outlet_comp_%03d' % i])
    else:
        for i in range(ncomp):
            comps.append(simulation.root.output.solution.unit_001['solution_outlet_comp_%03d' % i])

    if hasSalt:
        graph.set_title("Output")
        graph.plot(solution_times, comps[0], 'b-', label="Salt")
        graph.set_xlabel('time (s)')
        
        # Make the y-axis label, ticks and tick labels match the line color.
        graph.set_ylabel('mMol Salt', color='b')
        graph.tick_params('y', colors='b')

        colors = ['r', 'g', 'c', 'm', 'y', 'k']
        axis2 = graph.twinx()
        for idx, comp in enumerate(comps[1:]):
            axis2.plot(solution_times, comp, '%s-' % colors[idx], label="P%s" % idx)
        axis2.set_ylabel('mMol Protein', color='r')
        axis2.tick_params('y', colors='r')


        lines, labels = graph.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis2.legend(lines + lines2, labels + labels2, loc=0)
    else:
        graph.set_title("Output")
        
        colors = ['r', 'g', 'c', 'm', 'y', 'k']
        for idx, comp in enumerate(comps):
            graph.plot(solution_times, comp, '%s-' % colors[idx], label="P%s" % idx)
        graph.set_ylabel('mMol Protein', color='r')
        graph.tick_params('y', colors='r')
        graph.set_xlabel('time (s)')

        lines, labels = graph.get_legend_handles_labels()
        graph.legend(lines, labels, loc=0)

def plotExperiments(save_name_base, json_path, directory, file_pattern):
    if json_path != cache.json_path:
        cache.setup(json_path)

    target = cache.target
    settings = cache.settings
    for experiment in settings['experiments']:
        experimentName = experiment['name']
        
        dst = Path(directory, file_pattern % (save_name_base, experimentName))

        if dst.exists():
            #this item has already been plotted, this means that plots are occurring faster than generations are so kill this version
            sys.exit()

        numPlots = len(experiment['features']) + 1  #1 additional plot added as an overview for the simulation

        exp_time = target[experimentName]['time']
        exp_value = target[experimentName]['value']

        fig = figure.Figure(figsize=[10, numPlots*10])
        canvas = FigureCanvas(fig)

        simulation = Cadet()
        h5_path = Path(directory) / ('%s_%s_EVO.h5' % (save_name_base, experimentName))
        simulation.filename = bytes(h5_path)
        simulation.load()

        results = {}
        results['simulation'] = simulation

        graph_simulation(simulation, fig.add_subplot(numPlots, 1, 1))

        for idx, feature in enumerate(experiment['features']):
            graph = fig.add_subplot(numPlots, 1, idx+1+1) #additional +1 added due to the overview plot
            
            featureName = feature['name']
            featureType = feature['type']

            feat = target[experimentName][featureName]

            selected = feat['selected']
            exp_time = feat['time'][selected]
            exp_value = feat['value'][selected]

            sim_time, sim_value = get_times_values(simulation, target[experimentName][featureName])

            if featureType in ('similarity', 'similarityDecay', 'similarityHybrid', 'similarityHybrid2', 'similarityHybridDecay', 'similarityHybridDecay2', 'curve', 'breakthrough', 'dextran', 'dextranHybrid', 
                               'similarityCross', 'similarityCrossDecay', 'breakthroughCross', 'SSE', 'LogSSE', 'breakthroughHybrid', 'breakthroughHybrid2'):
                graph.plot(sim_time, sim_value, 'r--', label='Simulation')
                graph.plot(exp_time, exp_value, 'g:', label='Experiment')
            elif featureType in ('derivative_similarity', 'derivative_similarity_hybrid', 'derivative_similarity_hybrid2', 'derivative_similarity_cross', 'derivative_similarity_cross_alt'):
                #try:
                sim_spline = scipy.interpolate.UnivariateSpline(sim_time, smoothing(sim_time, sim_value), s=smoothing_factor(sim_value)).derivative(1)
                exp_spline = scipy.interpolate.UnivariateSpline(exp_time, smoothing(exp_time, exp_value), s=smoothing_factor(exp_value)).derivative(1)

                graph.plot(sim_time, sim_spline(sim_time), 'r--', label='Simulation')
                graph.plot(exp_time, exp_spline(exp_time), 'g:', label='Experiment')
                #except:
                #    pass
            elif featureType in ('fractionation', 'fractionationCombine', 'fractionationMeanVariance', 'fractionationMoment', 'fractionationSlide'):
                cache.scores[featureType].run(results, feat)


                graph_exp = results['graph_exp']
                graph_sim = results['graph_sim']

                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

                for idx, (key, value) in enumerate(graph_sim.items()):
                    (time, values) = zip(*value)
                    graph.plot(time, values, '%s--' % colors[idx], label='Simulation Comp: %s' % key)

                for idx, (key, value) in enumerate(graph_exp.items()):
                    (time, values) = zip(*value)
                    graph.plot(time, values, '%s:' % colors[idx], label='Experiment Comp: %s' % key)
            graph.legend()

        fig.savefig(bytes(dst))

def graphSpace(cache):
    csv_path = Path(cache.settings['resultsDirBase']) / cache.settings['CSV']
    output = cache.settings['resultsDirSpace']

    comp_two = list(itertools.combinations(cache.parameter_indexes, 2))
    comp_one = list(itertools.combinations(cache.parameter_indexes, 1))

    #3d plots
    prod = list(itertools.product(comp_two, cache.score_indexes))
    seq = [(str(output), str(csv_path), i[0][0], i[0][1], i[1]) for i in prod]
    list(futures.map(plot_3d, seq))
    
    #2d plots
    prod = list(itertools.product(comp_one, cache.score_indexes))
    seq = [(str(output), str(csv_path), i[0][0], i[1]) for i in prod]
    list(futures.map(plot_2d, seq))

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

    list(futures.map(singleGraphProgress, itertools.product(x,y), itertools.repeat(df), itertools.repeat(output)))

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

def singleGraphProgress(tup, df, output):
    i,j = tup

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


def get_times_values(simulation, target, selected = None):

    times = simulation.root.output.solution.solution_times

    isotherm = target['isotherm']

    if isinstance(isotherm, list):
        values = numpy.sum([simulation[i] for i in isotherm], 0)
    else:
        values = simulation[isotherm]
    
    if selected is None:
        selected = target['selected']

    return times[selected], values[selected]

def smoothing(times, values):
    #temporarily get rid of smoothing for debugging
    #return values
    #filter length must be odd, set to 10% of the feature size and then make it odd if necesary
    filter_length = int(.1 * len(values))
    if filter_length % 2 == 0:
        filter_length += 1
    return scipy.signal.savgol_filter(values, filter_length, 3)

def smoothing_factor(y):
    return numpy.max(y)/1000000.0

if __name__ == "__main__":
    main()
