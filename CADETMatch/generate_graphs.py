import sys

import matplotlib
matplotlib.use('Agg')

size = 20

matplotlib.rc('font', size=size)          # controls default text sizes
matplotlib.rc('axes', titlesize=size)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=size)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=size)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=size)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=size)    # legend fontsize
matplotlib.rc('figure', titlesize=size)  # fontsize of the figure title
matplotlib.rc('figure', autolayout=True)

from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  

from cache import cache

from pathlib import Path
import pandas
import numpy
import scipy.interpolate
import itertools

from cadet import Cadet
from cadet import H5
from addict import Dict

#parallelization
from scoop import futures
import scoop

import os
import warnings
import corner
import util
import logging
import loggerwriter

saltIsotherms = {b'STERIC_MASS_ACTION', b'SELF_ASSOCIATION', b'MULTISTATE_STERIC_MASS_ACTION', 
                 b'SIMPLE_MULTISTATE_STERIC_MASS_ACTION', b'BI_STERIC_MASS_ACTION'}


from matplotlib.colors import ListedColormap
import matplotlib.cm
cmap = matplotlib.cm.winter

# Get the colormap colors
my_cmap = cmap(numpy.arange(cmap.N))

# Set alpha
my_cmap[:,-1] = 1.0

# Create new colormap
my_cmap = ListedColormap(my_cmap)

cm_plot = matplotlib.cm.gist_rainbow

if getattr(scoop, 'SIZE', 1) == 1:
    map_function = map
else:
    map_function = futures.map

def setupLog(log_directory):
    logger = scoop.logger
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_directory / "graph.log")
    fh.setLevel(logging.INFO)

    # add the handlers to the logger
    logger.addHandler(fh)

    sys.stdout = loggerwriter.LoggerWriter(logger.info)
    sys.stderr = loggerwriter.LoggerWriter(logger.warning)

def get_color(idx, max_colors, cmap):
    return cmap(1.*float(idx)/max_colors)

def main():
    cache.setup(sys.argv[1])
    
    setupLog(cache.settings['resultsDirLog'])

    scoop.logger.info("graphing directory %s", os.getcwd())

    fullGeneration = int(sys.argv[2])

    #full generation 1 = 2D, 2 = 2D + 3D

    graphMeta(cache)
    graphProgress(cache)

    if fullGeneration:
        graphSpace(fullGeneration, cache)
        graphExperiments(cache)    

def graphDistance(cache):

    resultDir = Path(cache.settings['resultsDir'])
    result_h5 = resultDir / "result.h5"

    output_distance = cache.settings['resultsDirSpace'] / "distance"
    output_distance_meta = output_distance / "meta"

    output_distance.mkdir(parents=True, exist_ok=True)

    output_distance_meta.mkdir(parents=True, exist_ok=True)

    parameter_headers_actual = cache.parameter_headers_actual
    score_headers = cache.score_headers
    meta_headers = cache.meta_headers

    if result_h5.exists():
        data = H5()
        data.filename = result_h5.as_posix()
        data.load()

        temp = []

        if 'mean' in data.root and 'confidence' in data.root and 'distance_correct' in data.root:
            for idx_parameter, parameter in enumerate(parameter_headers_actual):
                for idx_score, score in enumerate(score_headers):
                    temp.append( (output_distance, parameter, idx_parameter, score, idx_score, data.root.distance_correct[:,idx_parameter], data.root.output[:,idx_score],
                                    data.root.mean[:,idx_parameter], data.root.confidence[:,idx_parameter]) )

            for idx_parameter, parameter in enumerate(parameter_headers_actual):
                for idx_score, score in enumerate(meta_headers):
                    temp.append( (output_distance_meta, parameter, idx_parameter, score, idx_score, data.root.distance_correct[:,idx_parameter], data['output_meta'][:,idx_score],
                                data.root.mean[:,idx_parameter], data.root.confidence[:,idx_parameter]) )

            list(map_function(plot_2d_scatter, temp))

def plot_2d_scatter(args):
    output_directory, parameter, parameter_idx, score, score_idx, parameter_data, score_data, mean_data, confidence_data = args
    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)
    graph.scatter(parameter_data, score_data, c=score_data, cmap=my_cmap)
    graph.set_xlabel(parameter)
    graph.set_ylabel(score)
    filename = "%s_%s.png" % (parameter_idx, score_idx)
    fig.savefig(str(output_directory / filename))

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)
    graph.scatter(mean_data, score_data, c=score_data, cmap=my_cmap)
    graph.set_xlabel(parameter)
    graph.set_ylabel(score)
    filename = "%s_%s_mean.png" % (parameter_idx, score_idx)
    fig.savefig(str(output_directory / filename))

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)
    graph.scatter(confidence_data, score_data, c=score_data, cmap=my_cmap)
    graph.set_xlabel(parameter)
    graph.set_ylabel(score)
    filename = "%s_%s_confidence.png" % (parameter_idx, score_idx)
    fig.savefig(str(output_directory / filename))

def graphExperiments(cache):
    directory = Path(cache.settings['resultsDirEvo'])

    #find all items in directory
    pathsH5 = directory.glob('*.h5')
    pathsPNG = directory.glob('*.png')

    #make set of items based on removing everything after _
    existsH5 = {str(path.name).split('_', 1)[0] for path in pathsH5}
    existsPNG = {str(path.name).split('_', 1)[0] for path in pathsPNG}

    toGenerate = existsH5 - existsPNG

    list(map_function(plotExperiments, toGenerate, itertools.repeat(sys.argv[1]), itertools.repeat(cache.settings['resultsDirEvo']), itertools.repeat('%s_%s_EVO.png')))

def graphMeta(cache):
    directory = Path(cache.settings['resultsDirMeta'])

    #find all items in directory
    pathsH5 = directory.glob('*.h5')
    pathsPNG = directory.glob('*.png')

    #make set of items based on removing everything after _
    existsH5 = {str(path.name).split('_', 1)[0] for path in pathsH5}
    existsPNG = {str(path.name).split('_', 1)[0] for path in pathsPNG}

    toGenerate = existsH5 - existsPNG

    list(map_function(plotExperiments, toGenerate, itertools.repeat(sys.argv[1]), itertools.repeat(cache.settings['resultsDirMeta']), itertools.repeat('%s_%s_meta.png')))


def graph_simulation(simulation, graph):
    ncomp = int(simulation.root.input.model.unit_001.ncomp)
    isotherm = bytes(simulation.root.input.model.unit_001.adsorption_model)

    hasSalt = isotherm in saltIsotherms

    solution_times = simulation.root.output.solution.solution_times

    comps = []

    hasColumn = any('column' in i for i in simulation.root.output.solution.unit_001.keys())
    hasPort = any('port' in i for i in simulation.root.output.solution.unit_001.keys())
    #hasColumn = isinstance(simulation.root.output.solution.unit_001.solution_outlet_comp_000, Dict)

    if hasColumn:
        for i in range(ncomp):
            comps.append(simulation.root.output.solution.unit_001['solution_column_outlet_comp_%03d' % i])
    elif hasPort:
        for i in range(ncomp):
            comps.append(simulation.root.output.solution.unit_001['solution_outlet_port_000_comp_%03d' % i])
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

        #colors = ['r', 'g', 'c', 'm', 'y', 'k']
        axis2 = graph.twinx()
        for idx, comp in enumerate(comps[1:]):
            axis2.plot(solution_times, comp, '-', color=get_color(idx, len(comps) - 1, cm_plot), label="P%s" % idx)
        axis2.set_ylabel('mMol Protein', color='r')
        axis2.tick_params('y', colors='r')


        lines, labels = graph.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis2.legend(lines + lines2, labels + labels2, loc=0)
    else:
        graph.set_title("Output")
        
        #colors = ['r', 'g', 'c', 'm', 'y', 'k']
        for idx, comp in enumerate(comps):
            graph.plot(solution_times, comp, '-', color=get_color(idx, len(comps), cm_plot), label="P%s" % idx)
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

        numPlotsSeq = [1]
        #Shape and ShapeDecay have a chromatogram + derivative
        for feature in experiment['features']:
            if feature['type'] in ('Shape', 'ShapeDecay'):
                numPlotsSeq.append(2)
            elif feature['type'] in ('AbsoluteTime', 'AbsoluteHeight'):
                pass
            else:
                numPlotsSeq.append(1)

        numPlots = sum(numPlotsSeq)

        exp_time = target[experimentName]['time']
        exp_value = target[experimentName]['value']

        fig = figure.Figure(figsize=[10, numPlots*10])
        canvas = FigureCanvas(fig)

        simulation = Cadet()
        h5_path = Path(directory) / (file_pattern.replace('png', 'h5') % (save_name_base, experimentName))
        simulation.filename = h5_path.as_posix()
        simulation.load()

        results = {}
        results['simulation'] = simulation

        graph_simulation(simulation, fig.add_subplot(numPlots, 1, 1))

        graphIdx = 2
        for idx, feature in enumerate(experiment['features']):
            featureName = feature['name']
            featureType = feature['type']

            feat = target[experimentName][featureName]

            selected = feat['selected']
            exp_time = feat['time'][selected]
            exp_value = feat['value'][selected] / feat['factor']

            sim_time, sim_value = get_times_values(simulation, target[experimentName][featureName])
            

            if featureType in ('similarity', 'similarityDecay', 'similarityHybrid', 'similarityHybrid2', 'similarityHybrid2_spline', 'similarityHybridDecay', 
                               'similarityHybridDecay2', 'curve', 'breakthrough', 'dextran', 'dextranHybrid', 'dextranHybrid2', 'dextranHybrid2_spline',
                               'similarityCross', 'similarityCrossDecay', 'breakthroughCross', 'SSE', 'LogSSE', 'breakthroughHybrid', 'breakthroughHybrid2',
                               'Shape', 'ShapeDecay', 'Dextran', 'DextranAngle', 'DextranTest', 'DextranQuad',
                               'Dextran3', 'DextranShape', 'ShapeDecaySimple', 'ShapeSimple', 'DextranSSE'):
                
                graph = fig.add_subplot(numPlots, 1, graphIdx) #additional +1 added due to the overview plot
                graph.plot(sim_time, sim_value, 'r--', label='Simulation')
                graph.plot(exp_time, exp_value, 'g:', label='Experiment')
                graphIdx += 1
            
            if featureType in ('derivative_similarity', 'derivative_similarity_hybrid', 'derivative_similarity_hybrid2', 'derivative_similarity_cross', 'derivative_similarity_cross_alt',
                                 'derivative_similarity_hybrid2_spline', 'similarityHybridDecay2_spline',
                                 'Shape', 'ShapeDecay'):
                #try:
                sim_spline = util.create_spline(sim_time, sim_value).derivative(1)
                exp_spline = util.create_spline(exp_time, exp_value).derivative(1)
                
                graph = fig.add_subplot(numPlots, 1, graphIdx) #additional +1 added due to the overview plot
                graph.plot(sim_time, sim_spline(sim_time), 'r--', label='Simulation')
                graph.plot(exp_time, exp_spline(exp_time), 'g:', label='Experiment')
                graphIdx += 1
                #except:
                #    pass
            
            if featureType in ('fractionation', 'fractionationCombine', 'fractionationMeanVariance', 'fractionationMoment', 'fractionationSlide'):
                cache.scores[featureType].run(results, feat)


                graph_exp = results['graph_exp']
                graph_sim = results['graph_sim']

                #colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

                findMax = 0
                for idx, (key, value) in enumerate(graph_sim.items()):
                    (time, values) = zip(*value)
                    findMax = max(findMax, max(values))

                for idx, (key, value) in enumerate(graph_exp.items()):
                    (time, values) = zip(*value)
                    findMax = max(findMax, max(values))

                graph = fig.add_subplot(numPlots, 1, graphIdx) #additional +1 added due to the overview plot
                factors = []
                for idx, (key, value) in enumerate(graph_sim.items()):
                    (time, values) = zip(*value)
                    mult = 1.0
                    if max(values) < .2 * findMax:
                        mult = findMax/(2*max(values))
                        values = [i* mult for i in values]
                    factors.append(mult)
                    graph.plot(time, values, '--', color=get_color(idx, len(graph_sim), cm_plot), label='Simulation Comp: %s Mult:%.2f' % (key, mult))

                for idx, (key, value) in enumerate(graph_exp.items()):
                    (time, values) = zip(*value)
                    mult = factors[idx]
                    values = [i* mult for i in values]
                    graph.plot(time, values, ':', color=get_color(idx, len(graph_sim), cm_plot), label='Experiment Comp: %s Mult:%.2f' % (key, mult))
                graphIdx += 1
            graph.legend()
        fig.set_size_inches((12,12))
        fig.savefig(str(dst))

def graphSpace(fullGeneration, cache):
    progress_path = Path(cache.settings['resultsDirBase']) / "result.h5"

    output_2d = cache.settings['resultsDirSpace'] / "2d"
    output_3d = cache.settings['resultsDirSpace'] / "3d"

    output_2d.mkdir(parents=True, exist_ok=True)
    output_3d.mkdir(parents=True, exist_ok=True)
    
    results = H5()
    results.filename = progress_path.as_posix()

    results.load(paths=['/output', '/output_meta', '/is_extended_input'])

    if results.root.is_extended_input:
        results.load(paths=['/input_transform_extended'], update=True)
        input = results.root.input_transform_extended
    else:
        results.load(paths=['/input_transform'], update=True)
        input = results.root.input_transform

    output = results.root.output
    output_meta = results.root.output_meta

    input_headers = cache.parameter_headers
    output_headers = cache.score_headers
    meta_headers = cache.meta_headers

    input_indexes = list(range(len(input_headers)))
    output_indexes = list(range(len(output_headers)))
    meta_indexes = list(range(len(meta_headers)))

    comp_two = list(itertools.combinations(input_indexes, 2))
    comp_one = list(itertools.combinations(input_indexes, 1))

    #2d plots
    if fullGeneration >= 1:
        graphDistance(cache)

        seq = []
        for (x,), y in itertools.product(comp_one, output_indexes):
            seq.append( [output_2d.as_posix(), input_headers[x], output_headers[y], input[:,x], output[:,y]] )

        for (x,), y in itertools.product(comp_one, meta_indexes):
            seq.append( [output_2d.as_posix(), input_headers[x], meta_headers[y], input[:,x], output_meta[:,y]] )
        list(map_function(plot_2d, seq))

    #3d plots
    if fullGeneration >= 2:
        seq = []
        for (x, y), z in itertools.product(comp_two, output_indexes):
            seq.append( [output_3d.as_posix(), input_headers[x], input_headers[y], output_headers[z], input[:,x], input[:,y], output[:,z]] )

        for (x, y), z in itertools.product(comp_two, meta_indexes):
            seq.append( [output_3d.as_posix(), input_headers[x], input_headers[y], meta_headers[z], input[:,x], input[:,y], output_meta[:,z]] )
        list(map_function(plot_3d, seq))

def plot_3d(arg):
    "This leaks memory and is run in a separate short-lived process, do not integrate into the matching main process"
    directory_path, header1, header2, header3, data1, data2, scores = arg
    directory = Path(directory_path)

    scoreName = header3
    if scoreName == 'SSE':
        scores = -numpy.log(scores)
        scoreName = '-log(%s)' % scoreName
    
    x = data1
    y = data2

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(numpy.log(x), numpy.log(y), scores, c=scores, cmap=my_cmap)
    ax.set_xlabel('log(%s)' % header1)
    ax.set_ylabel('log(%s)' % header2)
    ax.set_zlabel(scoreName)
    filename = "%s_%s_%s.png" % (header1, header2, scoreName)
    filename = filename.replace(':', '_')
    fig.savefig(str(directory / filename))
    
def plot_2d(arg):
    directory_path, header_x, scoreName, data, scores = arg
    directory = Path(directory_path)

    if numpy.all(scores <= 0):
        scores = numpy.abs(scores)

    score_scale = numpy.max(scores)/numpy.min(scores)

    if  score_scale > 1e2:
        scores = numpy.log(scores)
        scoreName = 'log(%s)' % scoreName

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)

    format = '%s'
    if numpy.max(data)/numpy.min(data) > 100.0:
        data = numpy.log(data)
        format = 'log(%s)'

    graph.scatter(data, scores, c=scores, cmap=my_cmap)
    graph.set_xlabel(format % header_x)
    graph.set_ylabel(scoreName)
    graph.set_xlim(min(data), max(data), auto=True)
    filename = "%s_%s.png" % (header_x, scoreName)
    filename = filename.replace(':', '_').replace('/', '_')
    fig.savefig(str(directory / filename))

def graphProgress(cache):
    results = Path(cache.settings['resultsDirBase'])
    progress = results / "progress.csv"
    
    df = pandas.read_csv(progress)

    output = cache.settings['resultsDirProgress']

    x = ['Generation',]
    y = ['Average Score', 'Minimum Score', 'Product Score',
         'Pareto Meta Product Score', 'Pareto Meta Min Score', 
         'Pareto Meta Mean Score',]

    temp = []
    for x,y in itertools.product(x,y):
        if x != y:
            temp.append( (df[[x,y]], output) )

    list(map_function(singleGraphProgress, temp))

def singleGraphProgress(arg):
    df, output = arg

    i,j = list(df)

    fig = figure.Figure(figsize=(10,10))
    canvas = FigureCanvas(fig)

    graph = fig.add_subplot(1, 1, 1)

    graph.plot(df[i],df[j])
    a = max(df[j])
    graph.set_ylim((0,1.1*max(df[j])))
    graph.set_title('%s vs %s' % (i,j))
    graph.set_xlabel(i)
    graph.set_ylabel(j)

    filename = "%s vs %s.png" % (i,j)
    file_path = output / filename
    fig.savefig(str(file_path))



    fig = figure.Figure(figsize=(10,10))
    canvas = FigureCanvas(fig)

    graph = fig.add_subplot(1, 1, 1)

    graph.plot(df[i],numpy.log(1-df[j]))
    a = max(df[j])
    #graph.set_ylim((0,1.1*max(df[j])))
    graph.set_title('%s vs log(1-%s)' % (i,j))
    graph.set_xlabel(i)
    graph.set_ylabel('log(1-%s)' % j)

    filename = "%s vs log(1-%s).png" % (i,j)
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
