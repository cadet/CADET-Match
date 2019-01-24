import sys

import matplotlib
matplotlib.use('Agg')

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
import scoop

import os
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import corner

saltIsotherms = {b'STERIC_MASS_ACTION', b'SELF_ASSOCIATION', b'MULTISTATE_STERIC_MASS_ACTION', 
                 b'SIMPLE_MULTISTATE_STERIC_MASS_ACTION', b'BI_STERIC_MASS_ACTION'}


from matplotlib.colors import ListedColormap
cmap = matplotlib.cm.winter

# Get the colormap colors
my_cmap = cmap(numpy.arange(cmap.N))

# Set alpha
my_cmap[:,-1] = 0.05

# Create new colormap
my_cmap = ListedColormap(my_cmap)

def main():
    cache.setup(sys.argv[1])
    cache.progress_path = Path(cache.settings['resultsDirBase']) / "progress.csv"

    scoop.logger.info("graphing directory %s", os.getcwd())

    fullGeneration = int(sys.argv[2])

    #full generation 1 = 2D, 2 = 2D + 3D

    graphMeta(cache)
    graphProgress(cache)
    graphCorner(cache)

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
        data = {}
        with h5py.File(result_h5, 'r') as h5:
            for key in h5.keys():
                data[key] = h5[key].value

            temp = []

            if 'mean' in data and 'confidence' in data and 'distance_correct' in data:
                for idx_parameter, parameter in enumerate(parameter_headers_actual):
                    for idx_score, score in enumerate(score_headers):
                        temp.append( (output_distance, parameter, idx_parameter, score, idx_score, data['distance_correct'][:,idx_parameter], data['output'][:,idx_score],
                                      data['mean'][:,idx_parameter], data['confidence'][:,idx_parameter]) )

                for idx_parameter, parameter in enumerate(parameter_headers_actual):
                    for idx_score, score in enumerate(meta_headers):
                        temp.append( (output_distance_meta, parameter, idx_parameter, score, idx_score, data['distance_correct'][:,idx_parameter], data['output_meta'][:,idx_score],
                                   data['mean'][:,idx_parameter], data['confidence'][:,idx_parameter]) )

                list(futures.map(plot_2d_scatter, temp))

def plot_2d_scatter(args):
    output_directory, parameter, parameter_idx, score, score_idx, parameter_data, score_data, mean_data, confidence_data = args
    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)
    graph.scatter(parameter_data, score_data, c=score_data, cmap=my_cmap)
    graph.set_xlabel(parameter)
    graph.set_ylabel(score)
    filename = "%s_%s.png" % (parameter_idx, score_idx)
    fig.savefig(str(output_directory / filename), bbox_inches='tight')

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)
    graph.scatter(mean_data, score_data, c=score_data, cmap=my_cmap)
    graph.set_xlabel(parameter)
    graph.set_ylabel(score)
    filename = "%s_%s_mean.png" % (parameter_idx, score_idx)
    fig.savefig(str(output_directory / filename), bbox_inches='tight')

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)
    graph.scatter(confidence_data, score_data, c=score_data, cmap=my_cmap)
    graph.set_xlabel(parameter)
    graph.set_ylabel(score)
    filename = "%s_%s_confidence.png" % (parameter_idx, score_idx)
    fig.savefig(str(output_directory / filename), bbox_inches='tight')


def graphCorner(cache):
    headers = list(cache.parameter_headers_actual)
    headers = [header.split()[0] for header in headers]
    
    resultDir = Path(cache.settings['resultsDir'])
    result_h5 = resultDir / "result.h5"

    miscDir = Path(cache.settings['resultsDirMCMC'])
    mcmc_h5 = miscDir / "mcmc.h5"

    if mcmc_h5.exists():
        data = {}
        with h5py.File(mcmc_h5, 'r') as h5:
            for key in h5.keys():
                data[key] = h5[key].value

        if len(data['flat_chain']) > 1e5:
            indexes = numpy.random.choice(data['flat_chain'].shape[0], int(1e5), replace=False)
            chain = data['flat_chain'][indexes]
            chain_transform = data['flat_chain_transform'][indexes]
        else:
            chain = data['flat_chain']
            chain_transform = data['flat_chain_transform']
        
        out_dir = cache.settings['resultsDirProgress']

        fig = corner.corner(chain, quantiles=(0.16, 0.5, 0.84),
                       show_titles=True, title_kwargs={"fontsize": 12}, labels=headers, bins=20)
        fig.savefig(str(out_dir / "corner.png"), bbox_inches='tight')

        fig = corner.corner(chain_transform, quantiles=(0.16, 0.5, 0.84),
                       show_titles=True, title_kwargs={"fontsize": 12}, labels=headers, bins=20)
        fig.savefig(str(out_dir / "corner_transform.png"), bbox_inches='tight')

        if 'burn_in_acceptance' in data:
            fig = figure.Figure(figsize=[10, 10])
            canvas = FigureCanvas(fig)
            graph = fig.add_subplot(1, 1, 1)
            graph.plot(data['burn_in_acceptance'])
            graph.set_title("Burn In Acceptance")
            graph.set_xlabel('Step')
            graph.set_ylabel('Acceptance')
            fig.savefig(str(out_dir / "burn_in_acceptance.png"), bbox_inches='tight')

        if 'mcmc_acceptance' in data:
            fig = figure.Figure(figsize=[10, 10])
            canvas = FigureCanvas(fig)
            graph = fig.add_subplot(1, 1, 1)
            graph.plot(data['mcmc_acceptance'])
            graph.set_title("MCMC Acceptance")
            graph.set_xlabel('Step')
            graph.set_ylabel('Acceptance')
            fig.savefig(str(out_dir / "mcmc_acceptance.png"), bbox_inches='tight')

    else:
        data = {}
        with h5py.File(result_h5, 'r') as h5:
            for key in h5.keys():
                data[key] = h5[key].value

        if len(data['input']) > 1e5:
            indexes = numpy.random.choice(data['input'].shape[0], int(1e5), replace=False)
            data_input = data['input'][indexes]
            data_input_transform = data['input_transform'][indexes]
            weight_min = data['output_meta'][indexes,1]
            weight_prod = data['output_meta'][indexes,2]
            weight_norm = data['output_meta'][indexes,3]
            all_scores = data['output'][indexes]
        else:
            data_input = data['input']
            data_input_transform = data['input_transform']
            weight_min = data['output_meta'][:,1]
            weight_prod = data['output_meta'][:,2]
            weight_norm = data['output_meta'][:,3]
            all_scores = data['output']

        max_scores = numpy.max(all_scores, 1)
        acceptable = max_scores > 0.01

        if numpy.any(acceptable):
            scoop.logger.info("graphing remove %s points", len(max_scores) - numpy.sum(acceptable))

            data_input = data_input[acceptable]
            data_input_transform = data_input_transform[acceptable]
            weight_min = weight_min[acceptable]
            weight_prod = weight_prod[acceptable]
            weight_norm = weight_norm[acceptable]

        out_dir = cache.settings['resultsDirProgress']

        create_corner(out_dir, "corner.png", headers, data_input, weights=None)
        create_corner(out_dir, "corner_min.png", headers, data_input, weights=weight_min)
        create_corner(out_dir, "corner_prod.png", headers, data_input, weights=weight_prod)
        create_corner(out_dir, "corner_norm.png", headers, data_input, weights=weight_norm)

        #transformed entries
        create_corner(out_dir, "corner_transform.png", headers, data_input_transform, weights=None)
        create_corner(out_dir, "corner_min_transform.png", headers, data_input_transform, weights=weight_min)
        create_corner(out_dir, "corner_prod_transform.png", headers, data_input_transform, weights=weight_prod)
        create_corner(out_dir, "corner_norm_transform.png", headers, data_input_transform, weights=weight_norm)

def create_corner(dir, filename, headers, data, weights=None):
    if  numpy.all(numpy.min(data,0) < numpy.max(data,0)):
        if weights is None or numpy.max(weights) > numpy.min(weights):
            fig = corner.corner(data, quantiles=(0.16, 0.5, 0.84), weights=weights,
                show_titles=True, title_kwargs={"fontsize": 12}, labels=headers, bins=20)
            fig.savefig(str(dir / filename), bbox_inches='tight')

def graphExperiments(cache):
    directory = Path(cache.settings['resultsDirEvo'])

    #find all items in directory
    pathsH5 = directory.glob('*.h5')
    pathsPNG = directory.glob('*.png')

    #make set of items based on removing everything after _
    existsH5 = {str(path.name).split('_', 1)[0] for path in pathsH5}
    existsPNG = {str(path.name).split('_', 1)[0] for path in pathsPNG}

    toGenerate = existsH5 - existsPNG

    list(futures.map(plotExperiments, toGenerate, itertools.repeat(sys.argv[1]), itertools.repeat(cache.settings['resultsDirEvo']), itertools.repeat('%s_%s_EVO.png')))

def graphMeta(cache):
    directory = Path(cache.settings['resultsDirMeta'])

    #find all items in directory
    pathsH5 = directory.glob('*.h5')
    pathsPNG = directory.glob('*.png')

    #make set of items based on removing everything after _
    existsH5 = {str(path.name).split('_', 1)[0] for path in pathsH5}
    existsPNG = {str(path.name).split('_', 1)[0] for path in pathsPNG}

    toGenerate = existsH5 - existsPNG

    list(futures.map(plotExperiments, toGenerate, itertools.repeat(sys.argv[1]), itertools.repeat(cache.settings['resultsDirMeta']), itertools.repeat('%s_%s_meta.png')))


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

        numPlotsSeq = [1]
        #Shape and ShapeDecay have a chromatogram + derivative
        for feature in experiment['features']:
            if feature['type'] in ('Shape', 'ShapeDecay'):
                numPlotsSeq.append(2)
            else:
                numPlotsSeq.append(1)

        numPlots = sum(numPlotsSeq)

        exp_time = target[experimentName]['time']
        exp_value = target[experimentName]['value']

        fig = figure.Figure(figsize=[10, numPlots*10])
        canvas = FigureCanvas(fig)

        simulation = Cadet()
        h5_path = Path(directory) / (file_pattern.replace('png', 'h5') % (save_name_base, experimentName))
        simulation.filename = str(h5_path)
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
                               'Shape', 'ShapeDecay', 'Dextran'):
                
                graph = fig.add_subplot(numPlots, 1, graphIdx) #additional +1 added due to the overview plot
                graph.plot(sim_time, sim_value, 'r--', label='Simulation')
                graph.plot(exp_time, exp_value, 'g:', label='Experiment')
                graphIdx += 1
            
            if featureType in ('derivative_similarity', 'derivative_similarity_hybrid', 'derivative_similarity_hybrid2', 'derivative_similarity_cross', 'derivative_similarity_cross_alt',
                                 'derivative_similarity_hybrid2_spline', 'similarityHybridDecay2_spline',
                                 'Shape', 'ShapeDecay'):
                #try:
                sim_spline = scipy.interpolate.UnivariateSpline(sim_time, smoothing(sim_time, sim_value), s=smoothing_factor(sim_value)).derivative(1)
                exp_spline = scipy.interpolate.UnivariateSpline(exp_time, smoothing(exp_time, exp_value), s=smoothing_factor(exp_value)).derivative(1)

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

                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

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
                    graph.plot(time, values, '%s--' % colors[idx], label='Simulation Comp: %s Mult:%.2f' % (key, mult))

                for idx, (key, value) in enumerate(graph_exp.items()):
                    (time, values) = zip(*value)
                    mult = factors[idx]
                    values = [i* mult for i in values]
                    graph.plot(time, values, '%s:' % colors[idx], label='Experiment Comp: %s Mult:%.2f' % (key, mult))
                graphIdx += 1
            graph.legend()

        fig.savefig(str(dst))

def graphSpace(fullGeneration, cache):
    csv_path = Path(cache.settings['resultsDirBase']) / cache.settings['CSV']
    output_2d = cache.settings['resultsDirSpace'] / "2d"
    output_3d = cache.settings['resultsDirSpace'] / "3d"

    output_2d.mkdir(parents=True, exist_ok=True)
    output_3d.mkdir(parents=True, exist_ok=True)

    comp_two = list(itertools.combinations(cache.parameter_indexes, 2))
    comp_one = list(itertools.combinations(cache.parameter_indexes, 1))

    #3d plots
    if fullGeneration >= 2:
        prod = list(itertools.product(comp_two, cache.score_indexes))
        seq = [(str(output_3d), str(csv_path), i[0][0], i[0][1], i[1], sys.argv[1]) for i in prod]
        list(futures.map(plot_3d, seq))
    
    #2d plots
    if fullGeneration >= 1:
        graphDistance(cache)

        prod = list(itertools.product(comp_one, cache.score_indexes))
        seq = [(str(output_2d), str(csv_path), i[0][0], i[1], sys.argv[1]) for i in prod]
        list(futures.map(plot_2d, seq))

def plot_3d(arg):
    "This leaks memory and is run in a separate short-lived process, do not integrate into the matching main process"
    directory_path, csv_path, c1, c2, score, json_path = arg

    if json_path != cache.json_path:
        cache.setup(json_path)

    dataframe = pandas.read_csv(csv_path)
    directory = Path(directory_path)

    headers = dataframe.columns.values.tolist()

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
    ax.scatter(numpy.log(x), numpy.log(y), scores, c=scores, cmap=my_cmap)
    ax.set_xlabel('log(%s)' % headers[c1])
    ax.set_ylabel('log(%s)' % headers[c2])
    ax.set_zlabel(scoreName)
    filename = "%s_%s_%s.png" % (c1, c2, score)
    fig.savefig(str(directory / filename), bbox_inches='tight')
    
def plot_2d(arg):
    directory_path, csv_path, c1, score, json_path = arg

    if json_path != cache.json_path:
        cache.setup(json_path)

    dataframe = pandas.read_csv(csv_path)
    directory = Path(directory_path)
    headers = dataframe.columns.values.tolist()

    scores = dataframe.iloc[:, score]
    scoreName = headers[score]
    if headers[score] == 'SSE':
        scores = -numpy.log(scores)
        scoreName = '-log(%s)' % headers[score]

    fig = figure.Figure()
    canvas = FigureCanvas(fig)
    graph = fig.add_subplot(1, 1, 1)

    data = dataframe.iloc[:, c1]
    format = '%s'
    if numpy.max(data)/numpy.min(data) > 100.0:
        data = numpy.log(data)
        format = 'log(%s)'

    graph.scatter(data, scores, c=scores, cmap=my_cmap)
    graph.set_xlabel(format % headers[c1])
    graph.set_ylabel(scoreName)
    filename = "%s_%s.png" % (c1, score)
    fig.savefig(str(directory / filename), bbox_inches='tight')

def graphProgress(cache):
    results = Path(cache.settings['resultsDirBase'])
    
    df = pandas.read_csv(str(cache.progress_path))

    hof = results / "meta_hof.npy"

    with hof.open('rb') as hof_file:
        data = numpy.load(hof_file)

    output = cache.settings['resultsDirProgress']

    x = ['Generation', 'Total CPU Time', 'Population']
    y = ['Average Score', 'Minimum Score', 'Product Score',
         'Pareto Mean Average Score', 'Pareto Mean Minimum Score', 'Pareto Mean Product Score', 'Population']

    list(futures.map(singleGraphProgress, itertools.product(x,y), itertools.repeat(df), itertools.repeat(output), itertools.repeat(sys.argv[1])))

def singleGraphProgress(tup, df, output, json_path):
    if json_path != cache.json_path:
        cache.setup(json_path)

    i,j = tup

    fig = figure.Figure()
    canvas = FigureCanvas(fig)

    graph = fig.add_subplot(1, 1, 1)

    graph.plot(df[i],df[j])
    graph.set_ylim((0,1.1*max(df[j])))
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
