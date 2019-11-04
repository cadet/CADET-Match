"Cache module. Used to store per worker information so it does not need to be recalculated"

from pathlib import Path
import json
from cadet import Cadet

import numpy

import CADETMatch.plugins as plugins
import os
from deap import tools
from sklearn import preprocessing

import scoop

class Cache:
    def __init__(self):
        self.settings = None
        self.headers = None
        self.numGoals = None
        self.target = None
        self.MIN_VALUE = None
        self.MAX_VALUE = None
        self.badScore = None
        self.WORST = None
        self.json_path = None
        self.adaptive = None
        self.parameter_indexes = None
        self.score_indexes = None
        self.score_headers = None
        self.graphGenerateTime = None
        self.graphMetaTime = None
        self.lastGraphTime = None
        self.lastMetaTime = None
        self.metaResultsOnly = 1
        self.stallGenerations = 10
        self.stallCorrect = 5
        self.progressCorrect = 5
        self.lastProgressGeneration = -1
        self.generationsOfProgress = 0
        self.fullTrainingData = 0
        self.normalizeOutput = True
        self.sobolGeneration = True
        self.graphSpearman = False
        self.continueMCMC = False
        self.errorBias = True
        self.altFeatures = False
        self.altFeatureNames = []
        self.progress_headers = ['Generation', 'Population', 'Dimension In', 'Dimension Out', 'Search Method',
                                 'Pareto Front', 'Product Score', 'Minimum Score', 'Average Score',
                                 'Pareto Mean Product Score', 'Pareto Mean Minimum Score', 'Pareto Mean Average Score',
                                 'Pareto Meta Product Score', 'Pareto Meta Min Score', 'Pareto Meta Mean Score',
                                 'Elapsed Time', 'Generation Time', 'Total CPU Time', 'Last Progress Generation',
                                 'Generations of Progress']

    def setup(self, json_path, load_plugins=True):
        "setup the cache based on the json file being used"
        if load_plugins:
            self.scores = plugins.get_plugins('scores')
            self.search = plugins.get_plugins('search')
            self.transforms = plugins.get_plugins('transform')

        self.json_path = json_path
        self.setupSettings()

        self.abstolFactor = self.settings.get('abstolFactor', 1e-4)
        self.reltol = self.settings.get('reltol', 1e-4)
        self.abstolFactorGrad = self.settings.get('abstolFactorGrad', 1e-8)
        self.reltolGrad = self.settings.get('reltolGrad', 1e-8)

        self.errorBias = bool(self.settings.get('errorBias', True))

        baseDir = self.settings.get('baseDir', None)
        if baseDir is not None:
            os.chdir(baseDir)
            self.settings['resultsDir'] = Path(baseDir) / self.settings['resultsDir']

        Cadet.cadet_path = self.settings['CADETPath']

        self.normalizeOutput = bool(self.settings.get('normalizeOutput', False))
        self.connectionNumberEntries = int(self.settings.get('connectionNumberEntries', 5))

        self.setupHeaders()
        self.setupTarget()
        self.setupMinMax()

        self.WORST = [self.badScore] * self.numGoals

        self.settings['transform'] = self.transform
        #self.settings['grad_transform'] = self.grad_transform

        self.correct = None
        if "correct" in self.settings:
            self.correct = numpy.array([f(v) for f, v in zip(self.settings['transform'], self.settings['correct'])])

        #create used paths in settings, only the root process will make the directories later
        self.settings['resultsDirEvo'] = Path(self.settings['resultsDir']) / "evo"
        self.settings['resultsDirMeta'] = Path(self.settings['resultsDir']) / "meta"
        self.settings['resultsDirGrad'] = Path(self.settings['resultsDir']) / "grad"
        self.settings['resultsDirMisc'] = Path(self.settings['resultsDir']) / "misc"
        self.settings['resultsDirSpace'] = Path(self.settings['resultsDir']) / "space"
        self.settings['resultsDirProgress'] = Path(self.settings['resultsDir']) / "progress"
        self.settings['resultsDirLog'] = Path(self.settings['resultsDir']) / "log"
        self.settings['resultsDirMCMC'] = Path(self.settings['resultsDir']) / "mcmc"
        self.settings['resultsDirBase'] = Path(self.settings['resultsDir'])

        self.error_path = Path(self.settings['resultsDirBase'], "error.csv")

        self.graphGenerateTime = int(self.settings.get('graphGenerateTime', 3600))
        self.graphMetaTime = int(self.settings.get('graphMetaTime', 1200))

        self.metaResultsOnly = self.settings.get('metaResultsOnly', 0)
        self.stallGenerations = int(self.settings.get('stallGenerations', 10))
        self.stallCorrect = int(self.settings.get('stallCorrect', 5))
        self.progressCorrect = int(self.settings.get('progressCorrect', 5))

        self.fullTrainingData = int(self.settings.get('fullTrainingData', 0))

        self.sobolGeneration = bool(self.settings.get('soboloGeneration', False)) or bool(self.settings.get('sobolGeneration', False))
        self.graphSpearman = bool(self.settings.get('graphSpearman', False))
        self.scoreMCMC = self.settings.get('scoreMCMC', "sse")

        self.continueMCMC = bool(self.settings.get('continueMCMC', False))
        self.MCMCTauMult = int(self.settings.get('MCMCTauMult', 50))

        self.cross_eta = int(self.settings.get('cross_eta', 30))
        self.mutate_eta = int(self.settings.get('mutate_eta', 70))

        self.gradVector = bool(self.settings.get('gradVector', 0))

        self.tempDir = self.settings.get('tempDir', None)
        self.graphType = self.settings.get('graphType', 1)

        self.checkpointInterval = self.settings.get('checkpointInterval', 600)
        self.setupMetaMask()

        self.debugWrite = bool(self.settings.get('debugWrite', False))

        self.finalGradRefinement = bool(self.settings.get('finalGradRefinement', False))

        self.multiStartPercent = self.settings.get('multiStartPercent', 0.1)

        if "MCMCpopulation" not in self.settings:
            self.settings['MCMCpopulation'] = self.settings['population']


        if self.numGoals == 1:
            #with one goal one of the emo functions breaks, this is a temporary fix

            def sortNDHelperB(best, worst, obj, front):
                if obj < 0:
                    return
                sortNDHelperB(best, worst, obj, front)

            tools.emo.sortNDHelperB = sortNDHelperB

    def setupMetaMask(self):
        meta_mask_seq = []

        for idx, experiment in enumerate(self.settings['experiments']):
            for feature in experiment['features']:
                if feature['type'] in self.scores:
                    settings = self.scores[feature['type']].settings
                    meta_mask = settings.meta_mask
                    count = settings.count
                    #scoop.logger.info('%s %s %s %s', idx, feature, meta_mask, count)

                    meta_mask_seq.extend([meta_mask,] * count)
        #scoop.logger.info("%s", meta_mask_seq)
        self.meta_mask = numpy.array(meta_mask_seq)
        
    def setupSettings(self):
        settings_file = Path(self.json_path)
        with settings_file.open() as json_data:
            self.settings = json.load(json_data)

            if 'CSV' in self.settings:
                self.settings['csv'] = self.settings['CSV']
            if 'csv' not in self.settings:
                self.settings['csv'] = 'results.csv'

            if self.settings['searchMethod'] == 'Gradient':
                self.settings['population'] = 1
            else:
                self.settings['population'] = int(self.settings['population'])
            
            self.settings['maxPopulation'] = int(self.settings.get('maxPopulation', self.settings['population']))
            self.settings['minPopulation'] = int(self.settings.get('minPopulation', self.settings['population']))


            if "bootstrap" in self.settings:
                self.settings['bootstrap']['samples'] = int(self.settings['bootstrap']['samples'])

    def setupHeaders(self):
        self.headers = ['Time', 'Name', 'Method', 'Condition Number',]

        self.numGoals = 0
        self.badScore = 0.0

        base = len(self.headers)

        parameter_headers = []
        parameter_headers_actual = []
        
        for parameter in self.settings['parameters']:
            parameter_headers.extend(self.transforms[parameter['transform']].getHeaders(parameter))
            parameter_headers_actual.extend(self.transforms[parameter['transform']].getHeadersActual(parameter))

        self.parameter_headers = parameter_headers
        self.parameter_headers_actual = parameter_headers_actual

        self.headers.extend(parameter_headers)

        parameters = len(self.headers)
        self.parameter_indexes = list(range(base, parameters))

        self.score_headers = []

        badScore = []

        for idx, experiment in enumerate(self.settings['experiments']):
            experimentName = experiment['name']
            experiment['headers'] = []
            for feature in experiment['features']:
                if feature['type'] in self.scores:
                    temp = self.scores[feature['type']].headers(experimentName, feature)

                    if self.scores[feature['type']].settings.meta_mask:
                        self.numGoals += len(temp)
                    badScore.append(self.scores[feature['type']].settings.badScore)

                    self.score_headers.extend(temp)
                    experiment['headers'].extend(temp)

        self.badScore = min(badScore)

        self.headers.extend(self.score_headers)                      
        
        self.meta_headers = ['Product Root Score', 'Min Score', 'Mean Score', 'SSE']

        self.headers.extend(self.meta_headers)

        scores = len(self.headers)
        self.score_indexes = list(range(parameters, scores))

    def setupTarget(self):
        self.target = {}
        self.adaptive = True

        for experiment in self.settings['experiments']:
            self.target[experiment["name"]] = self.setupExperiment(experiment)
        self.target['bestHumanScores'] = numpy.ones(5) * self.badScore

        #SSE are negative so they sort correctly with better scores being less negative
        self.target['bestHumanScores'][4] = self.badScore;  

        #setup sensitivities
        parms = []
        sensitivityOk = 1
        for parameter in self.settings['parameters']:
            try:
                comp = parameter['component']
            except KeyError:
                sensitivityOk = 0
                break

            transform = parameter['transform']

            sens_parms, sensitivityOk = self.transforms[transform].setupTarget(parameter)
            parms.extend(sens_parms)

        if sensitivityOk:
            self.target['sensitivities'] = parms
        else:
            self.target['sensitivities'] = []

    def setupExperiment(self, experiment, sim=None, dataFromSim=0):
        temp = {}
        
        if sim is None:
            sim = Cadet()
            sim.filename = Path(experiment['HDF5']).as_posix()
            sim.load()

        abstol = sim.root.input.solver.time_integrator.abstol

        conn = sim.root.input.model.connections.switch_000.connections

        conn = numpy.array(conn)
        conn = numpy.reshape(conn, [-1, self.connectionNumberEntries])

        #find all the entries that connect to the column
        filter = conn[:, 1] == 1

        #flow is the sum of all flow rates that connect to this column which is in the last column
        flow = sum(conn[filter, -1])

        if sim.root.input.model.unit_001.unit_type == b'CSTR':
            volume = float(sim.root.input.model.unit_001.init_volume)

            CV_time = volume / flow

        else:
            #CV needs to be based on superficial velocity not interstitial velocity
            length = float(sim.root.input.model.unit_001.col_length)

            velocity = sim.root.input.model.unit_001.velocity
            if velocity == {}:
                velocity = 1.0
            velocity = float(velocity)

            area = sim.root.input.model.unit_001.cross_section_area
            if area == {}:
                area = 1.0
            area = float(area)

            if area == 1 and abs(velocity) != 1:
                CV_time = length / velocity
            else:
                CV_time = (area * length) / flow

        #force CSV to lowercase
        if 'CSV' in experiment:
            experiment['csv'] = experiment['CSV']

        if dataFromSim:
            temp['time'], temp['value'] = get_times_values(sim, {'isotherm':experiment['isotherm']})

        elif 'csv' in experiment:
            data = numpy.loadtxt(experiment['csv'], delimiter=',')

            temp['time'] = data[:, 0]
            temp['value'] = data[:, 1]

            if self.normalizeOutput:
                temp['factor'] = 1.0/numpy.max(temp['value'])
            else:
                temp['factor'] = 1.0
            temp['valueFactor'] = temp['value'] * temp['factor']

        if "featuresAlt" in experiment:
            self.altFeatures = True
            self.altFeatureNames = [altFeature['name'] for altFeature in experiment['featuresAlt']]

        peak_maxes = []
        for feature in experiment['features']:
            featureName = feature['name']
            featureType = feature['type']
            
            temp[featureName] = {}

            #switch to lower case
            if 'CSV' in feature:
                feature['csv'] = feature['CSV']

            if 'csv' in feature:
                if dataFromSim:
                    temp[featureName]['time'], temp[featureName]['value'] = get_times_values(sim, {'isotherm':feature['isotherm']})
                else:
                    dataLocal = numpy.loadtxt(feature['csv'], delimiter=',')
                    temp[featureName]['time'] = dataLocal[:, 0]
                    temp[featureName]['value'] = dataLocal[:, 1]
            else:
                temp[featureName]['time'] = temp['time']
                temp[featureName]['value'] = temp['value']

            if self.normalizeOutput:
                temp[featureName]['factor'] = 1.0/numpy.max(temp[featureName]['value'])
                temp[featureName]['value'] = temp[featureName]['value'] * temp[featureName]['factor']
            else:
                temp[featureName]['factor'] = 1.0

            try:
                featureStart = float(feature['start'])
                featureStop = float(feature['stop'])
            except KeyError:
                feature['start']= featureStart = temp[featureName]['time'][0]
                feature['stop'] = featureStop = temp[featureName]['time'][-1]

            if 'isotherm' in feature:
                temp[featureName]['isotherm'] = feature['isotherm']
            else:
                temp[featureName]['isotherm'] = experiment['isotherm']

            temp[featureName]['selected'] = (temp[featureName]['time'] >= featureStart) & (temp[featureName]['time'] <= featureStop)
            
            selectedTimes = temp[featureName]['time'][temp[featureName]['selected']]
            selectedValues = temp[featureName]['value'][temp[featureName]['selected']]

            if featureType in self.scores:
                temp[featureName].update(self.scores[featureType].setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol))
                self.adaptive = self.scores[featureType].settings.adaptive
                if 'peak_max' in temp[featureName]:
                    peak_maxes.append(temp[featureName]['peak_max']/temp[featureName]['factor'])
         
        temp['smallest_peak'] = min(peak_maxes)        
        return temp

    def setupMinMax(self):
        "build the minimum and maximum parameter boundaries"
        self.MIN_VALUE = []
        self.MAX_VALUE = []
        #self.MIN_VALUE_GRAD = []
        #self.MAX_VALUE_GRAD = []
        self.transform = []
        #self.grad_transform = []

        for parameter in self.settings['parameters']:
            transform = parameter['transform']
            minValues, maxValues = self.transforms[transform].getBounds(parameter)
            #minGradValues, maxGradValues = self.transforms[transform].getGradBounds(parameter)
            transforms = self.transforms[transform].transform(parameter)
            #grad_transforms = self.transforms[transform].grad_transform(parameter)

            if minValues:
                self.MIN_VALUE.extend(minValues)
                self.MAX_VALUE.extend(maxValues)
                self.transform.extend(transforms)
                #self.grad_transform.extend(grad_transforms)
                #self.MIN_VALUE_GRAD.extend(minGradValues)
                #self.MAX_VALUE_GRAD.extend(maxGradValues)

cache = Cache()


def get_times_values(simulation, target):
    "simplified version of the function so that util does not have to be imported"
    times = simulation.root.output.solution.solution_times
    isotherm = target['isotherm']

    if isinstance(isotherm, list):
        values = numpy.sum([simulation[i] for i in isotherm], 0)
    else:
        values = simulation[isotherm]
        
    return times, values
