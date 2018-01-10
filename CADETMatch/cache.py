"Cache module. Used to store per worker information so it does not need to be recalculated"

from pathlib import Path
import json
from cadet import Cadet

import numpy

import plugins

class Cache:
    def __init__(self):
        self.scores = plugins.get_plugins('scores')
        self.search = plugins.get_plugins('search')
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
        self.transform = None
        self.parameter_indexes = None
        self.score_indexes = None
        self.progress_headers = ['Generation', 'Population', 'Dimension In', 'Dimension Out', 'Search Method',
                                 'Pareto Front', 'Hypervolume', 'Average Score', 'Minimum Score', 'Product Score',
                                 'Elapsed Time', 'Generation Time']

    def setup(self, json_path):
        "setup the cache based on the json file being used"
        self.json_path = json_path
        self.setupSettings()
        Cadet.cadet_path = self.settings['CADETPath']

        self.setupHeaders()
        self.setupTarget()
        self.setupMinMax()

        self.WORST = [self.badScore] * self.numGoals

        self.settings['transform'] = self.transform

        #create used paths in settings, only the root process will make the directories later
        self.settings['resultsDirEvo'] = Path(self.settings['resultsDir']) / "evo"
        self.settings['resultsDirGrad'] = Path(self.settings['resultsDir']) / "grad"
        self.settings['resultsDirMisc'] = Path(self.settings['resultsDir']) / "misc"
        self.settings['resultsDirSpace'] = Path(self.settings['resultsDir']) / "space"
        self.settings['resultsDirBase'] = Path(self.settings['resultsDir'])

    def setupSettings(self):
        settings_file = Path(self.json_path)
        with settings_file.open() as json_data:
            self.settings = json.load(json_data)

            self.settings['population'] = int(self.settings['population'])

            if "bootstrap" in self.settings:
                self.settings['bootstrap']['samples'] = int(self.settings['bootstrap']['samples'])

    def setupHeaders(self):
        self.headers = ['Time', 'Name', 'Method', 'Condition Number',]

        self.numGoals = 0
        self.badScore = 0.0
        base = len(self.headers)
        
        for parameter in self.settings['parameters']:
            try:
                comp = parameter['component']
            except KeyError:
                comp = 'None'
            if parameter['transform'] == 'keq':
                location = parameter['location']
                nameKA = location[0].rsplit('/', 1)[-1]
                nameKD = location[1].rsplit('/', 1)[-1]
                for bound in parameter['bound']:
                    self.headers.append("%s Comp:%s Bound:%s" % (nameKA, comp, bound))
                    self.headers.append("%s Comp:%s Bound:%s" % (nameKD, comp, bound))
                    self.headers.append("%s/%s Comp:%s Bound:%s" % (nameKA, nameKD, comp, bound))
            elif parameter['transform'] == 'log':
                location = parameter['location']
                name = location.rsplit('/', 1)[-1]
                for bound in parameter.get('bound', []):
                    self.headers.append("%s Comp:%s Bound:%s" % (name, comp, bound))
                for idx in parameter.get('indexes', []):
                    self.headers.append("%s Comp:%s Index:%s" % (name, comp, idx))

        parameters = len(self.headers)
        self.parameter_indexes = list(range(base, parameters))

        for idx, experiment in enumerate(self.settings['experiments']):
            experimentName = experiment['name']
            experiment['headers'] = []
            for feature in experiment['features']:
                if feature['type'] in self.scores:
                    temp = self.scores[feature['type']].headers(experimentName, feature)
                    self.numGoals += len(temp)
                    self.badScore = self.scores[feature['type']].badScore

                self.headers.extend(temp)
                experiment['headers'].extend(temp)
                               
        self.headers.extend(['Product Root Score', 'Min Score', 'Mean Score', 'Norm', 'SSE'])

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

            if transform == 'keq':
                location = parameter['location']
                nameKA = location[0].rsplit('/', 1)[-1]
                nameKD = location[1].rsplit('/', 1)[-1]
                unit = int(location[0].split('/')[3].replace('unit_', ''))

                for bound in parameter['bound']:
                    parms.append((nameKA, unit, comp, bound))
                    parms.append((nameKD, unit, comp, bound))

            elif transform == 'log':
                location = parameter['location']
                name = location.rsplit('/', 1)[-1]
                try:
                    unit = int(location.split('/')[3].replace('unit_', ''))
                except ValueError:
                    unit = ''
                    sensitivityOk = 0

                for bound in parameter['bound']:
                    parms.append((name, unit, comp, bound))

        if sensitivityOk:
            self.target['sensitivities'] = parms
        else:
            self.target['sensitivities'] = []

    def setupExperiment(self, experiment):
        temp = {}

        sim = Cadet()
        sim.filename = Path(experiment['HDF5'])
        sim.load()

        abstol = sim.root.input.solver.time_integrator.abstol

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

        porosity = sim.root.input.model.unit_001.col_porosity
        if porosity == {}:
            porosity = sim.root.input.model.unit_001.total_porosity
        if porosity == {}:
            porosity = 1.0
        porosity = float(porosity)

        conn = sim.root.input.model.connections.switch_000.connections

        conn = numpy.array(conn)
        conn = numpy.reshape(conn, [-1, 5])

        #find all the entries that connect to the column
        filter = conn[:, 1] == 1

        #flow is the sum of all flow rates that connect to this column which is in the last column
        flow = sum(conn[filter, -1])

        if area == 1 and abs(velocity) != 1:
            CV_time = length / velocity
        else:
            CV_time = (area * length) / flow

        if 'CSV' in experiment:
            data = numpy.genfromtxt(experiment['CSV'], delimiter=',')

            temp['time'] = data[:, 0]
            temp['value'] = data[:, 1]

        for feature in experiment['features']:
            featureName = feature['name']
            featureType = feature['type']
            featureStart = float(feature['start'])
            featureStop = float(feature['stop'])

            temp[featureName] = {}

            if 'CSV' in feature:
                dataLocal = numpy.genfromtxt(feature['CSV'], delimiter=',')
                temp[featureName]['time'] = dataLocal[:, 0]
                temp[featureName]['value'] = dataLocal[:, 1]
            else:
                temp[featureName]['time'] = data[:, 0]
                temp[featureName]['value'] = data[:, 1]

            if 'isotherm' in feature:
                temp[featureName]['isotherm'] = feature['isotherm']
            else:
                temp[featureName]['isotherm'] = experiment['isotherm']

            temp[featureName]['selected'] = (temp[featureName]['time'] >= featureStart) & (temp[featureName]['time'] <= featureStop)
            
            selectedTimes = temp[featureName]['time'][temp[featureName]['selected']]
            selectedValues = temp[featureName]['value'][temp[featureName]['selected']]

            if featureType in self.scores:
                temp[featureName].update(self.scores[featureType].setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol))
                self.adaptive = self.scores[featureType].adaptive
            
        return temp

    def setupMinMax(self):
        "build the minimum and maximum parameter boundaries"
        self.MIN_VALUE = []
        self.MAX_VALUE = []
        self.transform = []

        for parameter in self.settings['parameters']:
            transform = parameter['transform']
            location = parameter['location']

            if transform == 'keq':
                minKA = parameter['minKA']
                maxKA = parameter['maxKA']
                minKEQ = parameter['minKEQ']
                maxKEQ = parameter['maxKEQ']

                minValues = [item for pair in zip(minKA, minKEQ) for item in pair]
                maxValues = [item for pair in zip(maxKA, maxKEQ) for item in pair]

                self.transform.extend([numpy.log for i in minValues])

                minValues = numpy.log(minValues)
                maxValues = numpy.log(maxValues)

            elif transform == 'log':
                minValues = numpy.log(parameter['min'])
                maxValues = numpy.log(parameter['max'])
                self.transform.append(numpy.log)

            self.MIN_VALUE.extend(minValues)
            self.MAX_VALUE.extend(maxValues)

cache = Cache()
