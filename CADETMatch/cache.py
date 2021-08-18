"Cache module. Used to store per worker information so it does not need to be recalculated"

import multiprocessing
import os
import sys
from pathlib import Path

import jstyleson
import numpy
from cadet import Cadet
from sklearn import preprocessing

import CADETMatch.plugins as plugins

class Node:
    pass

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
        self.continueMCMC = False
        self.errorBias = True
        self.altScores = False
        self.altScoreNames = []
        self.progress_headers = [
            "Generation",
            "Population",
            "Dimension In",
            "Dimension Out",
            "Search Method",
            "Meta Front",
            "Meta Min",
            "Meta Product",
            "Meta Mean",
            "Meta SSE",
            "Meta RMSE",
            "Elapsed Time",
            "Generation Time",
            "Total CPU Time",
            "Last Progress Generation",
            "Generations of Progress",
        ]
        self.eval = Node()

    def setup_dir(self, json_path):
        self.json_path = json_path
        self.setupSettings()

        baseDir = self.settings.get("baseDir", None)
        if baseDir is not None:
            os.chdir(baseDir)
            self.settings["resultsDir"] = Path(baseDir) / self.settings["resultsDir"]

        # create used paths in settings, only the root process will make the directories later
        self.settings["resultsDirEvo"] = Path(self.settings["resultsDir"]) / "evo"
        self.settings["resultsDirMeta"] = Path(self.settings["resultsDir"]) / "meta"
        self.settings["resultsDirGrad"] = Path(self.settings["resultsDir"]) / "grad"
        self.settings["resultsDirMisc"] = Path(self.settings["resultsDir"]) / "misc"
        self.settings["resultsDirSpace"] = Path(self.settings["resultsDir"]) / "space"
        self.settings["resultsDirProgress"] = (
            Path(self.settings["resultsDir"]) / "progress"
        )
        self.settings["resultsDirLog"] = Path(self.settings["resultsDir"]) / "log"
        self.settings["resultsDirMCMC"] = Path(self.settings["resultsDir"]) / "mcmc"
        self.settings["resultsDirBase"] = Path(self.settings["resultsDir"])

    def setup(self, json_path, load_plugins=True):
        "setup the cache based on the json file being used"
        if load_plugins:
            self.scores = plugins.get_plugins("scores")
            self.search = plugins.get_plugins("search")
            self.transforms = plugins.get_plugins("transform")

        if json_path != self.json_path:
            self.json_path = json_path
            self.setupSettings()

            baseDir = self.settings.get("baseDir", None)
            if baseDir is not None:
                os.chdir(baseDir)
                self.settings["resultsDir"] = (
                    Path(baseDir) / self.settings["resultsDir"]
                )

            self.settings["resultsDirEvo"] = Path(self.settings["resultsDir"]) / "evo"
            self.settings["resultsDirMeta"] = Path(self.settings["resultsDir"]) / "meta"
            self.settings["resultsDirGrad"] = Path(self.settings["resultsDir"]) / "grad"
            self.settings["resultsDirMisc"] = Path(self.settings["resultsDir"]) / "misc"
            self.settings["resultsDirSpace"] = (
                Path(self.settings["resultsDir"]) / "space"
            )
            self.settings["resultsDirProgress"] = (
                Path(self.settings["resultsDir"]) / "progress"
            )
            self.settings["resultsDirLog"] = Path(self.settings["resultsDir"]) / "log"
            self.settings["resultsDirMCMC"] = Path(self.settings["resultsDir"]) / "mcmc"
            self.settings["resultsDirBase"] = Path(self.settings["resultsDir"])

        self.abstolFactor = self.settings.get("abstolFactor", 1e-3)
        self.abstolFactorGrad = self.settings.get("abstolFactorGrad", 1e-7)
        self.abstolFactorGradMax = self.settings.get("abstolFactorGradMax", 1e-10)
        self.dynamicTolerance = bool(self.settings.get("dynamicTolerance", False))
        self.gradFineStop = self.settings.get("gradFineStop", 1e-14)

        # When using only SSE based scores this changes if the scores are merged or left as multi objectives
        self.MultiObjectiveSSE = bool(self.settings.get("MultiObjectiveSSE", False))

        self.errorBias = bool(self.settings.get("errorBias", True))

        Cadet.cadet_path = self.settings["CADETPath"]

        self.normalizeOutput = bool(self.settings.get("normalizeOutput", False))
        self.connectionNumberEntries = int(
            self.settings.get("connectionNumberEntries", 5)
        )

        self.parameters = [
            self.transforms[parameter["transform"]](parameter, self)
            for parameter in self.settings["parameters"]
        ]
        self.setupHeaders()
        self.setupTarget()
        self.setupMinMax()

        self.WORST = [self.badScore] * self.numGoalsOrig
        self.WORST_META = [1e308] * len(self.meta_headers)

        self.settings["transform"] = self.transform

        self.correct = None
        if "correct" in self.settings:
            self.correct_transform = numpy.array(self.settings["correct"])
            self.correct = numpy.array(
                [
                    f(v)
                    for f, v in zip(
                        self.settings["transform"], self.settings["correct"]
                    )
                ]
            )

        self.error_path = Path(self.settings["resultsDirBase"], "error.csv")

        self.graphGenerateTime = int(self.settings.get("graphGenerateTime", 3600))
        self.graphMetaTime = int(self.settings.get("graphMetaTime", 1200))

        self.metaResultsOnly = self.settings.get("metaResultsOnly", 1)
        self.stallGenerations = int(self.settings.get("stallGenerations", 10))
        self.stallCorrect = int(self.settings.get("stallCorrect", 5))
        self.progressCorrect = int(self.settings.get("progressCorrect", 5))
        self.progress_elapsed_time = int(
            self.settings.get("progress_elapsed_time", 300)
        )

        self.fullTrainingData = int(self.settings.get("fullTrainingData", 0))

        self.sobolGeneration = bool(
            self.settings.get("soboloGeneration", True)
        ) or bool(self.settings.get("sobolGeneration", True))

        self.continueMCMC = bool(self.settings.get("continueMCMC", False))
        self.MCMCTauMult = int(self.settings.get("MCMCTauMult", 50))

        self.cross_eta = int(self.settings.get("cross_eta", 30))
        self.mutate_eta = int(self.settings.get("mutate_eta", 70))

        self.gradVector = bool(self.settings.get("gradVector", 0))

        self.tempDir = self.settings.get("tempDir", None)
        self.graphType = self.settings.get("graphType", 1)

        self.checkpointInterval = self.settings.get("checkpointInterval", 30)
        self.setupMetaMask()

        self.debugWrite = bool(self.settings.get("debugWrite", False))

        self.finalGradRefinement = bool(self.settings.get("finalGradRefinement", False))

        self.multiStartPercent = self.settings.get("multiStartPercent", 0.1)

        if "MCMCpopulation" not in self.settings:
            self.settings["MCMCpopulation"] = self.settings["population"]

    def resetTransform(self, json_path):
        if json_path != self.json_path:
            self.json_path = json_path

            settings_file = Path(self.json_path)
            with settings_file.open() as json_data:
                settings = jstyleson.load(json_data)

                self.settings["parameters"] = settings["parameters"]

                self.parameters = [
                    self.transforms[parameter["transform"]](parameter, self)
                    for parameter in self.settings["parameters"]
                ]
                self.setupHeaders()
                self.setupTarget()
                self.setupMinMax()

                self.WORST = [self.badScore] * self.numGoalsOrig

                self.settings["transform"] = self.transform

                self.correct = None
                if "correct" in self.settings:
                    self.correct = numpy.array(
                        [
                            f(v)
                            for f, v in zip(
                                self.settings["transform"], self.settings["correct"]
                            )
                        ]
                    )

    def setupMetaMask(self):
        meta_mask_seq = []

        for idx, experiment in enumerate(self.settings["experiments"]):
            for feature in experiment["scores"]:
                if feature["type"] in self.scores:
                    settings = self.scores[feature["type"]].get_settings(feature)
                    meta_mask = settings.meta_mask
                    count = settings.count
                    # multiprocessing.get_logger().info('%s %s %s %s', idx, feature, meta_mask, count)

                    meta_mask_seq.extend(
                        [
                            meta_mask,
                        ]
                        * count
                    )
        # multiprocessing.get_logger().info("%s", meta_mask_seq)
        self.meta_mask = numpy.array(meta_mask_seq)

    def setupSettings(self):
        settings_file = Path(self.json_path)
        with settings_file.open() as json_data:
            self.settings = jstyleson.load(json_data)

            if "CSV" in self.settings:
                self.settings["csv"] = self.settings["CSV"]
            if "csv" not in self.settings:
                self.settings["csv"] = "results.csv"

            #convert features to scores
            for experiment in self.settings['experiments']:
                if "features" in experiment:
                    experiment['scores'] = experiment['features']

                    for score in experiment['scores']:
                        if "isotherm" in score:
                            score['output_path'] = score['isotherm']

                if "isotherm" in experiment:
                    experiment["output_path"] = experiment["isotherm"]

                if "featuresAlt" in experiment:
                    experiment['scoresAlt'] = experiment['featureAlt']

                    for score in experiment['scoresAlt']:
                        if "isotherm" in score:
                            score['output_path'] = score['isotherm']

            if self.settings["searchMethod"] == "Gradient":
                self.settings["population"] = 1
            else:
                self.settings["population"] = int(self.settings["population"])

            self.settings["maxPopulation"] = int(
                self.settings.get("maxPopulation", self.settings["population"])
            )
            self.settings["minPopulation"] = int(
                self.settings.get("minPopulation", self.settings["population"])
            )

            if "bootstrap" in self.settings:
                self.settings["bootstrap"]["samples"] = int(
                    self.settings["bootstrap"]["samples"]
                )

    def setupHeaders(self):
        self.headers = [
            "Time",
            "Name",
            "Method",
            "Condition Number",
        ]

        self.numGoals = 0
        self.badScore = 1.0

        base = len(self.headers)

        parameter_headers = []
        parameter_headers_actual = []

        for parameter in self.parameters:
            parameter_headers.extend(parameter.getHeaders())
            parameter_headers_actual.extend(parameter.getHeadersActual())

        self.parameter_headers = parameter_headers
        self.parameter_headers_actual = parameter_headers_actual

        self.headers.extend(parameter_headers)

        parameters = len(self.headers)
        self.parameter_indexes = list(range(base, parameters))

        self.score_headers = []

        badScore = []

        for idx, experiment in enumerate(self.settings["experiments"]):
            experimentName = experiment["name"]
            experiment["headers"] = []
            for feature in experiment["scores"]:
                if feature["type"] in self.scores:
                    temp = self.scores[feature["type"]].headers(experimentName, feature)

                    settings = self.scores[feature["type"]].get_settings(feature)

                    # if settings.meta_mask:
                    self.numGoals += len(temp)
                    badScore.append(settings.badScore)

                    self.score_headers.extend(temp)
                    experiment["headers"].extend(temp)

        self.badScores = numpy.array(badScore)
        self.allScoreNorm = numpy.all(self.badScores != sys.float_info.max)
        self.allScoreSSE = numpy.all(self.badScores == sys.float_info.max)
        self.badScore = max(badScore)
        self.numGoalsOrig = self.numGoals

        if self.allScoreSSE and self.MultiObjectiveSSE is False:
            self.numGoals = 1

        if self.allScoreNorm:
            # use the first 4 indexes to also keep the SSE
            self.meta_slice = slice(0, 3, 1)
        elif self.allScoreSSE:
            if self.MultiObjectiveSSE:
                # use the first 3 indexes
                self.meta_slice = slice(0, 3, 1)
            else:
                # only used index = 3
                self.meta_slice = slice(3, 4, 1)

        self.headers.extend(self.score_headers)

        self.meta_headers = [
            "Product Root Score",
            "Min Score",
            "Mean Score",
            "SSE",
            "RMSE",
        ]

        self.headers.extend(self.meta_headers)

        scores = len(self.headers)
        self.score_indexes = list(range(parameters, scores))

    def add_units_isotherm(self, units_used, isotherm):
        if not isinstance(isotherm, list):
            isotherm = [
                isotherm,
            ]
        for path in isotherm:
            for element in path.split("/"):
                if element.startswith("unit_"):
                    if element not in units_used:
                        units_used.append(element)

    def add_units_features_alt(self, units_used, features_alt):
        for feature_alt in features_alt:
            self.add_units_isotherm(units_used, feature_alt.get("output_path", ""))
            for feature in feature_alt["scores"]:
                self.add_units_isotherm(units_used, feature.get("output_path", ""))
                self.add_units_isotherm(units_used, feature.get("unit_name", ""))

    def add_units_error_model(self, error_models, target):
        for error_model in error_models:
            error_model_name = error_model["name"]
            if error_model_name in target:
                units_used = target[error_model_name]["units_used"]
                for unit in error_model["units"]:
                    unit_name = "unit_%03d" % int(unit)
                    if unit_name not in units_used:
                        units_used.append(unit_name)

    def setupTarget(self):
        self.target = {}
        self.adaptive = True

        for experiment in self.settings["experiments"]:
            self.target[experiment["name"]] = self.setupExperiment(experiment)
        if "errorModel" in self.settings:
            self.add_units_error_model(self.settings["errorModel"], self.target)


    def setupExperiment(self, experiment, sim=None, dataFromSim=0):
        temp = {}

        units_used = []

        residence_time_unit = int(experiment.get('residence_time_unit', 1))

        if sim is None:
            sim = Cadet()
            sim.filename = Path(experiment["HDF5"]).as_posix()
            sim.load()
            sim.root.experiment_name = experiment["name"]

        abstol = sim.root.input.solver.time_integrator.abstol

        conn = sim.root.input.model.connections.switch_000.connections

        conn = numpy.array(conn)
        conn = numpy.reshape(conn, [-1, self.connectionNumberEntries])

        # find all the entries that connect to the column
        filter = conn[:, 1] == residence_time_unit

        # flow is the sum of all flow rates that connect to this column which is in the last column
        flow = sum(conn[filter, -1])

        unit = sim.root.input.model[f'unit_{residence_time_unit:03d}']

        if unit.unit_type == b"CSTR":
            volume = float(unit.init_volume)

            CV_time = volume / flow

        else:
            # CV needs to be based on superficial velocity not interstitial velocity
            length = float(unit.col_length)

            velocity = unit.velocity
            if velocity == {}:
                velocity = 1.0
            velocity = float(velocity)

            area = unit.cross_section_area
            if area == {}:
                area = 1.0
            area = float(area)

            if area == 1 and abs(velocity) != 1:
                CV_time = length / velocity
            else:
                CV_time = (area * length) / flow

        # force CSV to lowercase
        if "CSV" in experiment:
            experiment["csv"] = experiment["CSV"]

        if dataFromSim:
            temp["time"], temp["value"] = get_times_values(
                sim, {"output_path": experiment["output_path"]}
            )

        elif "csv" in experiment:
            data = numpy.loadtxt(experiment["csv"], delimiter=",")

            temp["time"] = data[:, 0]
            temp["value"] = data[:, 1]

            if self.normalizeOutput:
                temp["factor"] = 1.0 / numpy.max(temp["value"])
            else:
                temp["factor"] = 1.0
            temp["valueFactor"] = temp["value"] * temp["factor"]

        if "scoresAlt" in experiment:
            self.altScores = True
            self.altScoreNames = [
                altFeature["name"] for altFeature in experiment["scoresAlt"]
            ]

            self.add_units_features_alt(units_used, experiment["scoresAlt"])

        peak_maxes = []
        for feature in experiment["scores"]:
            featureName = feature["name"]
            featureType = feature["type"]

            temp[featureName] = feature

            # switch to lower case
            if "CSV" in feature:
                feature["csv"] = feature["CSV"]

            if "csv" in feature:
                if dataFromSim:
                    (
                        temp[featureName]["time"],
                        temp[featureName]["value"],
                    ) = get_times_values(sim, {"output_path": feature["output_path"]})
                else:
                    dataLocal = numpy.loadtxt(feature["csv"], delimiter=",")
                    temp[featureName]["time"] = dataLocal[:, 0]
                    temp[featureName]["value"] = dataLocal[:, 1]
            else:
                temp[featureName]["time"] = temp["time"]
                temp[featureName]["value"] = temp["value"]

            if self.normalizeOutput:
                temp[featureName]["factor"] = 1.0 / numpy.max(
                    temp[featureName]["value"]
                )
                temp[featureName]["value"] = (
                    temp[featureName]["value"] * temp[featureName]["factor"]
                )
            else:
                temp[featureName]["factor"] = 1.0

            try:
                featureStart = float(feature["start"])
                featureStop = float(feature["stop"])
            except KeyError:
                feature["start"] = featureStart = temp[featureName]["time"][0]
                feature["stop"] = featureStop = temp[featureName]["time"][-1]

            if "output_path" in feature:
                temp[featureName]["output_path"] = feature["output_path"]
            else:
                temp[featureName]["output_path"] = experiment["output_path"]

            self.add_units_isotherm(units_used, temp[featureName]["output_path"])

            temp[featureName]["selected"] = (
                temp[featureName]["time"] >= featureStart
            ) & (temp[featureName]["time"] <= featureStop)

            selectedTimes = temp[featureName]["time"][temp[featureName]["selected"]]
            selectedValues = temp[featureName]["value"][temp[featureName]["selected"]]

            if "unit_name" in feature:
                self.add_units_isotherm(units_used, feature["unit_name"])

            if featureType in self.scores:
                temp[featureName].update(
                    self.scores[featureType].setup(
                        sim,
                        feature,
                        selectedTimes,
                        selectedValues,
                        CV_time,
                        abstol,
                        self,
                    )
                )
                settings = self.scores[featureType].get_settings(feature)
                self.adaptive = settings.adaptive
                if "peak_max" in temp[featureName]:
                    peak_maxes.append(
                        temp[featureName]["peak_max"] / temp[featureName]["factor"]
                    )

        temp["smallest_peak"] = min(peak_maxes)
        temp["largest_peak"] = max(peak_maxes)

        #if min(peak_maxes) == max(peak_maxes):
        #    self.dynamicTolerance = False

        temp["units_used"] = units_used
        return temp

    def setupMinMax(self):
        "build the minimum and maximum parameter boundaries"
        self.MIN_VALUE = []
        self.MAX_VALUE = []
        # self.MIN_VALUE_GRAD = []
        # self.MAX_VALUE_GRAD = []
        self.transform = []
        # self.grad_transform = []

        for parameter in self.parameters:
            minValues, maxValues = parameter.getBounds()
            # minGradValues, maxGradValues = self.transforms[transform].getGradBounds(parameter)
            transforms = parameter.transform()
            # grad_transforms = self.transforms[transform].grad_transform(parameter)

            if minValues:
                self.MIN_VALUE.extend(minValues)
                self.MAX_VALUE.extend(maxValues)
                self.transform.extend(transforms)
                # self.grad_transform.extend(grad_transforms)
                # self.MIN_VALUE_GRAD.extend(minGradValues)
                # self.MAX_VALUE_GRAD.extend(maxGradValues)


cache = Cache()


def get_times_values(simulation, target):
    "simplified version of the function so that util does not have to be imported"
    times = simulation.root.output.solution.solution_times
    output_path = target["output_path"]

    if isinstance(output_path, list):
        values = numpy.sum([simulation[i] for i in output_path], 0)
    else:
        values = simulation[output_path]

    return times, values
