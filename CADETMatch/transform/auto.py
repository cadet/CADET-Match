import multiprocessing

import numpy

import CADETMatch.util as util
from CADETMatch.abstract.transform import AbstractTransform


class AutoTransform(AbstractTransform):
    @property
    def name(self):
        return "auto"

    @property
    def count(self):
        return 1

    @property
    def count_extended(self):
        return self.count

    @property
    def okay_linear(self):
        minValue = self.parameter["min"]
        maxValue = self.parameter["max"]
        maxFactor = self.parameter.get("maxFactor", 1000)

        return maxValue / minValue < maxFactor

    @property
    def okay_log(self):
        return not self.okay_linear

    def transform(self):
        if self.okay_log:
            return self.transform_log()
        else:
            return self.transform_linear()

    def transform_log(self):
        minValue = numpy.log(self.parameter["min"])
        maxValue = numpy.log(self.parameter["max"])

        def trans(i):
            return (numpy.log(i) - minValue) / (maxValue - minValue)

        return [
            trans,
        ]

    def transform_linear(self):
        minValue = self.parameter["min"]
        maxValue = self.parameter["max"]

        def trans(i):
            return (i - minValue) / (maxValue - minValue)

        return [
            trans,
        ]

    grad_transform = transform

    def untransform_inputorder(self, seq):
        if self.okay_log:
            return self.untransform_log_inputorder(seq)
        else:
            return self.untransform_linear_inputorder(seq)

    def untransform(self, seq):
        if self.okay_log:
            return self.untransform_log(seq)
        else:
            return self.untransform_linear(seq)

    def untransform_log_inputorder(self, seq):
        minValue = numpy.log(self.parameter["min"])
        maxValue = numpy.log(self.parameter["max"])

        values = [
            (maxValue - minValue) * seq[0] + minValue,
        ]

        values = [
            numpy.exp(values[0]),
        ]
        return values

    def untransform_log(self, seq):
        values = self.untransform_log_inputorder(seq)
        headerValues = values
        return values, headerValues

    def untransform_linear_inputorder(self, seq):
        minValue = self.parameter["min"]
        maxValue = self.parameter["max"]

        values = [
            (maxValue - minValue) * seq[0] + minValue,
        ]
        return values

    def untransform_linear(self, seq):
        values = self.untransform_linear_inputorder(seq)
        headerValues = values
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        if self.okay_log:
            return self.untransform_matrix_log(matrix)
        else:
            return self.untransform_matrix_linear(matrix)

    def untransform_matrix_log(self, matrix):
        minValue = numpy.log(self.parameter["min"])
        maxValue = numpy.log(self.parameter["max"])

        temp = (maxValue - minValue) * matrix + minValue

        values = numpy.exp(temp)

        return values

    def untransform_matrix_linear(self, matrix):
        minValue = self.parameter["min"]
        maxValue = self.parameter["max"]

        values = (maxValue - minValue) * matrix + minValue
        return values

    untransform_matrix_inputorder = untransform_matrix

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)

        if (
            self.parameter.get("experiments", None) is None
            or experiment["name"] in self.parameter["experiments"]
        ):
            location = self.parameter["location"]

            try:
                comp = self.parameter["component"]
                bound = self.parameter["bound"]
                index = None
            except KeyError:
                index = self.parameter["index"]
                bound = None

            if bound is not None:
                unit = self.getUnit(location)
                boundOffset = util.getBoundOffset(sim.root.input.model[unit])

                if comp == -1:
                    position = ()
                    sim[location.lower()] = values[0]
                else:
                    position = boundOffset[comp] + bound
                    sim[location.lower()][position] = values[0]

            if index is not None:
                sim[location.lower()][index] = values[0]
        return values, headerValues

    def getBounds(self):
        return [0.0,], [
            1.0,
        ]

    def getGradBounds(self):
        return [self.parameter["min"],], [
            self.parameter["max"],
        ]

    def getHeaders(self):
        if self.okay_log:
            multiprocessing.get_logger().info(
                "parameter %s log", self.parameter["location"]
            )
        else:
            multiprocessing.get_logger().info(
                "parameter %s linear", self.parameter["location"]
            )

        location = self.parameter["location"]

        try:
            comp = self.parameter["component"]
        except KeyError:
            comp = "None"

        name = location.rsplit("/", 1)[-1]
        bound = self.parameter.get("bound", None)
        index = self.parameter.get("index", None)

        headers = []
        if bound is not None:
            headers.append("%s Comp:%s Bound:%s" % (name, comp, bound))
        elif index is not None:
            headers.append("%s Comp:%s Index:%s" % (name, comp, index))
        return headers

    def getHeadersActual(self):
        return self.getHeaders()

    def setBounds(self, parameter, lb, ub):
        parameter["min"] = lb[0]
        parameter["max"] = ub[0]


plugins = {"auto": AutoTransform}
