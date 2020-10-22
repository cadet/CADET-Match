import CADETMatch.util as util
from CADETMatch.abstract.transform import AbstractTransform
import numpy


class NullTransform(AbstractTransform):
    @property
    def name(self):
        return "null"

    @property
    def count(self):
        return 1

    @property
    def count_extended(self):
        return self.count

    def transform(self):
        def trans(i):
            return i

        return [
            trans,
        ]

    grad_transform = transform


    def untransform_inputorder(self, seq):
        values = [
            seq[0],
        ]
        return values

    def untransform(self, seq):
        values = self.untransform_inputorder(seq)
        headerValues = values
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = numpy.zeros(matrix.shape)
        values[:, 0] = matrix[:, 0]
        return values

    untransform_matrix_inputorder = untransform_matrix

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)

        if self.parameter.get("experiments", None) is None or experiment["name"] in self.parameter["experiments"]:
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
        minValue = self.parameter["min"]
        maxValue = self.parameter["max"]

        return [minValue,], [maxValue,]

    def getGradBounds(self):
        return [self.parameter["min"],], [self.parameter["max"],]

    def getHeaders(self):
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


class NormTransform(NullTransform):
    @property
    def name(self):
        return "norm"

    def transform(self):
        minValue = self.parameter["min"]
        maxValue = self.parameter["max"]

        def trans(i):
            return (i - minValue) / (maxValue - minValue)

        return [
            trans,
        ]

    grad_transform = transform

    def untransform_inputorder(self, seq):
        minValue = self.parameter["min"]
        maxValue = self.parameter["max"]

        values = [
            (maxValue - minValue) * seq[0] + minValue,
        ]
        return values


    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        minValue = self.parameter["min"]
        maxValue = self.parameter["max"]

        values = (maxValue - minValue) * matrix + minValue
        return values

    untransform_matrix_inputorder = untransform_matrix

    def getBounds(self):
        return [0.0,], [1.0,]

    def getGradBounds(self):
        return [self.parameter["min"],], [self.parameter["max"],]


plugins = {"norm": NormTransform, "null": NullTransform}
