import CADETMatch.util as util
import numpy
from CADETMatch.abstract.transform import AbstractTransform
import multiprocessing

class AutoKeqTransform(AbstractTransform):
    @property
    def name(self):
        return "auto_keq"

    @property
    def count(self):
        return 2

    @property
    def count_extended(self):
        return 3

    @property
    def okay_linear_ka(self):
        minValue = self.parameter["minKA"]
        maxValue = self.parameter["maxKA"]
        maxFactor = self.parameter.get('maxFactor', 1000)

        return maxValue/minValue < maxFactor

    @property
    def okay_log_ka(self):
        return not self.okay_linear_ka

    @property
    def okay_linear_keq(self):
        minValue = self.parameter["minKEQ"]
        maxValue = self.parameter["maxKEQ"]
        maxFactor = self.parameter.get('maxFactor', 1000)

        return maxValue/minValue < maxFactor

    @property
    def okay_log_keq(self):
        return not self.okay_linear_keq

    def transform(self):
        minKA = self.parameter["minKA"]
        maxKA = self.parameter["maxKA"]
        minKEQ = self.parameter["minKEQ"]
        maxKEQ = self.parameter["maxKEQ"]

        if self.okay_log_ka:
            def trans_a(i):
                return (numpy.log(i) - numpy.log(minKA)) / (numpy.log(maxKA) - numpy.log(minKA))
        else:
            def trans_a(i):
                return (i - minKA) / (maxKA - minKA)

        if self.okay_log_keq:
            def trans_b(i):
                return (numpy.log(i) - numpy.log(minKEQ)) / (numpy.log(maxKEQ) - numpy.log(minKEQ))
        else:
            def trans_b(i):
                return (i - minKEQ) / (maxKEQ - minKEQ)

        return [trans_a, trans_b]

    grad_transform = transform

    def untransform_inputorder(self, seq):
        minKA = self.parameter["minKA"]
        maxKA = self.parameter["maxKA"]
        minKEQ = self.parameter["minKEQ"]
        maxKEQ = self.parameter["maxKEQ"]

        if self.okay_log_ka:
            ka = (numpy.log(maxKA) - numpy.log(minKA)) * seq[0] + numpy.log(minKA)
            ka = numpy.exp(ka)
        else:
            ka = (maxKA - minKA) * seq[0] + minKA

        if self.okay_log_keq:
            keq = (numpy.log(maxKEQ) - numpy.log(minKEQ)) * seq[1] + numpy.log(minKEQ)
            keq = numpy.exp(keq)
        else:
            keq = (maxKEQ - minKEQ) * seq[1] + minKEQ

        return [ka, keq]

    def untransform(self, seq):
        ka, keq = self.untransform_inputorder(seq)
        kd = ka/keq

        values = [ka, kd]
        headerValues = [ka, kd, keq]
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = self.untransform_matrix_inputorder(matrix)
        values[:, 1] = values[:, 0] / values[:, 1]
        return values

    def untransform_matrix_inputorder(self, matrix):
        minKA = self.parameter["minKA"]
        maxKA = self.parameter["maxKA"]
        minKEQ = self.parameter["minKEQ"]
        maxKEQ = self.parameter["maxKEQ"]

        if self.okay_log_ka:
            minKA = numpy.log(minKA)
            maxKA = numpy.log(maxKA)

        if self.okay_log_keq:
            minKEQ = numpy.log(minKEQ)
            maxKEQ = numpy.log(maxKEQ)

        minValues = numpy.array([minKA, minKEQ])
        maxValues = numpy.array([maxKA, maxKEQ])

        values = (maxValues - minValues) * matrix + minValues

        if self.okay_log_ka and self.okay_log_keq:
            values = numpy.exp(values)
        elif self.okay_log_ka and self.okay_linear_keq:
            values[:,0] = numpy.exp(values[:,0])
        elif self.okay_linear_ka and self.okay_log_keq:
            values[:,1] = numpy.exp(values[:,1])

        return values

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)

        if self.parameter.get("experiments", None) is None or experiment["name"] in self.parameter["experiments"]:
            location = self.parameter["location"]

            comp = self.parameter["component"]
            bound = self.parameter["bound"]

            unit = self.getUnit(location[0])
            boundOffset = util.getBoundOffset(sim.root.input.model[unit])

            position = boundOffset[comp] + bound
            sim[location[0].lower()][position] = values[0]
            sim[location[1].lower()][position] = values[1]

        return values, headerValues

    def getBounds(self):
        return [0.0, 0.0], [1.0, 1.0]

    def getGradBounds(self):
        return [self.parameter["min"],], [self.parameter["max"],]

    def getHeaders(self):
        if self.okay_log_ka:
            message_ka = "log ka"
        else:
            message_ka = "linear ka"

        if self.okay_log_keq:
            message_keq = "log keq"
        else:
            message_keq = "linear keq"

        multiprocessing.get_logger().info("parameter %s %s %s", self.parameter['location'], message_ka, message_keq)

        location = self.parameter["location"]
        nameKA = location[0].rsplit("/", 1)[-1]
        nameKD = location[1].rsplit("/", 1)[-1]
        bound = self.parameter["bound"]
        comp = self.parameter["component"]

        headers = []
        headers.append("%s Comp:%s Bound:%s" % (nameKA, comp, bound))
        headers.append("%s Comp:%s Bound:%s" % (nameKD, comp, bound))
        headers.append("%s/%s Comp:%s Bound:%s" % (nameKA, nameKD, comp, bound))
        return headers

    def getHeadersActual(self):
        location = self.parameter["location"]
        nameKA = location[0].rsplit("/", 1)[-1]
        nameKD = location[1].rsplit("/", 1)[-1]
        bound = self.parameter["bound"]
        comp = self.parameter["component"]

        headers = []
        headers.append("%s Comp:%s Bound:%s" % (nameKA, comp, bound))
        headers.append("%s/%s Comp:%s Bound:%s" % (nameKA, nameKD, comp, bound))
        return headers

    def setBounds(self, parameter, lb, ub):
        parameter["minKA"] = lb[0]
        parameter["maxKA"] = ub[0]
        parameter["minKEQ"] = lb[1]
        parameter["maxKEQ"] = ub[1]


plugins = {"auto_keq": AutoKeqTransform}

