import numpy

import CADETMatch.calc_coeff as calc_coeff
import CADETMatch.util as util
from CADETMatch.abstract.transform import AbstractTransform


class LinearTransform(AbstractTransform):
    @property
    def name(self):
        return "linear"

    @property
    def count(self):
        return 2

    @property
    def count_extended(self):
        return self.count

    def transform(self):
        def trans_a(i):
            return i

        def trans_b(i):
            return i

        return [trans_a, trans_b]

    grad_transform = transform

    def untransform_inputorder(self, seq):
        values = [seq[0], seq[1]]
        return values

    def untransform(self, seq):
        values = self.untransform_inputorder(seq)
        headerValues = values
        return values, headerValues

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        values = numpy.array(matrix)
        return values

    untransform_matrix_inputorder = untransform_matrix

    def setSimulation(self, sim, seq, experiment):
        values, headerValues = self.untransform(seq)

        if (
            self.parameter.get("experiments", None) is None
            or experiment["name"] in self.parameter["experiments"]
        ):
            location = self.parameter["location"]

            minX = self.parameter["minX"]
            maxX = self.parameter["maxX"]

            x_name = self.parameter["x_name"]

            x_value = experiment[x_name]

            slope, intercept = calc_coeff.linear_coeff(minX, values[0], maxX, values[1])

            value = calc_coeff.linear(x_value, slope, intercept)

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
                    sim[location.lower()] = value
                else:
                    position = boundOffset[comp] + bound
                    sim[location.lower()][position] = value

            if index is not None:
                sim[location.lower()][index] = value

        return values, headerValues

    def getBounds(self):
        minLower = self.parameter["minLower"]
        maxLower = self.parameter["maxLower"]
        minUpper = self.parameter["minUpper"]
        maxUpper = self.parameter["maxUpper"]

        minValues = [minLower, minUpper]
        maxValues = [maxLower, maxUpper]

        return minValues, maxValues

    def getGradBounds(self):
        return self.getBounds()

    def getHeaders(self):
        bound = self.parameter["bound"]
        comp = self.parameter["component"]

        headers = []
        headers.append("Lower Comp:%s Bound:%s" % (comp, bound))
        headers.append("Upper Comp:%s Bound:%s" % (comp, bound))
        return headers

    def getHeadersActual(self):
        return self.getHeaders()

    def setBounds(self, parameter, lb, ub):
        parameter["minLower"] = lb[0]
        parameter["maxLower"] = ub[0]
        parameter["minUpper"] = lb[1]
        parameter["maxUpper"] = ub[1]


class NormLinearTransform(LinearTransform):
    @property
    def name(self):
        return "norm_linear"

    def transform(self):
        minLower = self.parameter["minLower"]
        maxLower = self.parameter["maxLower"]
        minUpper = self.parameter["minUpper"]
        maxUpper = self.parameter["maxUpper"]

        def trans_a(i):
            return (i - minLower) / (maxLower - minLower)

        def trans_b(i):
            return (i - minUpper) / (maxUpper - minUpper)

        return [trans_a, trans_b]

    grad_transform = transform

    def untransform_inputorder(self, seq):
        minLower = self.parameter["minLower"]
        maxLower = self.parameter["maxLower"]
        minUpper = self.parameter["minUpper"]
        maxUpper = self.parameter["maxUpper"]
        minX = self.parameter["minX"]
        maxX = self.parameter["maxX"]

        minValues = numpy.array([minLower, minUpper])
        maxValues = numpy.array([maxLower, maxUpper])

        values = numpy.array(seq)

        values = (maxValues - minValues) * values + minValues
        return values

    def grad_untransform(self, seq):
        return self.untransform(seq)[0]

    def untransform_matrix(self, matrix):
        minLower = self.parameter["minLower"]
        maxLower = self.parameter["maxLower"]
        minUpper = self.parameter["minUpper"]
        maxUpper = self.parameter["maxUpper"]

        minValues = numpy.array([minLower, minUpper])
        maxValues = numpy.array([maxLower, maxUpper])

        values = (maxValues - minValues) * values + minValues

        return values

    untransform_matrix_inputorder = untransform_matrix

    def getBounds(self):
        return [0.0, 0.0], [1.0, 1.0]

    def getGradBounds(self):
        minLower = self.parameter["minLower"]
        maxLower = self.parameter["maxLower"]
        minUpper = self.parameter["minUpper"]
        maxUpper = self.parameter["maxUpper"]
        return [minLower, minUpper], [maxLower, maxUpper]


plugins = {"norm_linear": NormLinearTransform, "linear": LinearTransform}
