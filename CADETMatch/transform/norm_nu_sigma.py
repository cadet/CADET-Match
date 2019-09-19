import CADETMatch.util as util
import numpy

name = "norm_nu_sigma"
count = 2
count_extended = 3

def getUnit(location):
    return location.split('/')[3]

def untransform(seq, cache, parameter):
    minNu = parameter['minNu']
    maxNu = parameter['maxNu']
    minSigma = parameter['minSigma']
    maxSigma = parameter['maxSigma']

    minValues = numpy.array([minNu, minNu+minSigma])
    maxValues = numpy.array([maxNu, maxNu+maxSigma])

    values = numpy.array(seq)

    values = (maxValues - minValues) * values + minValues

    values = [values[0], values[1] - values[0]]
    headerValues = [values[0], values[1], values[0]+values[1]]
    return values, headerValues

def untransform_matrix(matrix, cache, parameter):
    values = untransform_matrix_inputorder(matrix, cache, parameter)
    values[:,1] = values[:,1] - values[:,0]
    return values

def untransform_matrix_inputorder(matrix, cache, parameter):
    minNu = parameter['minNu']
    maxNu = parameter['maxNu']
    minSigma = parameter['minSigma']
    maxSigma = parameter['maxSigma']

    minValues = numpy.array([minNu, minNu+minSigma])
    maxValues = numpy.array([maxNu, maxNu+maxSigma])

    temp = (maxValues - minValues) * matrix + minValues

    values = numpy.zeros(temp.shape)
    values[:,0] = temp[:,0]
    values[:,1] = temp[:,1]

    return values

def setSimulation(sim, parameter, seq, cache, experiment):
    values, headerValues = untransform(seq, cache, parameter)
    
    if parameter.get('experiments', None) is None or experiment['name'] in parameter['experiments']:
        nu_location = parameter['nu_location']
        sigma_location = parameter['sigma_location']
    
        comp = parameter['component']
        bound = parameter['bound']
    
        unit = getUnit(nu_location)
        boundOffset = util.getBoundOffset(sim.root.input.model[unit])

        position = boundOffset[comp] + bound
        sim[nu_location.lower()][position] = values[0]
        sim[sigma_location.lower()][position] = values[1]

    return values, headerValues

def setupTarget(parameter):
    nu_location = parameter['nu_location']
    sigma_location = parameter['sigma_location']
    bound = parameter['bound']
    comp = parameter['component']

    sensitivityOk = 1
    nameNu = nu_location.rsplit('/', 1)[-1]
    nameSigma = sigma_location.rsplit('/', 1)[-1]
    unit = int(nu_location.split('/')[3].replace('unit_', ''))

    return [(nameNu, unit, comp, bound), (nameSigma, unit, comp, bound)], sensitivityOk

def getBounds(parameter):
    return [0.0, 0.0], [1.0, 1.0]

def getHeaders(parameter):
    nu_location = parameter['nu_location']
    sigma_location = parameter['sigma_location']
    nameNu = nu_location.rsplit('/', 1)[-1]
    nameSigma = sigma_location.rsplit('/', 1)[-1]
    bound = parameter['bound']
    comp = parameter['component']
    
    headers = []
    headers.append("%s Comp:%s Bound:%s" % (nameNu, comp, bound))
    headers.append("%s Comp:%s Bound:%s" % (nameSigma, comp, bound))
    headers.append("%s+%s Comp:%s Bound:%s" % (nameNu, nameSigma, comp, bound))
    return headers

def getHeadersActual(parameter):
    nu_location = parameter['nu_location']
    sigma_location = parameter['sigma_location']
    nameNu = nu_location.rsplit('/', 1)[-1]
    nameSigma = sigma_location.rsplit('/', 1)[-1]
    bound = parameter['bound']
    comp = parameter['component']
    
    headers = []
    headers.append("%s Comp:%s Bound:%s" % (nameNu, comp, bound))
    headers.append("%s+%s Comp:%s Bound:%s" % (nameNu, nameSigma, comp, bound))
    return headers

def setBounds(parameter, lb, ub):
    parameter['minNu'] = lb[0]
    parameter['maxNu'] = ub[0]
    parameter['minSigma'] = lb[1]
    parameter['maxSigma'] = ub[1]