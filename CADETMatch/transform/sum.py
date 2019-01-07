import util
import numpy
import calc_coeff

name = "sum"
count = 0
count_extended = 0

def getUnit(location):
    return location.split('/')[3]

def transform(parameter):
    return []

def untransform(seq, cache, parameter, fullPrecision=False):
    return [], []

def getValue(sim, location, bound=None, comp=None, index=None):
    if bound is not None:
        unit = getUnit(location)
        boundOffset = util.getBoundOffset(sim.root.input.model[unit])

        if comp == -1:
            position = ()
            return sim[location.lower()]
        else:
            position = boundOffset[comp] + bound
            return sim[location.lower()][position]

    if index is not None:
        if index == -1:
            return sim[location.lower()]
        else:
            return sim[location.lower()][index]

def setValue(sim, value, location, bound=None, comp=None, index=None):
    if bound is not None:
        unit = getUnit(location)
        boundOffset = util.getBoundOffset(sim.root.input.model[unit])

        if comp == -1:
            position = ()
            sim[location.lower()] = value
        else:
            position = boundOffset[comp] + bound
            sim[location.lower()][position] = value

    if index is not None:
        if index == -1:
            sim[location.lower()] = value
        else:
            sim[location.lower()][index] = value
        
def setSimulation(sim, parameter, seq, cache, experiment, fullPrecision=False):
    location1 = parameter['location1']
    location2 = parameter['location2']
    locationSum = parameter['locationSum']
    
    try:
        comp1 = parameter['component1']
        bound1 = parameter['bound1']
        index1 = None
    except KeyError:
        index1 = parameter['index1']
        bound1 = None
        comp1 = None
    value1 = getValue(sim, location1, bound=bound1, comp=comp1, index=index1)

    try:
        comp2 = parameter['component2']
        bound2 = parameter['bound2']
        index2 = None
    except KeyError:
        index2 = parameter['index2']
        bound2 = None
        comp2 = None

    value2 = getValue(sim, location2, bound=bound2, comp=comp2, index=index2)

    try:
        compSum = parameter['componentSum']
        boundSum = parameter['boundSum']
        indexSum = None
    except KeyError:
        indexSum = parameter['indexSum']
        boundSum = None
        compSum = None
    setValue(sim, value1+value2, locationSum, bound=boundSum, comp=compSum, index=indexSum)

    return [],[]

def setupTarget(parameter):
    return [], 0

def getBounds(parameter):
    return None,None

def getHeaders(parameter):
    return []

def getHeadersActual(parameter):
    return []

def setBounds(parameter, lb, ub):
    return None



