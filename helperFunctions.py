import numpy as np
from functools import reduce, partial
from mpmath import mp
from mpmath import pi as Pi
from mpmath import sin as Sin
from mpmath import asin as Asin
from mpmath import sqrt as Sqrt
from multiprocessing import Pool

import datetime


dyadicMap = lambda x : (2 * x) % 1

# --------------------------------------
tau = 12
# --------------------------------------


# --------------------------------------
# convert between binary / decimal basis
# --------------------------------------
def decimalToBinary(decimalInitial, targetBinaryPrecision = tau):
    return reduce(lambda acc, _: [dyadicMap(acc[0]), acc[1] + ('0' if acc[0] < 0.5 else '1')], 
                  range(targetBinaryPrecision), 
                  [decimalInitial, ''])[1]

def binaryToDecimal(binaryInitial):
    return reduce(lambda acc, val: acc + int(val[1]) / mp.power(2, (val[0] + 1)), 
                  enumerate(binaryInitial), 
                  mp.mpf(0.0))


def binaryReducer(val):
    return int(val[1]) / mp.power(2, (val[0] + 1))

def binaryToDecFaster(binaryInitial):

    with Pool(8) as p:
        tt = p.map(binaryReducer, enumerate(binaryInitial))

    res = mp.mpf(0)
    for _ in tt:
        res += _
    
    return res
# --------------------------------------

# --------------------------------------
phiInv                 = lambda z: np.arcsin(np.sqrt(z)) / (2.0 * np.pi)
decimalToBinary_phiInv = lambda z: decimalToBinary(phiInv(z))
phi                    = lambda theta: Sin(theta * Pi * 2.0) ** 2
# --------------------------------------

# --------------------------------------
# decoding functions
# --------------------------------------

def dyadicDecoder(decimalInitial, k):
    return (2 ** (k * tau) * decimalInitial) % 1

def logisticDecoder(decimalInitial, k):
    return float(Sin(2 ** (k * tau) * Asin(Sqrt(decimalInitial))) ** 2)

def findInitialCondition(trainData):
    conjugateInitial_binary = ''.join(map(decimalToBinary_phiInv, trainData))

    necessaryPrecision = len(conjugateInitial_binary)
    assert tau * len(trainData) == necessaryPrecision
    
    # data is passed through sequentially so no need to worry
    # plus, all samples have the same size anyway
    # to be safe, these global settings should be handled more carefully with context managers
    
    mp.prec = necessaryPrecision 
    print('significance = %d bits ; %d digits (base-10) ; ratio = %.3f\n' % (mp.prec, mp.dps, mp.prec / mp.dps))

    # conjugateInitial = binaryToDecimal(conjugateInitial_binary)
    conjugateInitial = binaryToDecFaster(conjugateInitial_binary)
    decimalInitial = phi(conjugateInitial)

    return decimalInitial

# --------------------------------------
def generateData(decimalInitial, howManyPoints):
    p_logisticDecoder = partial(logisticDecoder, decimalInitial)
    with Pool(8) as p:
        decodedValues = p.map(p_logisticDecoder, range(howManyPoints))
    return decodedValues
# --------------------------------------
