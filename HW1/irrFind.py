import numpy as np
from scipy.optimize import fsolve
def equation(r, cashFlowVec, payment, compound):
    equ = 0
    for i, value in enumerate(cashFlowVec, start=1):
        equ += value / ((1 + (r / compound))**((i-1)*payment))
    return equ
def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    payment = cashFlowPeriod / compoundPeriod
    compound = 12 / compoundPeriod
    irr = fsolve(equation, x0=0, args=(cashFlowVec, payment, compound))

    return irr[0]
