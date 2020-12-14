#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 15:10:49 2019
This is a library of tensor network method for Ising models
@author: Bruce
"""

import numpy as np
import scipy.integrate as integrate
from ncon import ncon
from tntools.initialtensors import get_initial_tensor

def exactSol(J):
    """
    g = exactSol(J).
    ---------------------
    Return: g is defined as ln(Z) / N = -beta * f, where Z is the partition
    function, N is the total number of spin, f the free energy per site.
    """

    g = np.zeros(len(J))
    for i in range(len(J)):
        k = 1 / (np.sinh(2*J[i])**2)
        integrand = lambda theta: np.log((np.cosh(2*J[i]))**2
        + 1 / k * np.sqrt(1 + k**2 - 2*k * np.cos(2*theta)))
        g[i] = np.log(2) / 2 + 1 / (2 * np.pi) * integrate.quad(integrand, 0,np.pi)[0]
    return g


def Ising2dT(beta = 0.4, h = 0, isSym = False):
    """
    T = Ising2dT(J,h).
    -------------------------
    Set up the initial tensor for 2d classical Ising model on a square lattice.
    Argument: J is defined to be beta * J = J / kT, and h is
    defined to be beta*h = h / kT, where J and h are conventional coupling constants.
    Return: a rank 4 tensor T[i,j,k,l]. Each index of the tensor represents
    physical classical spin, and the tensor T represents the Boltzmann weight
    for interaction on one plaquettes.
    """
    pars = {"model":"ising", "dtype":"float64", 
            "J":1, "H":h, "beta":beta, "symmetry_tensors":isSym}
    T0 = get_initial_tensor(pars)

    return T0

def Ising2dTwall(J = 0.1):
    """
    Set up the initial tensor for 2d classical Ising model on a square lattice
    according to the method described in Xiang's HOTRG paper and Hauru's Gilt
    paper. The number of initial tensor is the same as the number of initial
    spin variables.

    Parameters
    ----------
    J : double, optional
        Inverse temperature J/T. The default is 0.1.

    Returns
    -------
    T: four leg tensor

    """
    raise Warning("This function is deprecated now!")
    # The sqaure root of weight matrix (See Hauru's Gilt paper for details)
    M = np.array([[np.sqrt(np.cosh(J)), np.sqrt(np.sinh(J))],
         [np.sqrt(np.cosh(J)), - np.sqrt(np.sinh(J))]])
    T = np.einsum("ai,aj,ak,al->ijkl",M,M,M,M)
    return T

def checkSym(T):
    """
    Check whether a tensor T has horizontal and verticle symmetry

    Parameters
    ----------
    T : four-leg tensor
       j|
     i--T--k.
       l|

    Returns
    -------
    None.

    """
    Tarr = T.to_ndarray()
    ishSym = np.allclose(Tarr,Tarr.transpose([2,1,0,3]))
    isvSym = np.allclose(Tarr,Tarr.transpose([0,3,2,1]))
    print("Horizontal symmetry: {}.".format(ishSym))
    print("Vertical symmetry: {}.".format(isvSym))

def calg(A,Anorm,initN = 4, traceord = [1,1,2,2]):
    numlevels = len(A)-1
    FreeEnergy = np.zeros(numlevels)
    Anorm = np.array(Anorm)
    for k in range(1,numlevels+1):
        FreeEnergy[k-1] = (np.sum(4.0 ** (-np.arange(0,k+1)) * 
                                  np.log(Anorm[:(k+1)])) + 
                           (1/4)**k * np.log(ncon(
                               [A[k]],[traceord]).norm())) /(2*initN)
    return FreeEnergy

if __name__ == "__main__":
    import argparse
    from trg import mainTRG, calg
    
    def compareg(n_iter = 12,Dcut = 8, isSym = False):
        Tc = 2 / np.log(1 + np.sqrt(2))
        gext = exactSol([1/Tc])[0]
        A, Anorm = mainTRG(1.0, Dcut, n_iter, isSym = isSym)
        gapp = calg(A,Anorm, initN = 1,  traceord = [1,2,1,2])
        err = np.abs(gapp - gext) / gext
        return err, A
    parser = argparse.ArgumentParser(
        "Test TRG implementation")
    parser.add_argument("--chi", dest = "chi", type = int,
                    help = "bound dimension (default: 10)",
                    default = 10)
    parser.add_argument("--iterN", dest = "iterN", type = int,
                    help = "Number of TRG iterations (default: 25)",
                    default = 25)
    parser.add_argument("--isSym", help = "whether to use Z2 symmetry",
                        action = "store_true")
    args = parser.parse_args()
    chi = args.chi
    iterN = args.iterN
    isSym = args.isSym
    err,A = compareg(iterN, chi, isSym)
    print("Error of free energy at Tc is ")
    print(err)
    


