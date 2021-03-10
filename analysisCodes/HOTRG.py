#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : HOTRG.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 11.03.2021
# Last Modified Date: 11.03.2021
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:31:37 2020
Implementation of high order TRG (HOTRG)
proposed by T. Xiang in 2012
@author: Bruce
-----
HOTRG + transfer matirx M; chi=16, s = 2
[-0.          0.1234995   1.02673965  1.2456992   1.2456992   2.36789891
  2.5695128   2.5695128   2.5695128   2.5695128   4.11228583  5.13902217
  5.94595048  6.26003998  6.72931109  7.41802666]
------
HOTRG + hyper-operator without first order correction about projection
operator; chi = 16, s = 2
[-0.        ,  4.2549851 ,  4.38465272,  4.38465272,  4.88206002,
        4.88206002,  4.89675284,  6.07642182,  6.07642182,  6.61270868]
Notice, this result is extremely unstable
-------
HOTRG + hyper-operator with first order correction about projection
operator; chi = 16, s = 2
"""

import jax
import numpy as np
import jax.numpy as jnp
from numpy import linalg as LA
from jax.numpy import linalg as jLA
from jncon import jncon
from ncon import ncon
from Isings import Ising2dT
from scipy.sparse.linalg import eigs,LinearOperator
from gilts import gilt_hotrgplaq
from abeliantensors import Tensor
from itertools import product
import pickle as pkl
import os

## Functions for converting between ordinary numpy tensor
## and abeliantensors
def convertAbe(A):
    """
    Convert A to class abeliantensors if it is not
    """
    if type(A).__module__.split(".")[0] != 'abeliantensors':
        A = Tensor.from_ndarray(A)
    return A

def convertAbeBack(A):
    """
    Convert A to numpy.array if it is abelieantensors
    """
    if type(A).__module__.split(".")[0] == 'abeliantensors':
        A = A.to_ndarray()
    return A

## Calculate the invariant tensor that has the correct
## overall magnitute
def getInvA(A, Anorm):
    """
    Given list A containing renormalized tensor with norm 1
    and a list of its norm, return the invariant version of Ainv
    Parameters
    ----------
    A : list
        renormalized tensors with norm 1.
    Anorm : list
        norm of the tensors
    Returns
    -------
    Ainv : list
        a list of invariant tensors.

    """
    Ainv = []
    for k in range(len(A)):
        Ainv.append(A[k] * (Anorm[k])**(-1/3))
    return Ainv

## Calculating scaling dimensions a la Gu and Wen (2009)
def scDimWen(A, scaleN = 20):
    """
    Calculate scaling dimensions a la Wen (2009)
       k|
     i--A--j.
       l|
    Parameters
    ----------
    A : four leg tensor, A[i,j,k,l]

    Returns
    -------
    scaling dimension

    """
    A = convertAbe(A)
    # Minv = ncon([A],[[1,1,-1,-2]])
    Minv = ncon([A, A], [[1, 2, -1, -3], [2, 1, -2, -4]])
    s = Minv.eig([0, 1], [2, 3], sparse=True, chis=scaleN)[0]
    s = s.to_ndarray()
    s = np.abs(s)
    s = -np.sort(-s)
    ccharge = np.log(s[0]) * 6 / np.pi
    scdim = -np.log(s/s[0])/(2*np.pi) * 2
    return scdim, ccharge

def eigDecp(rho,chiCut, isjax = False, epsilon = 1e-15):
    """
    Perform eigendecomposition to a hermitian matrix rho
    rho is also positive semidefinite
    return its first chi largest eigenvalues and eigenvecotors
    """
    if not isjax:
        dtemp, utemp = LA.eigh(rho)
    else:
        dtemp, utemp = jLA.eigh(rho)

    # Be careful the value in dtemp is in ascending order
    chiCut = np.min([chiCut, np.sum(dtemp/dtemp[-1] > epsilon)])

    dcut = dtemp[:-chiCut - 1:-1]
    ucut = utemp[:,:-chiCut - 1:-1]
    if not isjax:
        err = 1 - np.sum(np.abs(dcut)) / np.sum(np.abs(dtemp))
    else:
        err = 1 - jnp.sum(jnp.abs(dcut)) / jnp.sum(jnp.abs(dtemp))
    return dcut, ucut, err


## Functions for determine the isometric tensors w,v in the HOTRG
def yProjectorAB(B, A, chiH, cg_eps = 1e-7, isjax = False,
                 evenTrunc = False):
    """
    Determine the projector for the contraction along y direction
    """

    if not isjax:
        B = convertAbe(B)
        A = convertAbe(A)
        M = ncon([B,A,B.conjugate(),A.conjugate()],
             [[-1,2,1,5],[-2,4,5,3],[-3,2,1,6],[-4,4,6,3]])
        d, w, err = M.eig([0,1],[2,3], hermitian = True,
                 chis = [i+1 for i in range(chiH)], eps = cg_eps,
                 evenTrunc = evenTrunc,
                 return_rel_err = True)
    else:
        M = jncon([B, A, B, A],
             [[-1,2,1,5],[-2,4,5,3],[-3,2,1,6],[-4,4,6,3]])
        chiHIB = B.shape[0]
        chiHIA = A.shape[0]
        M = M.reshape(chiHIB * chiHIA, chiHIB * chiHIA)
        chiH = min(chiH, chiHIB * chiHIA)
        d, w, err = eigDecp(M, chiH, isjax)
        chiHout = len(d)
        w = w.reshape(chiHIB,chiHIA,chiHout)
    return w, d, err

def xProjectorAB(B, A, chiV, cg_eps = 1e-7, isjax = False,
                 evenTrunc = False):
    """
    Determine the projector for the contraction along x direction
    """

    if not isjax:
        B = convertAbe(B)
        A = convertAbe(A)
        M = ncon([B,A,B.conjugate(),A.conjugate()],
                 [[1,5,-1,2],[5,3,-2,4],[1,6,-3,2],[6,3,-4,4]])
        d, v, err = M.eig([0,1],[2,3], hermitian = True,
                 chis = [i+1 for i in range(chiV)], eps = cg_eps,
                 evenTrunc = evenTrunc,
                 return_rel_err = True)
    else:
        M = jncon([B, A, B, A],
                 [[1,5,-1,2],[5,3,-2,4],[1,6,-3,2],[6,3,-4,4]])
        chiVIB = B.shape[2]
        chiVIA = A.shape[2]
        M = M.reshape(chiVIB * chiVIA, chiVIB * chiVIA)
        chiV = min(chiV, chiVIB * chiVIA)
        d, v, err = eigDecp(M,chiV, isjax)
        chiVout = len(d)
        v = v.reshape(chiVIB,chiVIA,chiVout)
    return v, d, err


## Functions for fixing the sign ambiguity in order to get the
## the manifest fixed-point tensor
def applySignFix(T, dh, dv):
    if type(dh) == int:
        res = T * 1
    else:
        res = T * dh[:, None, None, None] * dh[None, :, None, None] * \
                dv[None, None, :, None] * dv[None, None, None, :]
    return res

def fixSign(T,Told, trylegs = [0,2,0,2], debug = False):

    """
    Fixed the sign ambiguity of T tensor according to Told tensor
    This algorithm is design for abeliantensors.symmetrytensors.TensorZ2 type
    See my notes in iPad with the name fixSign in GoodNotes for more details
    tensors
    Parameters
    ----------
    T : 4-leg tensor T[i,j,k,l]
          k|
        i--T--j.
          l|
    Told : 4-leg tensor T[i,j,k,l]
        same as T.
    trylegs: list

    Returns
    -------
    Tfix : 4-leg tensor
        fixed version T s.t. it has the same sign with Told.

    """
    if T.shape == [[1]]*4:
        return T, 1, 1
    legh0, legv0, legh1, legv1 = trylegs
    def legSlice(A,leg = 0):
        if leg == 0 :
            res = A[:,0,0,0]
        elif leg == 1:
            res = A[0,:,0,0]
        elif leg == 2:
            res = A[0,0,:,0]
        elif leg == 3:
            res = A[0,0,0,:]
        return res
    # ensure the shape of Told and T must be the same
    assert T.shape == Told .shape, "The shape of T and Told must be the same"
    assert T.shape[0] == T.shape[1], "Left and right indices should have the same shape"
    assert T.shape[2] == T.shape[3], "Up and down indices should have the same shape"
    # The arraies to keep track of sign difference between T and Told,
    # Both are organised in to two sectors
    dh = []
    dv = []
    # Check the two tensors should be consist with our basic assumption
    T0000 = T[(0,0,0,0)][0,0,0,0]
    Told0000 = Told[(0,0,0,0)][0,0,0,0]
    assert np.sign(T0000 * Told0000) > 0, "Two tensors doesn't obey our basic assumption"
    # determine dh and dv in 0 sector
    dh.append(np.sign(legSlice(Told[(0,0,0,0)], legh0) * \
                      legSlice(T[(0,0,0,0)], legh0)))
    dv.append(np.sign(legSlice(Told[(0,0,0,0)], legv0) * \
                      legSlice(T[(0,0,0,0)], legv0)))
    # determine dh in 1 sector
    dh.append(np.sign(legSlice(Told[(1,1,0,0)],legh1) * \
                      legSlice(T[(1,1,0,0)], legh1)))
    # determine dv in 1 sector
    dv10 = np.sign(Told[(1,0,1,0)][0,0,0,0] * T[(1,0,1,0)][0,0,0,0])
    dv.append(dv10 * np.sign(legSlice(Told[(0,0,1,1)], legv1) * \
                             legSlice(T[(0,0,1,1)], legv1)))

    # apply dh and dv on T to fixed the sign ambiguities
    Tfix = T.copy()
    for key in Tfix.sects.keys():
        l, r, u, d = key
        dl = dh[l]
        dr = dh[r]
        du = dv[u]
        dd = dv[d]
        Tfix[key] = Tfix[key] * dl[:, None, None, None] * dr[None, :, None, None] * \
            du[None, None, :, None] * dd[None, None, None, :]
    # Concatenate sectors in dh and dv for later use
    dh = np.concatenate(dh)
    dv = np.concatenate(dv)
    # for debug
    if debug:
        Tcheck = applySignFix(T.to_ndarray(), dh, dv)
        print("Is Tcheck calculated using dh and dv same as fixed T??")
        print(np.allclose(Tcheck, Tfix.to_ndarray()))
    return Tfix, dh, dv

def fixBestSign(T,Told, debug = False):
    bestlegh0 = 0
    bestlegv0 = 2
    bestlegh1 = 0
    bestlegv1 = 2
    diffTmin = 100
    hrange = range(2)
    vrange = range(2,4)
    for legh0, legv0, legh1, legv1 in product(hrange, vrange,
                                              hrange, vrange):
        tryleglist = [legh0, legv0, legh1, legv1]
        Tfix = fixSign(T, Told, tryleglist)[0]
        diffT = LA.norm(Tfix.to_ndarray() - Told.to_ndarray())
        if diffT < diffTmin:
            diffTmin = diffT
            bestlegh0, bestlegv0, bestlegh1, bestlegv1 = tryleglist
    bestlegs = [bestlegh0, bestlegv0, bestlegh1, bestlegv1]
    Tfix, dh, dv = fixSign(T, Told, bestlegs)
    # for debug
    if debug:
        Tcheck = T.to_ndarray() * dh[:, None, None, None] * dh[None, :, None, None] * \
            dv[None, None, :, None] * dv[None, None, None, :]
        print("Is Tcheck calculated using dh and dv same as fixed T??")
        print(np.allclose(Tcheck, Tfix.to_ndarray()))
    return Tfix, dh, dv



## The functions for the HOTRG and the Gilt-HOTRG

def halfHOTRG(B, A, chi, direction = "v", verbose = True, cg_eps = 1e-6,
              isjax = False, evenTrunc = False):
    """
    Perform half of the HOTRG coarse graining
    """

    if direction == "v":
        if verbose:
            print("Coarse graining in y direction...")
        # determine the isometry that squeezes legs in y direction
        if not isjax:
            w, dw, SP1err = yProjectorAB(B, A, chi, cg_eps,
                                         evenTrunc = evenTrunc)
        else:
            w, dw, SP1err = yProjectorAB(B, A, chi, isjax = True)
        if verbose:
            print("Spectrum of B @ A is:")
            dwarr = dw / dw.max()
            dwarr = np.abs(dwarr.to_ndarray())
            dwarr = - np.sort(-dwarr)
            print(dwarr[:10])
            print("Bound dimension in y direction is {:d}".format(len(dw)))
            print("Truncation error would be {:.3e}".format(SP1err))
            print("Perform contraction along y direction...")
        # contraction
        if not isjax:
            Ap = ncon([B, A, w.conjugate(), w],
                  [[1,4,-3,2],[3,5,2,-4],[1,3,-1],[4,5,-2]])
        else:
            Ap = jncon([B, A, w, w],
                  [[1,4,-3,2],[3,5,2,-4],[1,3,-1],[4,5,-2]])
        if verbose:
            print("Contraction in y direction finished!\n")
    elif direction == "h":
        if verbose:
            print("Coarse graining in x direction...")
        if not isjax:
            w,dw,SP2err = xProjectorAB(B, A, chi, cg_eps,
                                       evenTrunc = evenTrunc)
        else:
            w,dw,SP2err = xProjectorAB(B, A, chi, isjax = True)
        if verbose:
            print("Spectrum of B @ A is:")
            dwarr = dw / dw.max()
            dwarr = np.abs(dwarr.to_ndarray())
            dwarr = - np.sort(-dwarr)
            print(dwarr[:10])
            print("Bound dimension in y direction is {:d}".format(len(dw)))
            print("Truncation error would be {:.3e}".format(SP2err))
            print("Perform contraction along x direction...")
        # contraction
        if not isjax:
            Ap = ncon([B, A, w.conjugate(), w],
                   [[-1,2,1,4],[2,-2,3,5],[1,3,-3],[4,5,-4]])
        else:
            Ap = jncon([B, A, w, w],
                   [[-1,2,1,4],[2,-2,3,5],[1,3,-3],[4,5,-4]])
        if verbose:
            print("Contraction in x direction finished!\n")
    else:
        raise ValueError("variable direction can only choose between h and v.")
    return Ap, w


def oneHOTRG(A, allchi, isfixGauge = False, verbose = False,
             isGilt = False, gilt_eps = 1e-7, cg_eps = 1e-6,
             N_gilt = 2, legcut = 2, loop_red_scheme = "Gilt",
             argsFET = {}, RoptVerbose = False,
             suggestChiABvh = [[False, 1, 1], [False, 1, 1]],
             evenTrunc = False):
    """
    One HOTRG step in 2d. It first contracts in y direction and then
    x direction.
    Computational cost:
        1. memory cost is chi^5, coming from HOTRG contraction step,
        can be reduced to chi^4 but not done here.
        2. cpu cost is 2 * chi^7, coming from HOTRG contracting .
    Parameters
    ----------
    A : 4-leg tensor A[i,j,k,l], it should be properly normalized
          k|
        i--A--j.
          l|.
    allchi : list, allchi = [chi_H, chi_V]
        horizontal bound dimension and verticle one.
    isfixGauge : boolean, optional
        whether to fixed the sign ambiguity. The default is False.
    verbose : boolean, optional
        whether to print out information. The default is False.
    isGilt : boolean, optional
        whether to apply the Gilts to truncation the input tensor A.
        The default is False.

    Returns
    -------
    Aout : 4-leg tensor Aout[i,j,k,l], properly normalized version
        renormalized tensor, order of the leg is the same as A.
    Anorm : float
        Norm of the renormalized tensor.
    SPerr : list, SPerr = [SP1err, SP2err]
        Error of projective truncation in HOTRG.

    """
    # read horizontal and vertical bound dimensions
    chiH, chiV = allchi
    # make sure the data type of tensor A is abelieantensor
    A = convertAbe(A)
    if not isGilt:
        # This is just plain HOTRG
        if verbose:
            print("Perform coarse graining without loop reductions.")
        Ap, w = halfHOTRG(A, A, chiH, direction = "v", verbose = verbose,
                          cg_eps = cg_eps, evenTrunc = evenTrunc)
        Ap, v = halfHOTRG(Ap, Ap, chiV, direction = "h", verbose = verbose,
                          cg_eps = cg_eps, evenTrunc = evenTrunc)

        if verbose:
            print("HOTRG without loop reductions finished!")
            print("------------------------------")
        Anorm = Ap.norm()
        Ap = Ap / Anorm
        isometries, RABs, RABsh = 0, 0, 0
    else:
        # This is Gilt-HOTRG
        if verbose:
            print("Perform coarse graining with {:s}.".format(loop_red_scheme))
            print("First perform {:s} for y direction...".format(loop_red_scheme))
        # be careful with the different convention of
        # order of legs in Gilts and HOTRG
        suggestChiv = suggestChiABvh[0]
        Ap, Bp, RABs = gilt_hotrgplaq(A.transpose([0,2,1,3]),
                                A.transpose([0,2,1,3]),
                                gilt_eps, verbose = verbose, legcut = legcut,
                                loop_red_scheme = loop_red_scheme,
                                argsFET = argsFET, RoptVerbose = RoptVerbose,
                                suggestChiAB = suggestChiv)
        # transpose the order of leg back to HOTRG convention
        Ap = Ap.transpose([0,2,1,3])
        Bp = Bp.transpose([0,2,1,3])
        Ap, w = halfHOTRG(Bp, Ap, chiH, direction = "v", verbose = verbose,
                          cg_eps = cg_eps, evenTrunc = evenTrunc)

        if N_gilt == 2:
            if verbose:
                print("Perform {:s} for x direction...".format(loop_red_scheme))
            suggestChih = suggestChiABvh[1]
            Ap, Bp, RABsh = gilt_hotrgplaq(Ap.transpose([0,2,1,3]),
                                    Ap.transpose([0,2,1,3]),
                                    gilt_eps, verbose = verbose,
                                    direction = "h", legcut = legcut,
                                    loop_red_scheme = loop_red_scheme,
                                    argsFET = argsFET, RoptVerbose = RoptVerbose,
                                    suggestChiAB = suggestChih)
            # transpose the order of leg back to HOTRG convention
            Ap = Ap.transpose([0,2,1,3])
            Bp = Bp.transpose([0,2,1,3])
            Ap, v = halfHOTRG(Bp, Ap, chiV, direction = "h", verbose = verbose,
                              cg_eps = cg_eps, evenTrunc = evenTrunc)
        elif N_gilt == 1:
            Ap, v = halfHOTRG(Ap, Ap, chiV, direction = "h", verbose = verbose,
                              cg_eps = cg_eps, evenTrunc = evenTrunc)
            RABsh = 0
        else:
            raise ValueError("argument N_gilt can only be chosen between 1 and 2")
        if verbose:
            print("HOTRG with loop reductions finished!\n")
        Anorm = Ap.norm()
        Ap = Ap / Anorm
        # Perform sign fixing
        signfixDone = False
        if isfixGauge:
            # print("Norm of A and Ap is {} and {}".format(A.norm(),Ap.norm()))
            assert np.abs(Ap.norm() - A.norm()) < 1e-10, "Ap is not properly normalized as A."
            if Ap.shape == A.shape:
                Ap, dh, dv = fixBestSign(Ap, A)
                signfixDone = True
        # absorb sign fixing matrix into w, v isometries
        w = w.to_ndarray()
        v = v.to_ndarray()
        if signfixDone:
            # we use broadcase feature of numpy.array here
            # the 1-d array dh (or dv) will multiply the last index of w (or v)
            w = w * dh
            v = v * dv

        isometries = [w,v]

        # check we can recover Ap by using isometries and RABs directly.
        # This part is purely for debug
        if verbose:
            print("Let's check whether we can reproduce Ap by acting " +
                  "isometries and RABs on A...")
            RA, RB, RAl, RAr, RBl, RBr = RABs
            if loop_red_scheme == "Gilt":
                if legcut == 2:
                    # Apcheck = ncon([A, RA], [[-1, -2, -3, 1], [1, -4]])
                    # Bpcheck = ncon([A, RB], [[-1, -2, 1, -4], [1, -3]])
                    Apcheck = ncon([A, RAl, RAr], [[1, 2, -3, -4], [1,-1], [2,-2]])
                    Bpcheck = ncon([A, RBl, RBr], [[1, 2, -3, -4], [1,-1],[2,-2]])
                elif legcut == 4:
                    Apcheck = ncon([A, RA, RAl, RAr], [[1, 2, -3, 4], [4, -4],
                                                        [1,-1], [2,-2]])
                    Bpcheck = ncon([A, RB, RBl, RBr], [[1, 2, 3, -4], [3, -3],
                                                        [1,-1],[2,-2]])
                else:
                    raise ValueError("legcut argument can only be chosen between 2 and 4.")
            elif loop_red_scheme == "FET":
                Apcheck = ncon([A, RAl, RAr], [[1, 2, -3, -4], [1,-1], [2,-2]])
                Bpcheck = ncon([A, RBl, RBr], [[1, 2, -3, -4], [1,-1],[2,-2]])

            # perform half HOTRG
            Apcheck = doHalfHOTRGknownWV(Bpcheck, Apcheck, w, direction = "v")

            # horizontal procedure start
            if N_gilt == 1:
                Apcheck = doHalfHOTRGknownWV(Apcheck, Apcheck, v, direction = "h")
            elif N_gilt == 2:
                RA, RB, RAl, RAr, RBl, RBr = RABsh
                Apcheck = convertAbe(Apcheck)
                RAl = convertAbe(RAl.to_ndarray())
                RAr = convertAbe(RAr.to_ndarray())
                RBl = convertAbe(RBl.to_ndarray())
                RBr = convertAbe(RBr.to_ndarray())
                if loop_red_scheme == "Gilt":
                    if legcut == 2:
                        # Apcheck = ncon([Ap, RA], [[-1,2,-3,-4], [2,-2]])
                        # Bpcheck = ncon([Ap, RB], [[1,-2,-3,-4], [1,-1]])
                        Appcheck = ncon([Apcheck, RAl, RAr], [[-1,-2,3,4], [4,-4], [3,-3]])
                        Bppcheck = ncon([Apcheck, RBl, RBr], [[-1,-2,3,4], [4,-4], [3,-3]])
                    elif legcut == 4:
                        Appcheck = ncon([Apcheck, RA, RAl, RAr], [[-1,2,3,4], [2,-2],
                                                            [4,-4], [3,-3]])
                        Bppcheck = ncon([Apcheck, RB, RBl, RBr], [[1,-2,3,4], [1,-1],
                                                            [4,-4], [3,-3]])
                elif loop_red_scheme == "FET":
                    Appcheck = ncon([Apcheck, RAl, RAr], [[-1,-2,3,4], [4,-4], [3,-3]])
                    Bppcheck = ncon([Apcheck, RBl, RBr], [[-1,-2,3,4], [4,-4], [3,-3]])
                Apcheck = doHalfHOTRGknownWV(Bppcheck, Appcheck, v, direction = "h")
            Apcheck = Apcheck / LA.norm(Apcheck)
            print("Difference of Apcheck and Ap? ")
            print(LA.norm(Ap.to_ndarray() - Apcheck))
            print("-------------------------------\n")


    return Ap, Anorm, isometries, RABs, RABsh

## The isometric tensors are given, contract them with original tensors

def doHalfHOTRGknownWV(B, A, w, direction = "v"):
    B = convertAbeBack(B)
    A = convertAbeBack(A)
    if direction == "v":
        Ap = jncon([B, A, w.conjugate(), w],
                  [[1,4,-3,2],[3,5,2,-4],[1,3,-1],[4,5,-2]])
    elif direction == "h":
        Ap = jncon([B, A, w.conjugate(), w],
                   [[-1,2,1,4],[2,-2,3,5],[1,3,-3],[4,5,-4]])
    else:
        raise ValueError("Variable direction can only choose between h and v.")
    return Ap

## Calculate the singluar value spectrum of tensor A
def get_spectrum_A(A, leftgrp = [0,2], rightgrp = [1,3]):
    s = A.svd(leftgrp,rightgrp)[1]
    s = s/s.max()
    s = s.to_ndarray()
    s = np.abs(s)
    s = -np.sort(-s)
    return s


def normFlowHOTRG(relT, allchi, iterN = 12, isDisp = True, isfixGauge = False,
                  isGilt = False, isSym = False, gilt_eps = 1e-7,
                  cg_eps = 1e-6, return_sing = False,
                  N_gilt = 1, legcut = 4, loop_red_scheme = "Gilt",
                  argsFET = {}, stableStep = 9, saveData = [False, '']):
    """
    Similar to iterHOTRG, but here we just keep track of all the
    tensor norms. So this function is desgined to analyze the flow
    of tensor norms


    If saveData[0] is False, we simply save all data into a list
    and output them after all iterations.
    If saveData[0] is True, we will save tensor A into the directory
    saveData[1] of the disk before updating
    """
    Tc = 2 / np.log(1 + np.sqrt(2))
    beta = 1 / Tc / relT
    A0 = Ising2dT(beta, isSym = isSym)
    ## transform to the convention of HOTRG
    A0 = A0.transpose([0,2,1,3])
    # list for saving data
    Anorm = [0 for x in range(iterN +1)]
    Anorm[0] = A0.norm()
    A0 = A0 / Anorm[0]
    isomlist = [0 for x in range(iterN)]
    RABslist = [0 for x in range(iterN)]
    RABshlist = [0 for x in range(iterN)]
    # save A0 if desired
    if saveData[0]:
        savedirA = saveData[1] + "/As"
        # create the directory for saving A0 if it doesn't exist
        if not os.path.exists(savedirA):
            os.makedirs(savedirA)
        savefileA = savedirA + "/A00.pkl"
        with open(savefileA, "wb") as f:
            pkl.dump(A0, f)
    # whether to control squeezed chi or not
    suggestChiABvh = [[False, 1, 1], [False, 1, 1]]
    # whether to impose the condition that the HOTRG trunction
    # will be evenly distributed between (0, 0) and (1, 1) sector
    evenTrunc = False
    # to save singular values flow
    singlist = []
    Adifflist = []
    if return_sing:
        singlist.append(get_spectrum_A(A0))
    # enter the RG loop
    for k in range(iterN):
        if isDisp:
            print("Performing {:d}-th HOTRG...".format(k+1))
        # suggest chi after stableStep-th RG step
        if k + 1 >= stableStep  and isGilt:
            # for Vertical loop reduction
            oldRAl = RABslist[k-1][2]
            oldRBl = RABslist[k-1][4]
            suggChiA = oldRAl.to_ndarray().shape[1]
            suggChiB = oldRBl.to_ndarray().shape[1]
            suggestChiv = [True, suggChiA, suggChiB]
            # for horizontal loop reduction
            oldRAlh = RABshlist[k-1][2]
            oldRBlh = RABshlist[k-1][4]
            suggChiAh = oldRAlh.to_ndarray().shape[1]
            suggChiBh = oldRBlh.to_ndarray().shape[1]
            suggestChih = [True, suggChiAh, suggChiBh]
            suggestChiABvh = [suggestChiv, suggestChih]
            # for even truncation
            evenTrunc = True
            if isDisp:
                print("We use suggested squeezed bound dimension in this step.")
                print("We also make sure HOTRG truncation is even in two sectors.")
        # perform one Gilt-HOTRG step
        A0cur, Anorm[k+1], isomlist[k], RABslist[k], RABshlist[k] = oneHOTRG(A0, allchi, isfixGauge, isDisp,
                                  isGilt, gilt_eps, cg_eps,
                                  N_gilt = N_gilt, legcut = legcut,
                                  loop_red_scheme = loop_red_scheme,
                                  argsFET = argsFET,
                                  suggestChiABvh = suggestChiABvh,
                                  evenTrunc = evenTrunc)
        # calculate difference of adjacent A0
        if return_sing:
            if A0cur.shape == A0.shape:
                Adifflist.append((A0cur - A0).norm())
            else:
                Adifflist.append(1)
        # update A0
        A0 = A0cur * 1.0
        # save updated A0 to disk
        if saveData[0]:
            savefileA = savedirA + "/A{:02d}.pkl".format(k + 1)
            with open(savefileA, "wb") as f:
                pkl.dump(A0, f)
        # end of first if
        if return_sing:
            singlist.append(get_spectrum_A(A0))
        # print out informations
        if isDisp:
            s = get_spectrum_A(A0)
            print("Spectrum of A:")
            with np.printoptions(precision = 3, suppress = None):
                print(s[:30])
            print("{:d}-th HOTRG finished!".format(k+1))
            print("----------------------------------------------------")
            print("----------------------------------------------------\n")
        # end of for
    # save all other lists
    if saveData[0]:
        savefileOther = saveData[1] + "/otherTs.pkl"
        with open(savefileOther, "wb") as f:
            pkl.dump([Anorm, isomlist, RABslist, RABshlist], f)
    return Anorm, singlist, Adifflist




def diffGiltHOTRG(A, Anorm, isom, RABs, RABsh, scaleN = 20,
               isom_corr = False):
    """
    Similar as diffRGnew, but designed for Gilt-HOTRG-imp version
    """
    # define the invariant tensor where the magnetitute is properly taken care of
    Ainv = Anorm**(-1/3) * A
    # read of isometries and R matrices used in Gilt
    w, v = isom
    RAl, RAr, RBl, RBr = RABs[2:]
    RAlh, RArh, RBlh, RBrh = RABsh[2:]
    # convert everything to numpy.array for consistent, since we will
    # fall back to ordinary tensor multiplication in the calculation here
    Ainv = convertAbeBack(Ainv)
    N1, N2, N3, N4 = Ainv.shape
    w = convertAbeBack(w)
    v = convertAbeBack(v)
    RAl = convertAbeBack(RAl)
    RAr = convertAbeBack(RAr)
    RBl = convertAbeBack(RBl)
    RBr = convertAbeBack(RBr)

    RAlh = convertAbeBack(RAlh)
    RArh = convertAbeBack(RArh)
    RBlh = convertAbeBack(RBlh)
    RBrh = convertAbeBack(RBrh)

    # define the RG equation
    def equationRG(psiA):
        Aorg = psiA.reshape(N1,N2,N3,N4)
        # Gilt before y-contraction
        Ap = jncon([Aorg, RAl, RAr], [[1, 2, -3, -4], [1,-1], [2,-2]])
        Bp = jncon([Aorg, RBl, RBr], [[1, 2, -3, -4], [1,-1], [2,-2]])
        # perform HOTRG y-contraction
        if not isom_corr:
            Ap = doHalfHOTRGknownWV(Bp, Ap, w, direction = "v")
        else:
            chiH = w.shape[2]
            Ap = halfHOTRG(Bp, Ap, chiH, direction = "v", verbose = False,
                           isjax = True)[0]
        # Gilt before x-contraction
        App = jncon([Ap, RAlh, RArh], [[-1,-2,3,4], [4,-4], [3,-3]])
        Bpp = jncon([Ap, RBlh, RBrh], [[-1,-2,3,4], [4,-4], [3,-3]])
        # perform HOTRG x-contraction
        if not isom_corr:
            Ap = doHalfHOTRGknownWV(Bpp, App, v, direction = "h")
        else:
            chiV = v.shape[2]
            Ap = halfHOTRG(Bpp, App, chiV, direction = "h", verbose = False,
                           isjax = True)[0]
        psiAp = Ap.reshape(N1 * N2 * N3 * N4)
        return psiAp
    # linearlized the RG equation to get response matrix
    dimA = N1 * N2 * N3 * N4
    psiA = Ainv.reshape(dimA)
    psiAp, responseMat = jax.linearize(equationRG, psiA)
    # calculate its eigenvalues
    RGhyperM = LinearOperator((dimA,dimA), matvec = responseMat)
    dtemp = np.sort(abs(eigs(RGhyperM, k=scaleN,
                    which='LM', return_eigenvectors=False)))
    dtemp = dtemp[::-1]
    # calculate scaling dimensions
    scDims = -np.log2(abs(dtemp/dtemp[0]))
    return scDims


## test client here
if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        "Test implementation of HOTRG.")
    parser.add_argument("func", type = str,
                        help = "the function to be tested",
                        choices = ["old-Gilt-HOTRG", "Gilt-HOTRG-imp-gerr",
                                   "oneHOTRG", "Gilt-Full-HOTRG-gerr"],
                        default = "old-Gilt-HOTRG")
    parser.add_argument("--chi", dest = "chi", type = int,
                    help = "bound dimension (default: 10)",
                    default = 10)
    parser.add_argument("--relT", dest = "relT", type = float,
                    help = "relative temperature to Tc (default: 1)",
                    default = 1.0)
    parser.add_argument("--iterN", dest = "iterN", type = int,
                    help = "Number of HOTRG iteration (default: 24)",
                    default = 24)
    parser.add_argument("--isGilt", help = "whether to use Gilts",
                        action = "store_true")
    parser.add_argument("--isSym", help = "whether to use Z2 symmetric tensor",
                        action = "store_true")
    parser.add_argument("--gilteps", dest = "gilteps", type = float,
                        help = "a number smaller than which we think the" +
                        "singluar values for the environment spectrum is zero" +
                        "(default: 1e-7)",
                        default = 1e-7)
    parser.add_argument("--returnIsoRuv",
                       help = "whether to return isometries and Ruv matrices",
                       action = "store_true")
    parser.add_argument("--Ngilt", dest = "Ngilt", type = int,
                       help = "How many times do we perform Gilt in oneHOTRG",
                       choices = [1,2], default = 1)
    parser.add_argument("--legcut", dest = "legcut", type = int,
                       help = "number of leg to cut in gilt_hotrgplaq",
                       choices = [2,4], default = 4)
    parser.add_argument("--loopred", dest = "loopred", type = str,
                    help = "which loop reduction scheme to use",
                    choices = ["Gilt", "FET"], default = "Gilt")
    parser.add_argument("--chitid", dest = "chitid", type = int,
                    help = "bound dimension (default: 5)",
                    default = 5)
    parser.add_argument("--stbk", dest = "stbk", type = int,
                    help = "A int after which we will try to stabilize the gilt process",
                    default = 8)


    args = parser.parse_args()
    testFun = args.func
    chi = args.chi
    relT = args.relT
    iterN = args.iterN
    isGilt = args.isGilt
    isSym = args.isSym
    gilteps = args.gilteps
    return_iso_Ruvs = args.returnIsoRuv
    Ngilt = args.Ngilt
    legcut = args.legcut
    loopred = args.loopred
    chitid = args.chitid
    stablek = args.stbk

    argsFET = {'chitid':chitid, 'maxiter':20, 'initscheme':'Gilt',
               'giltdeg':0.5}
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d. %H:%M:%S")
    print("Running Time =", current_time)
    if testFun == "oneHOTRG":
        from gilts import cdl
        sqrtchi = 3
        cdlchi = sqrtchi ** 2
        cdlT, numb = cdl(sqrtchi,1, isSym)
        # rotation order of leg to HOTRG convention
        cdlT = cdlT.transpose([0,2,1,3])
        print("Test oneHOTRG on a random CDL tensor...")
        Ap, Anorm,isometries,RABs, RABsh = oneHOTRG(cdlT, [cdlchi]*2,
                                             verbose = True, isGilt = True,
                                             N_gilt = Ngilt, legcut = legcut,
                                             loop_red_scheme = loopred,
                                   argsFET = argsFET,
                                   cg_eps = 1e-4)
        extAp =numb ** 4
        errAp = (extAp - Anorm) / extAp
        print("Difference from exact Ap is {:.3e}".format(errAp))
        print("Test finished!!")

