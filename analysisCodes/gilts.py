#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:36:10 2020

Numpy implementation

This is a implementation of the graph-independent local truncations
described in paper,
Markus Hauru, Clement Delcamp, and Sebastian Mizera,
Phys. Rev. B 97, 045111 – Published 10 January 2018
url: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045111

On Oct 15 2020, I added Evenbly's full environment truncation (FET)
@author: brucelyu
"""

import numpy as np
from ncon import ncon
import itertools as itt
from abeliantensors import Tensor, TensorZ2

## Some useful functions

def cdl(sqrtchi = 2, corechi = 1, isSym = False):
    """
    generate a random corner double line tensor

    Parameters
    ----------
    sqrtchi : integer, optional
        dimension of the matrix, equal to square root of the bound
        dimension chi. The default is 2.

    Returns
    -------
    cdlT : four-leg tensor cdlT[i,j,k,l]
        with bound dimension chi = sqrtchi^2.

    """
    if not isSym:
        A, B, C, D = [Tensor.random([sqrtchi, sqrtchi]) for i in range(4)]
        coreT = Tensor.random([corechi] * 4)
        coreT = coreT / coreT.norm()
    else:
        A, B, C, D = [TensorZ2.random(shape = [[sqrtchi], [sqrtchi]],
                                      qhape = [[0],[0]], dirs = [1,-1]) for i in range(4)]
        coreT = TensorZ2.random(shape = [[corechi]]*4 ,
                                qhape = [[0]]*4, dirs = [1,1,-1,-1])
        coreT = coreT / coreT.norm()
        A = A.to_ndarray()
        B = B.to_ndarray()
        C = C.to_ndarray()
        D = D.to_ndarray()
        coreT = coreT.to_ndarray()
    cdlT = A[:,None,None,None,None,None,:,None] \
        * B[None,:,:,None,None,None,None,None]  \
        * C[None,None,None,:,None, :,None,None] \
        * D[None,None,None,None,:,None,None,:]
    if isSym:
        cdlT = TensorZ2.from_ndarray(cdlT, shape = [[sqrtchi]]*8, qhape = [[0]]*8,
                                 dirs = [1,1,1,1,-1,-1,-1,-1])
        cdlT = cdlT.join_indices([0,1],[2,3],[4,5],[6,7], dirs = [1,1,-1,-1])
        cdlT = cdlT.to_ndarray()
        totT = np.kron(coreT, cdlT)
        totT = TensorZ2.from_ndarray(totT, shape = [[sqrtchi**2 * corechi]]*4,
                                     qhape = [[0]]*4, dirs = [1,1,-1,-1])
    else:
        cdlT = cdlT.reshape([sqrtchi**2]*4)
        totT = np.kron(coreT, cdlT)

    numb = ncon([A,B,C,D],[[1,2],[3,2],[4,3],[1,4]])
    return totT, numb

def rotateLattice(Alist, Leg = "I"):
    """
    rotate the lattice so that the leg goes to where the
    leg "I" used to stay

    Parameters
    ----------
    Alist : list
        containing four tensors in a plaquette.
    Leg : tr, choose among ("I","II","III","IV")
        the leg whose environment info is calculated. The default is "I".

    Returns
    -------
    Arotlist : list
        representing the rotated list.

    """
    # We just need to write the code for leg I. Leg II, III and IV are obtained
    # by rotating the diagram by 90, 180 and 270 degrees. Notice this will
    # change both the order of tensors and the order of their legs
    legdic = {"I":0, "II":1, "III":2, "IV":3}
    if Leg not in legdic.keys():
        raise ValueError("Leg should be choisen among (I,II,III,IV)!")
    perm = np.array([i for i in range(4)])
    curperm = list(np.mod(perm + legdic[Leg],4))
    Arotlist = []
    for k in range(4):
        Arotlist.append(Alist[curperm[k]].transpose(curperm))
    return Arotlist, curperm

def envLeg(Alist,Leg = "I", return_envMat = False):
    """
    Return the enviroment spectrum s and tensor u of one leg in a sqaure
    plaquette made up of four leg tensor A[ijkl]:
       j|
     i--A--k.
       l|
     A1,A2,A3,A4 = Alist, the plaquette looks like
        | (I) |
     ---A1----A2---
    (IV)|     |(II)
        |     |
     ---A4----A3---
        |(III)|
    See my notes in iPad for the detailed implementation process

    Parameters
    ----------
    Alist : list
        list containing four tensors in a plaquette.
    Leg : str, choose among ("I","II","III","IV")
        the leg whose environment info is calculated. The default is "I".

    Returns
    -------
    U : three leg tensor U[ijk]
        Orthogonal matrix containing eigvenvectors of the environment matrix.
           |k
        i--U--j
    s : vector s[i]
        sqaure root of eigenvalues of the environment matrix.

    """
    # We just need to write the code for leg I. Leg II, III and IV are obtained
    # by rotating the diagram by 90, 180 and 270 degrees. Notice this will
    # change both the order of tensors and the order of their legs
    [Arot1, Arot2, Arot3, Arot4], _ = rotateLattice(Alist, Leg)
    # calculate the environment matrix and eigenvalue decompose it.
    legenv = ncon([Arot1, Arot2, Arot3, Arot4,
                   Arot1.conjugate(), Arot2.conjugate(),
                   Arot3.conjugate(), Arot4.conjugate()],
                  [[7,8,-1,9],[-2,12,11,13],[5,13,4,3],[2,9,5,1],
                   [7,8,-3,10],[-4,12,11,14],[6,14,4,3],[2,10,6,1]])
    d,U = legenv.eig([0,1],[2,3], hermitian = True)
    # abs is taken for numerical errors around zero
    s = d.abs().sqrt()
    if return_envMat:
        return U, s, legenv
    return U, s


###############################
# functions for Gilt starts...

def tvec(U):
    """
    Calculate t vector from U tesnor
    Parameters
    t--k =    |k
            --U--
            |____|
    ----------
    U : three leg tensor U[i,j,k]
        singular vectors (eigenvectors of an environment matrix).

    Returns
    -------
    t : t[k]

    """
    t = ncon([U],[[1,1,-1]])
    return t

def topt(t,s,epsilon = 1e-7):
    """
    One optimization method to choose t' such that the rank of
    the resultant Rp is low
    Parameters
    ----------
    t : vector t[k]
        constructed from tensor U
           |k
         --U--
        |____|
    s : vector s[i]
        singular spectrum of an eivironment matrix.
    epsilon : float, optional
         a small number below which we think the singular value is zero.
         The default is 1e-7.

    Returns
    -------
    tprime : vector tprime[k]
        optimized t.

    """
    # properly normalize the environment spectrum if it is not
    s = s / s.norm()
    # For Z2 symmetric tensor
    t = t.flip_dir(0)
    # choose t' according to eq.(31) in the paper
    if epsilon !=0:
        ratio = s / epsilon
        weight = ratio ** 2 / (1 + ratio**2)
        tprime = t.multiply_diag(weight, 0 , direction="left")
    else:
        tprime = t * 1.0
    return tprime

def Rtensor(U,t):
    """
    Calculate R tensor from U and t
                 t
    i--R--j =    |
              i--U--j

    Parameters
    ----------
    U : three leg tensor U[ijk]
           |k
        i--U--j
    t : vector t[k].

    Returns
    -------
    R : matrix R[i,j]
        a low rank matrix to truncate a leg..

    """
    R = ncon([U.conjugate(),t.conjugate()],[[-1,-2,1],[1]])
    return R


def Ropt(U, s, epsilon = 1e-7, convergence_eps = 1e-2, counter = 1,
         verbose = False, suggestChi = [False, 1]):
    """
    Given the environment spectrum s and the singular vectors U, choose
    t' and build the matrix R' (called tp and Rp in the code).
    Return also the truncation error caused in inserting this Rp into
    the environment. We use run this truncation recursively to get rid of
    CDL tensors

    Parameters
    ----------
    U : three leg tensor U[ijk]
           |k
        i--U--j
        singular vectors (eigenvectors of an environment matrix).
    s : vector s[i]
        singular spectrum of an eivironment matrix.
    epsilon : float, optional
        a small number below which we think the singular value is zero.
        The default is 1e-7.
    convergence_eps : float, optional
        difference from identity matrix smaller than which
        we think we get a good approximation. The default is 1e-2.
    counter : int, optional
        to counter the depth of the recursion. The default is 1.

    Returns
    -------
    Rprime : matrix Rprime[i,j]
        i--Rprime--j
        a low rank matrix to truncate a leg.
    counter : int
        to counter the depth of the recursion.
    """
    # perform one step of the truncation
    t = tvec(U)
    tprime = topt(t, s, epsilon)
    Rprime = Rtensor(U, tprime)
    # examine the singular value spectrum of Rprime and enter the recursion
    spliteps = epsilon * 1e-3 # small number for trucated svd
    Ruprime, Rs, Rvprime = Rprime.split([0],[1], return_sings = True,
                                        eps = spliteps)
    # if we input a suggested chi after squeezing
    suggestDone = False
    if suggestChi[0]:
        # original consideration
        suggChi = suggestChi[1]
        currentChi = Rs.to_ndarray().shape[0]
        if currentChi == suggChi and (Rs/Rs.max() - 1).abs().max() < 1e-1:
            suggestDone = True
    # convert Rs to numpy.array
    Rsarr = Rs.to_ndarray()
    Rsarr = -np.sort(-Rsarr)
    ## testing code starts
    if verbose:
        print("We are now in recursion #{:d}".format(counter))
        print("Spectrum of R' is:")
        print(Rsarr[:20])
    ## testing code ends
    done_recursing = (Rs/Rs.max() - 1).abs().max() < convergence_eps
    done_recursing = done_recursing or suggestDone
    if (not done_recursing):
        URR = ncon([U, Ruprime, Rvprime],[[1,2,-3], [1,-1],[-2,2]])
        URRs = URR.multiply_diag(s.flip_dir(0), 2, direction = "left")
        # D1,D2,D3 = URRs.shape
        # URRsM = URRs.reshape(D1*D2, D3)
        # Uinnver, sinner, Vinner = LA.svd(URRsM, full_matrices=False)
        # Uinnver = Uinnver.reshape(D1,D2,-1)
        # sinner = sinner / sinner[0]
        Uinner, sinner = URRs.svd([0,1],[2])[:2]
        sinner = sinner / sinner.sum()
        Rprimeinner, counter = Ropt(Uinner, sinner, epsilon,
                                    convergence_eps, counter = counter+1,
                                    verbose = verbose,
                                    suggestChi = suggestChi)
        Rprime = ncon([Rprimeinner, Ruprime, Rvprime],[[1,2],[-1,1],[2,-2]])
    return Rprime, counter

def gilt_err(U,s, Ru, Rv):
    """
    Calculate the error for truncation of one leg using Gilts.
    The error is defined in eq. (B1) in Markus's paper

    Parameters
    ----------
    U : three leg tensor U[ijk]
           |k
        i--U--j
        singular vectors (eigenvectors of an environment matrix).
    s : vector s[i]
        singular spectrum of an eivironment matrix.
    Ru : matrix
        left piece of Rprime.
    Rv : matrix
        right piece of Rprime.

    Returns
    -------
    err : float
        error of Gilts truncation.

    """
    t = tvec(U)
    tp = ncon([U, Ru, Rv],[[1,2,-1], [1,3],[3,2]])
    s = s.flip_dir(0)
    diff = t - tp
    diff = diff * s
    err = diff.norm() / (t * s).norm()
    return err

def cutLeg(A,B,Leg,epsilon = 1e-7,convergence_eps = 1e-2, verbose = True,
           forwhat = 'trg', suggestChi = [False, 1]):
    """
    Given a plaquette [A, B, A, B] that looks like
        | (I) |
     ---A-----B---
    (IV)|     |(II)
        |     |
     ---B-----A---
        |(III)|
    and also the leg to be cut. Return left and right R matrix

    Parameters
    ----------
    A : Four leg tensor A[i,j,k,l]
         j|
       i--A--k.
         l|.
    B : Four leg tensor B[i,j,k,l]
        same as A.
    Leg : str, choose among ("I","II","III","IV")
        the leg to be truncated.
    epsilon : float, optional
        a small number below which we think the singular value is zero.
        The default is 1e-7.
    convergence_eps : float, optional
        difference from identity matrix smaller than which
        we think we get a good approximation. The default is 1e-2.

    Returns
    -------
    Ruprime : matrix
        left piece of R'.
    Rvprime : matrix
        right piece of R'.
    done : boolean
        whether the spectrum of R is flat.
    err : float
        truncation error of gilts.
    counter : int
        the depth of the recursion.

    """
    if forwhat == 'trg':
        Alist = [A, B, A, B]
    elif forwhat == 'hotrg':
        Alist = [A, A, B, B]
    else:
        raise ValueError("forwhat should be chosen between trg or hotrg")
    U0, s0 = envLeg(Alist, Leg)
    if verbose:
        s0normed = s0 / s0.max()
        s0normed = s0normed.to_ndarray()
        # print("The environment singular value spectrum of leg {:s} is:".format(Leg))
        # print(s0normed[:16])
        cutpercent = np.sum(s0normed < epsilon) / len(s0normed)
        print("We are throwing away {:.0%} of the smallest singular values of leg environment.".format(cutpercent))
    Rprime, counter = Ropt(U0, s0, epsilon, convergence_eps, verbose = False,
                           suggestChi = suggestChi)

    spliteps = epsilon * 1e-3 # small number for trucated svd
    # Remember we should make sure Rprime is close to identity,
    # which means its singular value spectrum should all be close to 1
    Ruprime, Rs, Rvprime = Rprime.split([0],[1], return_sings=True,
                                        eps = spliteps)
    done = (Rs/Rs.max() - 1).abs().max() < convergence_eps

    # calculate gilts truncation error
    err = gilt_err(U0, s0, Ruprime, Rvprime)
    if verbose:
        print("Singular value spectrum of final $R^n$ is")
        print(Rs.to_ndarray())
    return Ruprime, Rvprime, done, err, counter

# end of for Gilt.
###############################
#
#
#
###############################
# funtions for FET starts... (not used in my tensor RG paper, but is
# a potential alternative of the Gilt. 
# I've checked that the Gilt is a better choice)

def envLegUpdate(Alist,u, s, v, Leg = "I", forwhat = "hotrg"):
    """
    Calcuate leg environments and fidelity of a FET scheme
    designed for HOTRG.
    Compared with the scheme proposed in the original paper,
    here the u, s, v will be mixed into leg environment

    For information of arguments, see function engLeg and
    fidelity for details

    """
    udots = u.multiply_diag(s, axis = 1, direction = 'r')
    Gamma = envLeg(Alist, Leg, return_envMat = True)[2]
    if forwhat == "hotrg":
        assert Leg == "I" or Leg == "III", "Leg can only be I or III in HOTRG + FET"
        [Arot1, Arot2, Arot3, Arot4] = rotateLattice(Alist, Leg)[0]
        # absorb u, s, v into Arot1 to get Arot1usv
        Arot1usv = ncon([Arot1, udots, v], [[1, -2, -3, -4],
                                            [-1, 2], [2, 1]])
        # calculate the environment matrix and eigenvalue decompose it.
        GammaUpper = ncon([Arot1, Arot2, Arot3, Arot4,
                   Arot1usv.conjugate(), Arot2.conjugate(),
                   Arot3.conjugate(), Arot4.conjugate()],
                  [[7,8,-1,9],[-2,12,11,13],[5,13,4,3],[2,9,5,1],
                   [7,8,-3,10],[-4,12,11,14],[6,14,4,3],[2,10,6,1]])
        GammaDown = ncon([Arot1usv, Arot2, Arot3, Arot4,
                   Arot1usv.conjugate(), Arot2.conjugate(),
                   Arot3.conjugate(), Arot4.conjugate()],
                  [[7,8,-1,9],[-2,12,11,13],[5,13,4,3],[2,9,5,1],
                   [7,8,-3,10],[-4,12,11,14],[6,14,4,3],[2,10,6,1]])
        # calculate fidelity in this situation
        psipsi = ncon([Gamma], [1, 1, 2, 2]).norm()
        phiphi = ncon([GammaDown, udots, v, udots.conjugate(), v.conjugate()],
                  [[1, 2, 4, 5], [1, 3], [3, 2], [4, 6], [6, 5]]).norm()
        phipsi = ncon([GammaUpper, udots.conjugate(), v.conjugate()],
                  [[1, 1, 4, 5], [4, 6], [6, 5]]).norm()
        f = phipsi ** 2 / (phiphi * psipsi)
    return Gamma, GammaUpper, GammaDown, 1 - f


def pinv(B, a = [0, 1], b = [2, 3], eps_mach = 1e-10, debug = False):
    """
    Calculate pesudo inverse of positive semi-definite matrix B.
    We first perform eigenvalue decomposition of B = U d Uh, and only keep
    eigenvalues with d > 1e-10. B^-1 = U d^-1 Uh

    Parameters
    ----------
    B : 4-leg tensor
        DESCRIPTION.
    eps_mach : float, optional
        If the singular value is smaller than this, we set it to be 0.
        The default is 1e-10.

    Returns
    -------
    Binv : inverse of B
        DESCRIPTION.

    """
    def invArray(tensor):
        """
        Invert every element in each block of tensor (Abeliean tensor)

        """
        if type(tensor).__module__.split(".")[1] == 'symmetrytensors':
            invtensor = tensor.copy()
            for mykey in tensor.sects.keys():
                invtensor[mykey] = 1 / tensor[mykey]
        else:
            invtensor = 1 / tensor
        return invtensor

    d, U = B.eig(a, b,hermitian = True, eps = eps_mach)
    if debug:
        print("Shape of d and U")
        print(d.shape)
        print(U.shape)
    dinv = invArray(d)
    contrLegU = list(-np.array(a) - 1) + [1]
    contrLegUh = list(-np.array(b) - 1) + [1]
    Ud = U.multiply_diag(dinv, axis = len(U.shape) - 1, direction = 'r')
    Binv = ncon([Ud, U.conjugate()], [contrLegU, contrLegUh])

    return Binv

def normalGamma(Gamma):
    GammaNorm = ncon([Gamma], [1,1,2,2]).norm()
    Gamma0 = Gamma / GammaNorm
    return Gamma0, GammaNorm

def fidelity(Gamma, u, s, v, debug = False, return_phiphi = False):
    """
    Fidelity defined in Evenbly's paper. Notice the leg environment
    Gamma should be properly normalized. It ranges from 0 to 1.
    If equal 1, the approximation is exact; if equal to 0, the approximation
    is bad.

    See linearOpt function for descriptions of input tensors

    """
    # We first properly normalized Gamma
    Gamma0 = normalGamma(Gamma)[0]
    # Multiply s into u first
    udots = u.multiply_diag(s, axis = 1, direction = 'r')
    phiphi = ncon([Gamma0, udots, v, udots.conjugate(), v.conjugate()],
                  [[1, 2, 4, 5], [1, 3], [3, 2], [4, 6], [6, 5]]).norm()
    phipsi = ncon([Gamma0, udots.conjugate(), v.conjugate()],
                  [[1, 1, 4, 5], [4, 6], [6, 5]]).norm()
    psiphi = ncon([Gamma0, udots, v], [[2, 3, 1, 1], [2, 4], [4, 3]]).norm()
    if debug == True:
        print("Do <psi|phi> and <phi|psi> have the same magnetitute? ")
        print("Their difference is {:.1e}".format(phipsi - psiphi))
    # if only phiphi is needed to fix the normalization constant of s
    f = phipsi * psiphi / phiphi
    if return_phiphi:
        return f, 1 - f, phiphi, phipsi, psiphi
    return f, 1 - f

def initusv(Gamma, chitid, spliteps = 1e-10, inituVer = 'org',
            debug = False):
    """
    Given the leg environment, initialize the u, s, v matrices.

    Parameters
    ----------
    Gamma : 4-leg tensor
        See linearOpt function for more description.
    chitid : int
        a smaller bound dimension to squeeze to

    Returns
    -------
    u, s, v matrices

    """
    # Normalized leg environment
    Gamma0 = normalGamma(Gamma)[0]
    P = ncon([Gamma0], [[-1, -2, 1, 1]])
    if inituVer == 'org':
        Binv = pinv(Gamma0, eps_mach = spliteps)
        R = ncon([Binv, P], [[-1, -2, 1, 2], [1, 2]])
        R = R.conjugate()
        u0, s0, v0 = R.svd([0],[1], chis = [i+1 for i in range(chitid)],
                           eps = spliteps, degeneracy_eps = 1e-20)
    elif inituVer == 'Morita':
        u0, s0, v0 = P.svd([0],[1], chis = [i+1 for i in range(chitid)],
                           eps = spliteps)
        u0 = u0.conjugate()
        s0 = s0.ones_like().flip_dir(0)
        v0 = u0.transpose().conjugate()
    elif inituVer == "skeleton":
        d, U = Gamma0.eig([0,1],[2,3], hermitian = True,
                       chis = 1, degeneracy_eps = 1e-20)
        if debug:
            print("d is ")
            print(d)
        dones = d.ones_like().flip_dir(0)
        UMat = ncon([U.conjugate(), dones], [[-1,-2,2], [2]])
        UMat.invar = True
        u0, s0, v0 = UMat.svd([0], [1], chis = [i+1 for i in range(chitid)],
                              eps = spliteps, degeneracy_eps = 1e-20)
        s0 = s0.ones_like()
        v0 = u0.transpose().conjugate()
    return u0, s0, v0

def initGilt(Gamma, chitid, spliteps = 1e-10, gilteps = 1e-6,
             verbose = False, N_recur = 5):
    """
    Estimate original u, s, v using gilt with recursion 5
    """
    def RoptCur(U, s, epsilon = 1e-7, counter = 1,
         verbose = False):
        # perform one step of the truncation
        t = tvec(U)
        tprime = topt(t, s, epsilon)
        Rprime = Rtensor(U, tprime)
        # examine the singular value spectrum of Rprime and enter the recursion
        Ruprime, Rs, Rvprime = Rprime.split([0],[1], return_sings = True)
        # convert Rs to numpy.array
        Rsarr = Rs.to_ndarray()
        Rsarr = -np.sort(-Rsarr)
        ## testing code starts
        if verbose:
            print("We are now in recursion #{:d}".format(counter))
            print("Spectrum of R' is:")
            print(Rsarr[:10])
        ## testing code ends
        if counter < N_recur + 1:
            URR = ncon([U, Ruprime, Rvprime],[[1,2,-3], [1,-1],[-2,2]])
            URRs = URR.multiply_diag(s.flip_dir(0), 2, direction = "left")
            Uinner, sinner = URRs.svd([0,1],[2])[:2]
            sinner = sinner / sinner.sum()
            Rprimeinner, counter = RoptCur(Uinner, sinner, epsilon,
                                    counter = counter+1, verbose = verbose)
            Rprime = ncon([Rprimeinner, Ruprime, Rvprime],[[1,2],[-1,1],[2,-2]])
        return Rprime, counter
    d,U = Gamma.eig([0,1],[2,3], hermitian = True)
    s = d.abs().sqrt()
    Rprime , counter= RoptCur(U, s, gilteps, verbose = verbose)
    u0, s0, v0 = Rprime.svd([0],[1], eps = spliteps,
                            chis = [i+1 for i in range(chitid)],
                            degeneracy_eps = 1e-20)
    return u0, s0, v0

def linearOpt(Gamma, GammaUpper, GammaDown, u, s, v, whichfix = "u"):
    """
    Given current u, s, v and the leg environment matrix Gamma,
    hold whichfix fixed and optimize the linearlized system.
    Return the updated u' s' and v'


    Parameters
    ----------
    Gamma : 4-leg tensor, leg environment matrix

        |1       2|
        ---Gamma---
          |Gamma|
        ---Gamma---
        |3       4|.
    u : isometric matrix
        second index labeling the eigenvectors.
    s : diagonal matrix
        singular value like thing.
    v : isometric matrix
        first index labeling the eigenvectors.
    whichfix : str, optional
        choose between u and v. The default is "u".

    Returns
    -------
    up : isometric matrix
        same shape as u.
    sp : diagonal matrix
        same shape as s.
    vp : isometric matrix
        same shape as v.

    """
    # Properly normalized GammaUpper and GammaDown
    GammaNorm = normalGamma(Gamma)[1]
    GammaUpper = GammaUpper / GammaNorm
    GammaDown = GammaDown / GammaNorm
    chiInner = s.to_ndarray().shape[0]
    if whichfix == "u":
        P = ncon([GammaUpper, u.conjugate()], [[1, 1, 2, -2], [2, -1]])
        B = ncon([GammaDown, u, u.conjugate()], [[1, -2, 2, -4], [1, -1],
                                             [2, -3]])
        Binv = pinv(B, eps_mach = 1e-16)
        R = ncon([Binv, P], [[1, 2, -1, -2], [1, 2]])
        Rtot = ncon([R, u], [[1, -2], [-1 ,1]])
    if whichfix == "v":
        P = ncon([GammaUpper, v.conjugate()], [[1, 1, -1 ,2], [-2, 2]])
        B = ncon([GammaDown, v, v.conjugate()], [[-1, 1, -3, 2], [-2, 1],
                                             [-4, 2]])
        Binv = pinv(B, eps_mach = 1e-16)
        R = ncon([Binv, P], [[1, 2,-1, -2], [1, 2]])
        Rtot = ncon([R, v], [[-1, 1], [1, -2]])
    # svd Rtot to get updated u, s, and v
    up, sp, vp = Rtot.svd([0], [1], chis = [i+1 for i in range(chiInner)],
                          eps = 1e-10)
    return up, sp, vp

def fetCut(A, B, Leg, chitid, max_iter = 20, verbose = False,
           forwhat = "trg", initscheme = "Gilt", giltdeg = 0.5,
           updateEnv = False):
    """
    Using FET to cut a leg in the plaquette defined by A, B tensors.
    This function will achieve the same goal as what cutLeg function for Gilt
    is designed to do

    Parameters
    ----------
    A : 4-leg tensor
        See cutLeg function for more details.
    B : 4-leg tensor
        See cutLeg function for more details.
    Leg : str, choose among ("I","II","III","IV")
        The leg to be truncated.
    chitid : int
        A smaller bound dimension we wish to squeeze to.
    max_iter : int, optional
        Maximal iteration of FET step. The default is 20.
    verbose : boolean, optional
        Whether to output information. The default is False.
    forwhat : str, optional, choose among ("trg", "hotrg")
        Define the plaquette, see cutLeg function for more details.
        The default is "trg".
    initscheme : str, optional
        Scheme for initialize the unknown matrices in FET.
        Choose among ("Gilt", "Morita", "org", "skeleton")
        The default is "Gilt".
    giltdeg : float, optional, between 0 and 1
        How much do we turn on gilt. The default is 0.5.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    u0 : matrix
         Isometry.
    s0 : matrix
        Diagonal.
    v0 : matrix
         Isometry.

    """
    if verbose:
        print("Perform FET for {:s} plaquette on leg {:s}.".format(forwhat,
                                                                 Leg))
        print("Bound dimension after squeezing is aimed at {:d}".format(chitid))
    # store the original shape of tensor A
    org_shape = str(type(A).flatten_shape(A.shape))
    if verbose:
        print("Original shape of tensor A is {}.".format(org_shape))
    # Define lattice, see docstring of envLeg for our convention
    if forwhat == 'trg':
        Alist = [A, B, A, B]
    elif forwhat == 'hotrg':
        Alist = [A, A, B, B]
    else:
        raise ValueError("forwhat should be chosen between trg or hotrg")
    # Calculate leg environment Gamma.
    # Note: Strickly speaking, we should insert u,s,v matrices into
    # the leg environment for what we are about to do below.
    # However, here we use a naive implementation for simplicity
    s0_env, Gamma = envLeg(Alist, Leg, return_envMat = True)[1:]
    # Normalized the leg environment
    Gamma, normSquare = normalGamma(Gamma)
    # Initialize u, s, v
    if initscheme == "Gilt":
        if verbose:
            print("Intialize u, s, v matrices using Gilt.")
        # estimate gilt_eps
        s0_env = s0_env.to_ndarray()
        s0_env = -np.sort(-s0_env)
        s0_env = s0_env / s0_env[0]
        N_s0 = len(s0_env)
        # automatically choose gilt eps
        gilteps = np.abs(s0_env[int((1 - giltdeg) * N_s0)])
        gilteps = np.min([gilteps, 1e-4])
        # perform gilt 5 iterations
        u0, s0, v0 = initGilt(Gamma, chitid, gilteps = gilteps)
        ugilt = u0.copy()
        sgilt = s0.copy()
        vgilt = v0.copy()
        chiGilt = s0.to_ndarray().shape[0]
        if chitid > chiGilt:
            chitid = chiGilt
            if verbose:
                print("The squeezed bound dimension suggested by Gilt",
                      "is {:d}, smaller than specified value.".format(chiGilt),
                      "We choose this one.")
        if verbose:
            print("The actualy squeesed bound dimension we use is {}.".format(chitid))
            s0arr = s0.to_ndarray()
            s0arr = -np.sort(-s0arr)
            print("The initial diagonal matrix s is ")
            with np.printoptions(precision = 3, suppress = None):
                print(s0arr)
    elif initscheme == "Morita":
        if verbose:
            print("Intialize u, s, v matrices using Morita-san's suggestion.")
        u0, s0, v0 = initusv(Gamma, chitid, inituVer = 'Morita')
        chitid = s0.to_ndarray().shape[0]
        if verbose:
            print("The actualy squeesed bound dimension we use is {}.".format(chitid))
            s0arr = s0.to_ndarray()
            s0arr = -np.sort(-s0arr)
            print("The initial diagonal matrix s is ")
            with np.printoptions(precision = 3, suppress = None):
                print(s0arr)
    # print the initial fidelity
    if updateEnv:
        ## This is really a BAD idea!!!!
        Gamma, GammaUpper, GammaDown, err0 = envLegUpdate(Alist, u0, s0, v0, Leg)
    else:
        # This is good
        err0 = fidelity(Gamma, u0, s0, v0)[1]
    if verbose:
        print("The starting error using {:s} initialization".format(initscheme),
        "is {:.2e}".format(err0))
    # iteratively optimize u, s, v
    uvcir = itt.cycle(["u","v"])
    for k in range(max_iter):
        whichfix = next(uvcir)
        if verbose:
            print("Hold {:s} fixed and apply linearlized optimazation.".format(whichfix))
        if updateEnv:
            # BAD idea!
            u0, s0, v0 = linearOpt(Gamma, GammaUpper, GammaDown, u0, s0, v0, whichfix)
        else:
            # Good
            u0, s0, v0 = linearOpt(Gamma, Gamma, Gamma, u0, s0, v0, whichfix)
        if verbose:
            s0arr = s0.to_ndarray()
            s0arr = -np.sort(-s0arr)
            chinew = s0arr.shape[0]
            print("The squeezed bound dimension at {:d}-th iteration is {:d}".format(k+1, chinew))
            print("The updated diagonal matrix s is ")
            with np.printoptions(precision = 3, suppress = None):
                print(s0arr)

        # update leg environment and calculate new error
        if updateEnv:
            # BAD idea
            GammaUpper, GammaDown, errcur = envLegUpdate(Alist, u0, s0, v0, Leg)[1:]
        else:
            # Good
            errcur = fidelity(Gamma, u0, s0, v0)[1]
        if verbose:
            print("The error at {:d}-th iteration is {:.2e}.".format(k+1, errcur))
        # if the change of error is too small, exit the loop
        if np.abs(errcur) < 1e-10 or np.abs(err0) < 1e-10:
            if verbose:
                print("Error smaller that 1e-10, exit the loop! ")
            break
        if k == 0:
            if np.abs(errcur) > np.abs(err0):
                u0 = ugilt
                s0 = sgilt
                v0 = vgilt
                if verbose:
                    print("Gilt initiazation is already very good, exit the loop! ")
                break

        if np.abs(err0 - errcur) / np.abs(err0) < 1e-2:
            if verbose:
                print("The change of updated error is too small, exit the loop! ")
            break
        # update the err0
        err0 = errcur
    err, phiphi, phipsi, psiphi = fidelity(Gamma, u0, s0, v0, return_phiphi = True)[1:]
    # fix the norm explicitly
    scaleFix = (phipsi + psiphi) / (2 * phiphi)
    s0 = s0 * scaleFix
    # update the error
    err, phiphi, phipsi, psiphi = fidelity(Gamma, u0, s0, v0, return_phiphi = True)[1:]
    loop_red_err = 1 - phipsi - psiphi + phiphi
    return u0, s0, v0, err, loop_red_err, k + 1

# end of for FET.
###############################
#
#
#
###############################
# Different applications of Gilts (or FET) on a square latiice

def applyRp(A,B, Rup, Rvp, Leg = "I", forwhat = 'trg', RABs = [1]*6):
    """
    Given a plaquette [A, B, A, B] that looks like
        | (I) |
     ---A-----B---
    (IV)|     |(II)
        |     |
     ---B-----A---
        |(III)|
    and also the leg to be cut. Apply Rup and Rvp on A and B so that
    the leg inside this loop is properly truncated. See Fig.4 of
    Markus's paper for more details.
    For example, if Leg = "I",
    Amod[i,j,k,l] = sum_n A[i,j,n,l] * Rup[n,k]
    Bmod[i,j,k,l] = sum_n B[n,j,k,l] * Rvp[n,i]
    Parameters
    ----------
    A : Four leg tensor A[i,j,k,l]
         j|
       i--A--k.
         l|
    B : Four leg tensor B[i,j,k,l]
        same as A.
    Rup : matrix
        left piece of R'.
    Rvp : matrix
        right piece of R'.
    Leg : str, choose among ("I","II","III","IV")
        the leg to be truncated.

    Returns
    -------
    Amod : Four leg tensor
        with one leg properly truncated.
    Bmod : Four leg tensor
        with one leg properly truncated.

    """
    if forwhat == 'trg':
        Alist = [A, B, A, B]
        [Arot1, Arot2, _, _], _ = rotateLattice(Alist, Leg)
        Arot1mod = ncon([Arot1, Rup],[[-1,-2,1,-4],[1,-3]])
        Arot2mod = ncon([Arot2, Rvp],[[1,-2,-3,-4],[-1,1]])
        Amodlist = [Arot1mod, Arot2mod, Arot1mod, Arot2mod]
        revLeg = {"I":"I", "II":"IV", "III":"III", "IV":"II"}
        [Amod, Bmod, _, _], _ = rotateLattice(Amodlist, revLeg[Leg])
        return Amod, Bmod
    elif forwhat == 'hotrg':
        # Alist = [A, A, B, B]
        # [Arot1, Arot2, _, _], _ = rotateLattice(Alist, Leg)
        # Arot1mod = ncon([Arot1, Rup],[[-1,-2,1,-4],[1,-3]])
        # Arot2mod = ncon([Arot2, Rvp],[[1,-2,-3,-4],[-1,1]])
        # Amodlist = [Arot1mod, Arot2mod, Arot2mod, Arot1mod]
        # revLeg = {"II":"IV", "IV":"II"}
        # Amod = rotateLattice(Amodlist, revLeg[Leg])[0][0]
        # Bmod = rotateLattice(Amodlist, revLeg[Leg])[0][2]
        RA, RB,  RAl, RAr, RBl, RBr = RABs
        if Leg == "II":
            Amod = ncon([A, Rup], [[-1, -2, -3, 1],[1, -4]])
            Bmod = ncon([B, Rvp], [[-1, 1, -3, -4],[-2, 1]])
            # update RA and RB matrices
            if type(RA).__module__.split(".")[0] != 'abeliantensors':
                RA = 1.0 * Rup
            else:
                RA = ncon([RA, Rup], [[-1,1],[1,-2]])
            if type(RB).__module__.split(".")[0] != 'abeliantensors':
                RB = 1.0 * Rvp.transpose([1,0])
            else:
                RB = ncon([RB, Rvp], [[-1,1],[-2,1]])
        elif Leg == "IV":
            Amod = ncon([A, Rvp], [[-1, -2, -3, 1],[-4, 1]])
            Bmod = ncon([B, Rup], [[-1, 1, -3, -4],[1, -2]])
            # update RA and RB matrices
            if type(RA).__module__.split(".")[0] != 'abeliantensors':
                RA = 1.0 * Rvp.transpose([1,0])
            else:
                RA = ncon([RA, Rvp], [[-1,1],[-2,1]])
            if type(RB).__module__.split(".")[0] != 'abeliantensors':
                RB = 1.0 * Rup
            else:
                RB = ncon([RB, Rup], [[-1,1],[1,-2]])
        elif Leg == "I":
            Amod = ncon([A, Rup, Rvp], [[1, -2, 2, -4], [2, -3], [-1, 1]])
            Bmod = B * 1.0
            # update RAl, RAr matrices
            if type(RAl).__module__.split(".")[0] != 'abeliantensors':
                RAl = 1.0 * Rvp.transpose([1,0])
            else:
                RAl = ncon([RAl, Rvp], [[-1,1],[-2,1]])
            if type(RAr).__module__.split(".")[0] != 'abeliantensors':
                RAr = 1.0 * Rup
            else:
                RAr = ncon([RAr, Rup], [[-1,1],[1,-2]])
        elif Leg == "III":
            Amod = A * 1.0
            Bmod = ncon([B, Rup, Rvp], [[1,-2, 2, -4], [1, -1], [-3, 2]])
            # update RBl, RBr matrices
            if type(RBl).__module__.split(".")[0] != 'abeliantensors':
                RBl = 1.0 * Rup
            else:
                RBl = ncon([RBl, Rup], [[-1,1],[1,-2]])
            if type(RBr).__module__.split(".")[0] != 'abeliantensors':
                RBr = 1.0 * Rvp.transpose([1,0])
            else:
                RBr = ncon([RBr, Rvp], [[-1,1],[-2,1]])
        else:
            raise ValueError("Leg can only be chosen from [II, IV, I, III]" +
                             "in this gilt-hotrg-imp version")
        RABs = [RA, RB, RAl, RAr, RBl, RBr]
        return Amod, Bmod, RABs
    else:
        raise ValueError("forwhat should be chosen between trg or hotrg")

def gilt_hotrgplaq(A,B,epsilon = 1e-7, convergence_eps = 5e-3,
                   verbose = True, direction = "v", legcut = 2,
                   loop_red_scheme = "Gilt", argsFET = {},
                   RoptVerbose = False, suggestChiAB = [False, 1, 1]):
    """
    Apply gilts on two of four leg. The plaquette looks like
        | (I) |
     ---A-----A---
    (IV)|     |(II)
        |     |
     ---B-----B---
        |(III)|
    and we truncate the leg according to order II -> IV
    Parameters
    ----------
    A : Four leg tensor A[i,j,k,l]
         j|
       i--A--k.
         l|
    B : Four leg tensor
        same as A.
   epsilon : float, optional
        a small number below which we think the singular value is zero.
        The default is 1e-7.
   convergence_eps : float, optional
        difference from identity matrix smaller than which
        we think we get a good approximation. The default is 1e-2.
    verbose : boolean, optional
        print information. The default is True.

    Returns
    -------
    A : Four leg tensor A[i,j,k,l]
         j|
       i--A--k.
         l|
    B : Four leg tensor
        same as A.

    """
    # make sure tensor A, B are abeliantensors
    if type(A).__module__.split(".")[0] != 'abeliantensors':
        A = Tensor.from_ndarray(A)
    if type(B).__module__.split(".")[0] != 'abeliantensors':
        B = Tensor.from_ndarray(B)
    # To consider horizontal direction, simply rotate the whole picture by
    # -90 degress
    if direction == "h":
        A = A.transpose([3,0,1,2])
        B = B.transpose([3,0,1,2])
    # store the original shape before gilts
    org_shape = str(type(A).flatten_shape(A.shape))
    # order of leg name for cutting
    if loop_red_scheme == "Gilt":
        # for Gilt
        if legcut == 2:
            # ord_leg = ['II','IV']
            ord_leg = ['I','III']
        elif legcut == 4:
            ord_leg = ['II','IV','I','III']
    elif loop_red_scheme == "FET":
        # for FET
        ord_leg = ['I','III']
    else:
        raise ValueError("loop_red_scheme can be only chosen between Gilt and FET!")
    # dictionary to keep track whether all legs are done.
    done_legs = {i: False for i in ord_leg}
    # total gilt error, which is simply the sum of all truncation errors
    loop_red_err = 0
    # store matrices used for leg truncations
    RABs = [1] * 6


    if verbose:
        print("Start Applying {:s}...".format(loop_red_scheme))
    if loop_red_scheme == "Gilt":
        for leg in itt.cycle(ord_leg):
            if leg == "I":
                suggestChi = [suggestChiAB[0], suggestChiAB[1]]
            elif leg == "III":
                suggestChi = [suggestChiAB[0], suggestChiAB[2]]
            Ruprime, Rvprime, done, err, nRecur = cutLeg(A, B, leg,
                                               epsilon, convergence_eps, verbose = verbose,
                                               forwhat = 'hotrg',
                                               suggestChi = suggestChi)
            A,B, RABs = applyRp(A,B,Ruprime, Rvprime, leg, forwhat = 'hotrg',
                          RABs = RABs)
            # ignore whether it is done or not
            done_legs[leg] = True
            loop_red_err += err
            if verbose:
                cur_shapeA = str(type(A).flatten_shape(A.shape))
                cur_shapeB = str(type(B).flatten_shape(B.shape))
                print("Applying Gilt on leg {:s},".format(leg),
                      "Error accumulated = {:.3e}.\n".format(loop_red_err),
                      "shape of A = {}({})\n".format(cur_shapeA, org_shape),
                      "shape of B = {}({})\n".format(cur_shapeB, org_shape),
                      "number of recursions = {:d}.".format(nRecur),
                      "Truncation done? {}".format(done))
            if all(done_legs.values()):
                break
    elif loop_red_scheme == "FET":
        # read FET parameters
        chitid = argsFET['chitid']
        maxiter = argsFET['maxiter']
        initscheme = argsFET['initscheme']
        giltdeg = argsFET['giltdeg']
        for leg in ord_leg:
            up, sp, vp, err, loop_red_err_cur, N_iter = fetCut(A, B, leg, chitid, maxiter, forwhat = "hotrg",
                                initscheme = initscheme, giltdeg = giltdeg)
            # absorb sp into up and vp to construct Rup and Rvp
            # absolute values for possible numerical very small negative numbers
            spSqrt = sp.abs().sqrt()
            Ruprime = up.multiply_diag(spSqrt, axis = 1, direction = 'r')
            Rvprime = vp.multiply_diag(spSqrt, axis = 0, direction = 'l')
            # Absorb Rup and Rvp into on our tensors A and B
            A,B, RABs = applyRp(A,B,Ruprime, Rvprime, leg, forwhat = 'hotrg',
                          RABs = RABs)
            loop_red_err += loop_red_err_cur
            if verbose:
                cur_shape = str(type(A).flatten_shape(A.shape))
                print("Applying FET on leg {:s},".format(leg),
                      "number of iterations = {:d}".format(N_iter),
                      "Current approximation error is {:.3e} and {:.3e}.\n".format(
                          err, loop_red_err_cur),
                      "Error accumulated = {:.3e}.\n".format(loop_red_err),
                      "shape = {}({})".format(cur_shape, org_shape))
    # rotation indices back for horizontal direction
    if direction == "h":
        A = A.transpose([1,2,3,0])
        B = B.transpose([1,2,3,0])
    if verbose:
        print("We only do one cycle here!")
        print("{:s}, done!".format(loop_red_scheme))
    return A, B, RABs



if __name__ == "__main__":
    ## Test clients for various functions here
    # from datetime import datetime
    import argparse

    parser = argparse.ArgumentParser(
        "Test gilts functions")
    parser.add_argument("func", type = str,
                        help = "the function to be tested",
                        choices = ["envLeg", "Ropt", "gilt_halfplaq",
                                   "gilt_fullplaq","gilt_trg",
                                   "gilt_iso_fullplaq",
                                   "gilt_hotrgplaq"])
    parser.add_argument("--chi", dest = "chi", type = int,
                    help = "bound dimension (default: 10)",
                    default = 10)
    parser.add_argument("--relT", dest = "relT", type = float,
                    help = "relative temperature to Tc",
                    default = 1.0)
    parser.add_argument("--legcut", dest = "legcut", type = int,
                    help = "number of leg to cut in gilt_hotrgplaq",
                    choices = [2, 4],
                    default = 2)
    parser.add_argument("--gilteps", dest = "gilteps", type = float,
                    help = "Gilt eps",
                    default = 1e-5)
    parser.add_argument("--corechi", dest = "corechi", type = int,
                    help = "bound dimension of core tensor mixed with CDL (default: 1)",
                    default = 1)
    parser.add_argument("--sqrtchi", dest = "sqrtchi", type = int,
                    help = "bound dimension of matrix making up of CDL (default: 2)",
                    default = 2)
    parser.add_argument("--loopred", dest = "loopred", type = str,
                    help = "which loop reduction scheme to use",
                    choices = ["Gilt", "FET"], default = "Gilt")
    args = parser.parse_args()
    func = args.func
    chi = args.chi
    relT = args.relT
    legcut = args.legcut
    gilteps = args.gilteps
    corechi = args.corechi
    sqrtchi = args.sqrtchi
    loopred = args.loopred

    argsFET = {'chitid':sqrtchi, 'maxiter':20, 'initscheme':'Gilt',
               'giltdeg':0.5}

    # np.random.seed(1)
    cdlT, numb = cdl(sqrtchi,corechi,isSym = True)
    if func == "envLeg":
        import matplotlib.pyplot as plt
        Trand = Tensor.random([4,4,4,4])
        _,scdl = envLeg([cdlT]*4)
        _,srand = envLeg([Trand]*4)
        plt.figure()
        plt.plot(srand/srand.norm(),'bx--', label= "random tensor")
        plt.plot(scdl.to_ndarray()/scdl.norm(),'kp--',label="CDL tensor")
        plt.yscale("log")
        plt.title("$chi = 4$")
        plt.legend()
        plt.savefig("../envs.png",dpi = 300, bbox_inches = 'tight')
    elif func == "Ropt":
        U,s = envLeg([cdlT]*4)
        epsilon = 1e-7
        spliteps = 1e-3
        Rprime, counter = Ropt(U,s, epsilon, verbose = True)
        print("# of recursion is {:d}.".format(counter))
        Ruprime, Rs, Rvprime = Rprime.split([0],[1],return_sings = True,
                                            eps = spliteps)
        print("singular value spectrum of final $R^n$ is")
        Rs = Rs.to_ndarray()
        print(Rs)
        print("normalize to 1")
        print(Rs/ Rs[0])
        print("Difference from 1 is ")
        print(Rs/ Rs[0] - 1)
        err = gilt_err(U, s, Ruprime, Rvprime)
        print("Gilts truncation error is {:.2e}".format(err))
    elif func == "gilt_hotrgplaq":
        print("Test client of function gilt_iso_fullplaq")
        print("1. Test vertical direction...")
        A, B, RABs = gilt_hotrgplaq(cdlT, cdlT, legcut = legcut,
                                    loop_red_scheme = loopred,
                                    argsFET = argsFET)
        print("CDL tensor shape after truncation is")
        print("Shape of A = ")
        print(A.to_ndarray().shape)
        print("Shape of B = ")
        print(B.to_ndarray().shape)
        print("Check whether we successfully keep track of RABs matrices...")
        print("Calculating truncated A and B using RABs matrices...")
        RA, RB, RAl, RAr, RBl, RBr = RABs
        if loopred == "Gilt":
            if legcut == 4:
                Acheck = ncon([cdlT, RA, RAl, RAr],[[2, -2, 3, 1],[1,-4], [2, -1], [3, -3]])
                Bcheck = ncon([cdlT, RB, RBl, RBr],[[2, 1, 3, -4],[1,-2], [2, -1], [3, -3]])
            elif legcut == 2:
                # Acheck = ncon([cdlT, RA],[[-1, -2, -3, 1],[1,-4]])
                # Bcheck = ncon([cdlT, RB],[[-1, 1, -3, -4],[1,-2]])
                Acheck = ncon([cdlT, RAl, RAr],[[2, -2, 3, -4], [2, -1], [3, -3]])
                Bcheck = ncon([cdlT, RBl, RBr],[[2, -2, 3, -4], [2, -1], [3, -3]])
        elif loopred == "FET":
            Acheck = ncon([cdlT, RAl, RAr],[[2, -2, 3, -4], [2, -1], [3, -3]])
            Bcheck = ncon([cdlT, RBl, RBr],[[2, -2, 3, -4], [2, -1], [3, -3]])
        print("Are the result consistent? ")
        print("Check A tensor...")
        print(A.allclose(Acheck))
        print("Check B tensor...")
        print(B.allclose(Bcheck))
        print("Vertical direction test finished! ")
        print("2. Test horizontal direction...")
        A, B, RABs = gilt_hotrgplaq(cdlT, cdlT, direction = "h", legcut = legcut,
                                    loop_red_scheme = loopred,
                                    argsFET = argsFET)
        print("CDL tensor shape after truncation is")
        print("Shape of A = ")
        print(A.to_ndarray().shape)
        print("Shape of B = ")
        print(B.to_ndarray().shape)
        print("Check whether we successfully keep track of RABs matrices...")
        print("Calculating truncated A and B using RABs matrices...")
        RA, RB, RAl, RAr, RBl, RBr = RABs
        if loopred == "Gilt":
            if legcut == 4:
                Acheck = ncon([cdlT, RA, RAl, RAr],[[-1, 3, 1, 2],[1,-3], [2, -4], [3, -2]])
                Bcheck = ncon([cdlT, RB, RBl, RBr],[[1, 3, -3, 2],[1,-1], [2, -4], [3, -2]])
            elif legcut == 2:
                # Acheck = ncon([cdlT, RA],[[-1, -2, 1, -4],[1,-3]])
                # Bcheck = ncon([cdlT, RB],[[1, -2, -3, -4],[1,-1]])
                Acheck = ncon([cdlT, RAl, RAr],[[-1, 3, -3, 2], [2, -4], [3, -2]])
                Bcheck = ncon([cdlT, RBl, RBr],[[-1, 3, -3, 2], [2, -4], [3, -2]])
        elif loopred == "FET":
            Acheck = ncon([cdlT, RAl, RAr],[[-1, 3, -3, 2], [2, -4], [3, -2]])
            Bcheck = ncon([cdlT, RBl, RBr],[[-1, 3, -3, 2], [2, -4], [3, -2]])
        print("Are the result consistent? ")
        print("Check A tensor...")
        print(A.allclose(Acheck))
        print("Check B tensor...")
        print(B.allclose(Bcheck))
        print("Horizontal direction test finished! ")
    else:
        raise ValueError("{:s} is not supported".format(func) +
                         " in this simplified gilt file")
