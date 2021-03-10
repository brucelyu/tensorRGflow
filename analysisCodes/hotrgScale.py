#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : hotrgScale.py
# Author            : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# Date              : 11.03.2021
# Last Modified Date: 11.03.2021
# Last Modified By  : Xinliang(Bruce) Lyu <lyu@issp.u-tokyo.ac.jp>
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:56:25 2020
Calculate the scaling dimension by differentiating on RG step.
The TRG scheme is HOTRG
@author: brucelyu
"""

import numpy as np
import argparse
import pickle as pkl
import time
from HOTRG import diffGiltHOTRG, scDimWen

parser = argparse.ArgumentParser(
    "Test calculation of scaling dimensions of 2d-Ising using differentiation" +
    "techniques applied on HOTRG.")
parser.add_argument("--chi", dest = "chi", type = int,
                help = "horizontal and vertical bound dimension (default: 10)",
                default = 10)
parser.add_argument("--gilteps", dest = "gilteps", type = float,
                    help = "a number smaller than which we think the" +
                    "singluar values for the environment spectrum is zero" +
                    "(default: 1e-7)",
                    default = 1e-7)
parser.add_argument("--NscaleD", dest = "NscaleD", type = int,
                    help = "number of scaling dimensions to be calculated (default: 20)",
                    default = 20)

parser.add_argument("--iRGlow", dest = "iRGlow", type = int,
                    help = "lower limit of the calculated RG step range (default: 1)",
                    default = 1)
parser.add_argument("--iRGhi", dest = "iRGhi", type = int,
                    help = "higher limit of the calculated RG step range (default: 10)",
                    default = 10)
parser.add_argument("--Ngilt", dest = "Ngilt", type = int,
                    help = "How many times do we perform Gilt in oneHOTRG",
                    choices = [1,2], default = 1)
parser.add_argument("--legcut", dest = "legcut", type = int,
                    help = "number of leg to cut in gilt_hotrgplaq",
                    choices = [2,4], default = 4)



args = parser.parse_args()

chi = args.chi
NscaleD = args.NscaleD
gilteps = args.gilteps
lowRG = args.iRGlow
hiRG = args.iRGhi
Ngilt = args.Ngilt
legcut = args.legcut


## Read data of flow of tensor in RG
figdir = "gilt_hotrg{:d}{:d}_flow".format(Ngilt, legcut)
chieps = "eps{:.0e}_chi{:02d}".format(gilteps, chi)


Tsdata = "./data/" + figdir + "/" + chieps + "/otherTs.pkl"
with open(Tsdata, "rb") as f:
    Anorm, isomlist, RABslist, RABshlist = pkl.load(f)


## file to read or save scaling dimensions
scDFile = "../out/" + figdir +  "/" + chieps + "/scDim.pkl"

# Calculating scaling dmensions by analyzing the response matrix
print("Response analysis in tensor space using Gilt-HOTRG start!")
klist = []
scDlist = []
for k in range(lowRG, hiRG):
    # save k
    klist.append(k)
    print("Performing {}-th to {}-th RG...".format(k,k+1))

    print("Calculate the response matrix using jax.linearize routine, " +
          "and calculate the first {:d} scaling dimensions from its eigenvalues.".format(NscaleD))

    startT = time.time()
    # read out relevent tensors and matrices
    AcurData = "./data/" + figdir + "/" + chieps + "/As/A{:02d}.pkl".format(k)
    with open(AcurData, "rb") as f:
        Acur = pkl.load(f)
    Anormcur = Anorm[k]
    isom = isomlist[k]
    RABs = RABslist[k]
    RABsh = RABshlist[k]
    scDims = diffGiltHOTRG(Acur, Anormcur, isom, RABs, RABsh, scaleN = NscaleD)
    endT = time.time()
    diffT = endT - startT
    # if type(scDims) == int:
    #     scDims = np.zeros(NscaleD)

    # save scaling dimensions
    scDlist.append(scDims)
    print("finished! Time elapsed = {:.2f}".format(diffT))
    print("The scaling dimensions are:")
    with np.printoptions(precision = 5, suppress = True):
        print(scDims)
    print("------------------------------------")
    print("Calculating the scaling dimensions a la Gu and Wen (2009)")
    Ainv = Acur * (Anormcur) ** (-1/3)
    scDimsW, ccharge = scDimWen(Ainv, scaleN=NscaleD)
    print("The scaling dimensions are:")
    with np.printoptions(precision = 5, suppress = True):
        print(scDimsW[:NscaleD])
    print("------------------------------------")
    print("------------------------------------")

with open(scDFile,"wb") as f:
    pkl.dump([klist,scDlist],f)



