#!/usr/bin/env python3
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
from HOTRG import diffRGver2, scDimWen

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

parser.add_argument("--fixGauge", dest = "fixG",
                help = "Fix the gauge using global method",
                action="store_true")
parser.add_argument("--iRGlow", dest = "iRGlow", type = int,
                    help = "lower limit of the calculated RG step range (default: 1)",
                    default = 1)
parser.add_argument("--iRGhi", dest = "iRGhi", type = int,
                    help = "higher limit of the calculated RG step range (default: 10)",
                    default = 10)
parser.add_argument("--scheme", dest = "scheme", type = str, 
                    help = "RG scheme to use",
                    choices = ["trg", "hotrg", "Gilt-HOTRG-imp"],
                    default = "hotrg")
parser.add_argument("--Ngilt", dest = "Ngilt", type = int,
                    help = "How many times do we perform Gilt in oneHOTRG",
                    choices = [1,2], default = 1)
parser.add_argument("--legcut", dest = "legcut", type = int,
                    help = "number of leg to cut in gilt_hotrgplaq",
                    choices = [2,4], default = 4)
parser.add_argument("--isomcorr", dest = "isomcorr",
                help = "whether to include first order correction of isometry",
                action="store_true")


args = parser.parse_args()

chi = args.chi
isfixG = args.fixG
NscaleD = args.NscaleD
gilteps = args.gilteps
lowRG = args.iRGlow
hiRG = args.iRGhi

scheme = args.scheme
Ngilt = args.Ngilt
legcut = args.legcut
isomcorr = args.isomcorr
## file to read or save scaling dimensions
# if isfixG:
#     scDFile = "../out/jaxTRG/hotrgchi{:02d}_eps{:.0e}_fixG.pkl".format(args.chi, gilteps)
# else:
#     scDFile = "../out/jaxTRG/hotrgchi{:02d}_eps{:.0e}_fixSign.pkl".format(args.chi, gilteps)


## Read data of flow of tensor in RG
if scheme == "hotrg":
    figdir = "gilt_hotrg_flow"
elif scheme == "trg":
    figdir = "gilt_trg_flow"
elif scheme == "Gilt-HOTRG-imp":
    figdir = "gilt_hotrg_imp{:d}{:d}_flow".format(Ngilt, legcut)
chieps = "eps{:.0e}_chi{:02d}".format(gilteps, chi)

if scheme == "hotrg":
    Asdata = "./data/" + figdir +  "/" + chieps + "/As_fixSign.pkl"
    with open(Asdata,"rb") as f:
        A, Anorm,  Ruvlist, isometrylist, dMslist = pkl.load(f)
elif scheme == "Gilt-HOTRG-imp":
    Tsdata = "./data/" + figdir + "/" + chieps + "/otherTs.pkl"
    with open(Tsdata, "rb") as f:
        Anorm, isomlist, RABslist, RABshlist = pkl.load(f)
    

## file to read or save scaling dimensions
if not isomcorr:
    scDFile = "../out/" + figdir +  "/" + chieps + "/scDim.pkl"
else:
    scDFile = "../out/" + figdir +  "/" + chieps + "/scDim_isomCorr.pkl"

# Calculating scaling dmensions by analyzing the response matrix
print("jaxTRG start!")
klist = []
scDlist = []
for k in range(lowRG, hiRG):
    # save k
    klist.append(k)
    print("Performing {}-th to {}-th RG...".format(k,k+1))
    mystr = "out"
    fixGname = "sign"
    if isfixG:
        mystr = ""
        fixGname = "svd"
    print("Calculate using forward AD techniques,",
          "with" + mystr + " gauge fixing,",
          "with scheme {:s}".format(scheme))
    if isomcorr:
        print("We include the first order correction of isometry here")
    startT = time.time()
    if scheme == "hotrg" or scheme == "trg":
        scDims = diffRGnew(A,Anorm,isometrylist,Ruvlist, dMslist,
                           rgstep = k, gaugefix = fixGname, scaleN = NscaleD)
    elif scheme == "Gilt-HOTRG-imp":
        # read out relevent tensors and matrices
        AcurData = "./data/" + figdir + "/" + chieps + "/As/A{:02d}.pkl".format(k)
        with open(AcurData, "rb") as f:
            Acur = pkl.load(f)
        Anormcur = Anorm[k]
        isom = isomlist[k]
        RABs = RABslist[k]
        RABsh = RABshlist[k]
        scDims = diffRGver2(Acur, Anormcur, isom, RABs, RABsh, scaleN = NscaleD,
                            isom_corr = isomcorr)
    endT = time.time()
    diffT = endT - startT
    if type(scDims) == int:
        scDims = np.zeros(NscaleD)
    # save scaling dimensions
    scDlist.append(scDims)
    print("finished! Time elapsed = {:.2f}".format(diffT))
    print("The scaling dimensions are:")
    with np.printoptions(precision = 5, suppress = True):
        print(scDims)
    print("------------------------------------")
    print("Calculating a la Wen (2009)")
    if scheme == "hotrg" or scheme == "trg":
        Ainv = A[k] * (Anorm[k])**(-1/3)
    elif scheme == "Gilt-HOTRG-imp":
        Ainv = Acur * (Anormcur) ** (-1/3)
    scDimsW, ccharge = scDimWen(Ainv)
    print("The scaling dimensions are:")
    with np.printoptions(precision = 3, suppress = True):
        print(scDimsW)
    print("------------------------------------")
    print("------------------------------------")

with open(scDFile,"wb") as f:
    pkl.dump([klist,scDlist],f)



