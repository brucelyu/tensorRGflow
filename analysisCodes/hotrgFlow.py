#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Sep  4 15:06:00 2020
Anlysis of the flow of 1) norm of the tensor, 2) singular value spectrum of
the tensor, and 3) norm of the difference of the two tensors in adjacent RG step

@author: brucelyu
"""

from HOTRG import get_spectrum_A, normFlowHOTRG
import numpy as np
import argparse
import os
import pickle as pkl
from datetime import datetime

# argument parser
parser = argparse.ArgumentParser(
    "Anlysis of the flow of 1) norm of the tensor, " + 
    "2) singular value spectrum of the tensor, and " + 
    "3) norm of the difference of the two tensors in adjacent RG step")
parser.add_argument("--chi", dest = "chi", type = int,
                    help = "bound dimension (default: 10)",
                    default = 10)
parser.add_argument("--maxiter", dest = "maxiter", type = int,
                   help = "maximal HOTRG iteration (default: 50)",
                   default = 50)
parser.add_argument("--gilteps", dest = "gilteps", type = float,
                        help = "a number smaller than which we think the" +
                        "singluar values for the environment spectrum is zero" +
                        "(default: 1e-7)",
                        default = 1e-7)
parser.add_argument("--verbose", help = "whether to print information",
                        action = "store_true")
parser.add_argument("--scheme", dest = "scheme", type = str, 
                    help = "RG scheme to use",
                    choices = ["hotrg", "Gilt-HOTRG"],
                    default = "Gilt-HOTRG")
parser.add_argument("--cgeps", dest = "cgeps", type = float,
                        help = "a number smaller than which we think the" +
                        "singluar values for the environment in RG spectrum is zero" +
                        "(default: 1e-10)",
                        default = 1e-10)
parser.add_argument("--Ngilt", dest = "Ngilt", type = int,
                    help = "How many times do we perform Gilt in oneHOTRG",
                    choices = [1,2], default = 1)
parser.add_argument("--legcut", dest = "legcut", type = int,
                    help = "number of leg to cut in gilt_hotrgplaq",
                    choices = [2,4], default = 4)
parser.add_argument("--stbk", dest = "stbk", type = int,
                    help = "A int after which we will try to stabilize the gilt process",
                    default = 1000)


# read from argument parser
args = parser.parse_args()
chi = args.chi
iter_max = args.maxiter
gilteps = args.gilteps
verbose = args.verbose
allchi = [chi,chi]
scheme = args.scheme
cgeps = args.cgeps
Ngilt = args.Ngilt
legcut = args.legcut
stablek = args.stbk
# Print out the time when the script is executed
now = datetime.now()
current_time = now.strftime("%Y-%m-%d. %H:%M:%S")
print("Running Time =", current_time)

# input and output file name
if scheme == "hotrg":
    figdir = "hotrgflow"
elif scheme == "Gilt-HOTRG":
    figdir = "gilt_hotrg{:d}{:d}_flow".format(Ngilt, legcut)


chieps = "eps{:.0e}_chi{:02d}".format(gilteps, chi)
savedirectory = "../out/" + figdir +  "/" + chieps
# read Tc if exists
relTc = 1.0
Tcfile = savedirectory + "/Tc.pkl"
if not os.path.exists(Tcfile):
    relTc = 1.0
    print("No estimated Tc exists, set Tc = 1.")
else:
    with open(Tcfile,"rb") as f:
        Tlow, Thi = pkl.load(f)
    relTc = 0.5 * (Tlow + Thi)
    Tcerr = abs(Thi - Tlow) / (Tlow + Thi)
    outacc = int("{:e}".format(Tcerr)[-2:])
    print("Read the estimated Tc = {Tcval:.{acc}f}".format(Tcval = relTc,
                                                           acc = outacc))
    print("Related error of the estimate is {:.1e}".format(Tcerr))


print("Step 2: Start to generate data of the flow of singular spectrum of A...")
# Generate data of 2) singular value spectrum of the tensor
singValFile = savedirectory + "/flowAtTc_fixSign.pkl"
if scheme == "hotrg":
    pass
    # appg, A, Anorm, Ruvslist, isometrylist, dMslist, Adifflist = mainHOTRG(relTc, 
    #                         allchi, iter_max, 
    #                         isGilt=isGilt, isSym = True, calcg = False,
    #                         gilt_eps = gilteps, return_iso_Ruvs = True,
    #                         isfixGauge = True, isAdiff = True, 
    #                         isDisp = verbose)
elif scheme == "Gilt-HOTRG":
    savedir = "./data/" + figdir +  "/" + chieps
    # create the directory if not exists
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    Anorm, slist, Adifflist = normFlowHOTRG(relTc,allchi, iter_max, isDisp = False, 
                              isGilt = True, isSym = True, isfixGauge = True,
                              gilt_eps = gilteps, cg_eps = cgeps,
                              return_sing = True,
                              N_gilt = Ngilt, legcut = legcut,
                              stableStep = stablek, saveData = [True, savedir])



# if scheme != "Gilt-HOTRG":
#     slist = []
#     for myA in A:
#         if scheme == "hotrg":
#             slist.append(get_spectrum_A(myA))
#         elif scheme == "trg":
#             slist.append(get_spectrum_A(myA, leftgrp = [0,1], rightgrp = [2,3]))




Nsing = max([len(inner) for inner in slist])
for i in range(len(slist)):
    temp = [slist[i][j] if j < len(slist[i]) else 0 for j in range(Nsing)]
    slist[i] = temp
sarr = np.array(slist)
if scheme == "hotrg" or scheme == "trg":
    with open(singValFile, "wb") as f:
        pkl.dump([sarr, Adifflist], f)
elif scheme =="Gilt-HOTRG":
    with open(singValFile, "wb") as f:
        pkl.dump([sarr, Adifflist], f)
print("Step 2 finished! ")


# print("Step 3: Save the data of renormalized tensors and their norms...")
# savedata = "./data/" + figdir +  "/" + chieps + "/As_fixSign.pkl"
# # create the directory if not exists
# if not os.path.exists("./data/" + figdir +  "/" + chieps):
#     os.makedirs("./data/" + figdir +  "/" + chieps)
    
# if scheme == "hotrg":
#     with open(savedata, "wb") as f:
#         pkl.dump([A, Anorm, Ruvslist, isometrylist, dMslist], f)
# elif scheme == "trg":
#     with open(savedata, "wb") as f:
#         pkl.dump([A, Anorm], f)

# print("Step 3 finished! ")

        
