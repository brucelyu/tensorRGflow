#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Sep  4 15:06:00 2020
Anlysis of the flow of 1) norm of the tensor, 2) singular value spectrum of
the tensor, and 3) norm of the difference of the two tensors in adjacent RG step

@author: brucelyu
"""

from HOTRG import normFlowHOTRG
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
parser.add_argument("--nosignfix", help = "whether to not fix sign",
                        action = "store_true")
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
fixSign = not args.nosignfix
# Print out the time when the script is executed
now = datetime.now()
current_time = now.strftime("%Y-%m-%d. %H:%M:%S")
print("Running Time =", current_time)

# input and output file name
if scheme == "hotrg":
    figdir = "hotrg"
    chieps = "chi{:02d}".format(chi)
elif scheme == "Gilt-HOTRG":
    if fixSign:
        figdir = "gilt_hotrg{:d}{:d}_flow".format(Ngilt, legcut)
    else:
        figdir = "gilt_hotrg{:d}{:d}_nosignfix".format(Ngilt, legcut)
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
    relTc = Thi * 1
    Tcerr = abs(Thi - Tlow) / (Tlow + Thi)
    outacc = int("{:e}".format(Tcerr)[-2:])
    print("Read the estimated Tc = {Tcval:.{acc}f}".format(Tcval = relTc,
                                                           acc = outacc))
    print("Related error of the estimate is {:.1e}".format(Tcerr))


print("Step 2: Start to generate data of the flow of the tensor A...")
# Generate data of 2) singular value spectrum of the tensor
singValFile = savedirectory + "/flowAtTc_fixSign.pkl"
if scheme == "hotrg":
    # generate flow of |A| at different temperature near Tc
    devTc = [3,6,8]
    datadic ={}
    for acc in devTc:
        Tdevhi = relTc + 10**(-acc)
        Tdevlow = relTc - 10**(-acc)
        AnormH = normFlowHOTRG(Tdevhi,[chi,chi], iter_max, isDisp = False, 
                         isGilt = False, isSym = False,
                         gilt_eps = gilteps, cg_eps = cgeps,
                         N_gilt = Ngilt, legcut = legcut,
                         stableStep = stablek)[0]
        AnormL = normFlowHOTRG(Tdevlow,[chi,chi], iter_max, isDisp = False, 
                         isGilt = False, isSym = False,
                         gilt_eps = gilteps, cg_eps = cgeps,
                         N_gilt = Ngilt, legcut = legcut,
                         stableStep = stablek)[0]
        datadic[acc] = [AnormL, AnormH]
        datadicFile = savedirectory + "/flowDiffAcc.pkl"
        with open(datadicFile,"wb") as f:
            pkl.dump(datadic, f)
elif scheme == "Gilt-HOTRG":
    # Generate the flow of |A| at the estimated Tc
    savedir = "./data/" + figdir +  "/" + chieps
    # create the directory if not exists
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print("At T = Tc")
    Anorm, slist, Adifflist = normFlowHOTRG(relTc,allchi, iter_max, isDisp = verbose, 
                              isGilt = True, isSym = True, isfixGauge = fixSign,
                              gilt_eps = gilteps, cg_eps = cgeps,
                              return_sing = True,
                              N_gilt = Ngilt, legcut = legcut,
                              stableStep = stablek, saveData = [True, savedir])
    
    # # generate flow of |A| at different temperature near Tc
    # devTc = [3,6,10]
    # datadic ={}
    # for acc in devTc:
    #     Tdevhi = Thi + 10**(-acc)
    #     Tdevlow = Tlow - 10**(-acc)
    #     print("At T = Tc + 10^-{:d}".format(acc))
    #     AnormH = normFlowHOTRG(Tdevhi,[chi,chi], iter_max, isDisp = verbose, 
    #                      isGilt = True, isSym = True,
    #                      gilt_eps = gilteps, cg_eps = cgeps,
    #                      N_gilt = Ngilt, legcut = legcut,
    #                      stableStep = stablek)[0]
    #     print("At T = Tc - 10^-{:d}".format(acc))
    #     AnormL = normFlowHOTRG(Tdevlow,[chi,chi], iter_max, isDisp = verbose, 
    #                      isGilt = True, isSym = True,
    #                      gilt_eps = gilteps, cg_eps = cgeps,
    #                      N_gilt = Ngilt, legcut = legcut,
    #                      stableStep = stablek)[0]
    #     datadic[acc] = [AnormL, AnormH]
    # datadicFile = savedirectory + "/flowDiffAcc.pkl"
    # with open(datadicFile,"wb") as f:
    #     pkl.dump(datadic, f)



if scheme == "hotrg" or scheme == "trg":
    pass
    # with open(singValFile, "wb") as f:
    #     pkl.dump([sarr, Adifflist], f)
elif scheme =="Gilt-HOTRG":
    Nsing = max([len(inner) for inner in slist])
    for i in range(len(slist)):
        temp = [slist[i][j] if j < len(slist[i]) else 0 for j in range(Nsing)]
        slist[i] = temp
    sarr = np.array(slist)
    with open(singValFile, "wb") as f:
        pkl.dump([sarr, Adifflist], f)
print("Step 2 finished! ")




        
