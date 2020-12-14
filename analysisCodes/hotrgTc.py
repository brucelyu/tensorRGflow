#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:54:09 2020
Perform the similar analysis of Hinczewski and Berker (2008), Fig. 4,
but here we use HOTRG, to determine critical temperature
@author: brucelyu
"""

from HOTRG import normFlowHOTRG
import pylab
import scipy.linalg as LA
import numpy as np
import os
import pickle as pkl
import argparse


# argument parser
parser = argparse.ArgumentParser(
    "Generate of flow of tensor represented by its Frobenius norm." + 
    " HOTRG is applied")
parser.add_argument("--chi", dest = "chi", type = int,
                    help = "bound dimension (default: 10)",
                    default = 10)
parser.add_argument("--maxiter", dest = "maxiter", type = int,
                   help = "maximal HOTRG iteration (default: 35)",
                   default = 35)
parser.add_argument("--rootiter", dest = "rootiter", type = int,
                    help = "iteration of finding the Tc (default:21)",
                    default = 21)  
parser.add_argument("--isGilt", help = "whether to use Gilts",
                        action = "store_true")
parser.add_argument("--isSym", help = "whether to use Z2 symmetric tensor",
                        action = "store_true")
parser.add_argument("--gilteps", dest = "gilteps", type = float,
                        help = "a number smaller than which we think the" +
                        "singluar values for the environment spectrum is zero" +
                        "(default: 1e-7)",
                        default = 1e-7)
parser.add_argument("--Tlow", dest = "Tlow", type = float,
                    help = "Estimated lower bound for critical temperature",
                    default = 0.989)

parser.add_argument("--Thi", dest = "Thi", type = float,
                    help = "Estimated higher bound for critical temperature",
                    default = 1.01)
# parser.add_argument("--ver", dest = "ver", type = str,
#                     help = "which version of gilt-hotrg to use",
#                     choices = ["old-Gilt-HOTRG", "Gilt-HOTRG-imp",
#                                "Gilt-Full-HOTRG"],
#                     default = "old-Gilt-HOTRG")
parser.add_argument("--cgeps", dest = "cgeps", type = float,
                        help = "a number smaller than which we think the" +
                        "singluar values for the environment in RG spectrum is zero" +
                        "(default: 1e-10)",
                        default = 1e-10)
parser.add_argument("--Ngilt", dest = "Ngilt", type = int,
                    help = "How many times do we perform Gilt in oneHOTRG",
                    choices = [1,2], default = 2)
parser.add_argument("--legcut", dest = "legcut", type = int,
                    help = "number of leg to cut in gilt_hotrgplaq",
                    choices = [2,4], default = 2)

parser.add_argument("--loopred", dest = "loopred", type = str,
                    help = "which loop reduction scheme to use",
                    choices = ["Gilt", "FET"], default = "Gilt")
parser.add_argument("--chitid", dest = "chitid", type = int,
                    help = "bound dimension (default: 5)",
                    default = 5)
parser.add_argument("--stbk", dest = "stbk", type = int,
                    help = "A int after which we will try to stabilize the gilt process",
                    default = 1000)

# read from argument parser
args = parser.parse_args()
chi = args.chi
iter_max = args.maxiter
iterRoot = args.rootiter
isGilt = args.isGilt
isSym = args.isSym
gilteps = args.gilteps
cgeps = args.cgeps
Ngilt = args.Ngilt
legcut = args.legcut
loopred = args.loopred
chitid = args.chitid
stablek = args.stbk

argsFET = {'chitid':chitid, 'maxiter':40, 'initscheme':'Gilt', 
               'giltdeg':0.5}

# # input and output file name
# if isGilt:
#     if rgver == "old-Gilt-HOTRG":
#         figdir = "gilt_hotrg_flow"
#     elif rgver == "Gilt-HOTRG-imp":
#         figdir = "gilt_hotrg_imp_flow"
#     else:
#         raise ValueError("--ver argument is not valid.")
# else:
#     figdir = "hotrgflow"
    
# input and output file name
if isGilt:
    figdir = "gilt_hotrg{:d}{:d}_flow".format(Ngilt, legcut)
else:
    figdir = "hotrg"


    
# create a directory with the name ?? to save all the figures
# if the directory does not exist
if isGilt:
    chieps = "eps{:.0e}_chi{:02d}".format(gilteps, chi)
else:
    chieps = "chi{:02d}".format(chi)

savedirectory = "../out/" + figdir +  "/" + chieps
if not os.path.exists(savedirectory):
    os.makedirs(savedirectory)

Tcfile = savedirectory + "/Tc.pkl"

# read Tc if exists
if not os.path.exists(Tcfile):
    Tlow = args.Tlow
    Thi = args.Thi
else:
    with open(Tcfile,"rb") as f:
        Tlow, Thi = pkl.load(f)

# The accuracy of the current estimation of the bisection method
Tdiff = abs(Thi - Tlow) / (Thi + Tlow)
accTc = "{:.2e}".format(Tdiff)
accTc = accTc[-2:]
print("Start the bisection algorithm...")
AnormL = normFlowHOTRG(Tlow,[chi,chi], iter_max, isDisp = False, 
                         isGilt = isGilt, isSym = isSym,
                         gilt_eps = gilteps, cg_eps = cgeps,
                         N_gilt = Ngilt, legcut = legcut,
                         stableStep = stablek)[0]
AnormH = normFlowHOTRG(Thi,[chi,chi], iter_max, isDisp = False, 
                         isGilt = isGilt, isSym = isSym,
                         gilt_eps = gilteps, cg_eps = cgeps,
                         N_gilt = Ngilt, legcut = legcut,
                         stableStep = stablek)[0]


for i in range(iterRoot): 
    print("Performing {}-th iteration to find the Tc".format(i+1))
    print("Tc estimated lowbound = {},\n highbound = {}".format(Tlow, Thi))
    # Examine the tensor RG flow at Ttry
    Ttry = 0.5*(Tlow + Thi)
    Tdiff = abs(Thi - Ttry) / Ttry
    AnormTry = normFlowHOTRG(Ttry,[chi,chi], iter_max, isDisp = False, 
                         isGilt = isGilt, isSym = isSym,
                         gilt_eps = gilteps, cg_eps = cgeps,
                         N_gilt = Ngilt, legcut = legcut,
                         stableStep = stablek)[0]
    # Plot the flow of tensor norm every three steps
    if (i+1) % 3 == 0:
        pylab.figure()
        pylab.title("Difference from Tc = {:.2e}, with $\chi = ${:d}".format(Tdiff,chi))
        pylab.plot(AnormL[1:],"bo--",label="low T = {:.10f}".format(Tlow))
        pylab.plot(AnormH[1:],"k.-",label="hi T = {:.10f}".format(Thi))
        pylab.plot(AnormTry[1:], "gx-.",label="try T = {:.10f}".format(Ttry))
        # pylab.ylim([0.10,0.50])
        pylab.yscale("log")
        pylab.legend()
        pylab.xlabel("RG step")
        pylab.ylabel("$|A|$")
        pylab.savefig(savedirectory + "/chi{:02d}{:02d}.png".format(chi,i+1), dpi=300)
        # pylab.show()
    
    # Calculate the "distances" of the Ttry tensor RG flow with
    # the original Thi and Tlow flows respectively
    if isGilt:
        distwithHi = np.abs(AnormTry[-1] - AnormH[-1])
        distwithL = np.abs(AnormTry[-1] - AnormL[-1])
    else:
        distwithHi = LA.norm(np.array(AnormTry[1:]) - np.array(AnormH[1:]))
        distwithL = LA.norm(np.array(AnormTry[1:]) - np.array(AnormL[1:]))
    # Determine Ttry is Thi or Tlow
    if distwithL < distwithHi:
        Tlow = Ttry
        AnormL = AnormTry.copy()
    else:
        Thi = Ttry
        AnormH = AnormTry.copy()
        
# save the lower and upper bound of Tc
with open(Tcfile,"wb") as f:
    pkl.dump([Tlow,Thi],f)
# Append all figures
orgfile = savedirectory + '/chi{:02d}*.png '.format(chi)
tarfile = savedirectory + '/allchi.png'
os.system('convert ' + orgfile + "-append " + tarfile)
os.system('rm ' + orgfile)
