#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:20:06 2020

@author: brucelyu
"""

import itertools
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
import os
from scipy.stats import entropy
import numpy as np

# Plot font size
middleSize = 12
plt.rc('font', size=middleSize)

marker = itertools.cycle(('.', 'x', '*','<','>','1','2','3','4',
                          's','p','*','h','H','+','x','D','d','|','_'))
lines =  itertools.cycle(('-','--'))
colors = itertools.cycle(('b','g'))

parser = argparse.ArgumentParser(
    "Draw the flow of TRG at a bound dimension")
parser.add_argument("--chi", dest = "chi", type = int,
                    help = "bound dimension (default: 10)",
                    default = 10)
parser.add_argument("--isGilt", help = "whether to use Gilts",
                        action = "store_true")
parser.add_argument("--gilteps", dest = "gilteps", type = float,
                        help = "a number smaller than which we think the" +
                        "singluar values for the environment spectrum is zero" +
                        "(default: 1e-7)",
                        default = 1e-7)
parser.add_argument("--scheme", dest = "scheme", type = str, 
                    help = "RG scheme to use",
                    choices = ["trg", "hotrg", "Gilt-HOTRG"],
                    default = "hotrg")
parser.add_argument("--Ngilt", dest = "Ngilt", type = int,
                    help = "How many times do we perform Gilt in oneHOTRG",
                    choices = [1,2], default = 1)
parser.add_argument("--legcut", dest = "legcut", type = int,
                    help = "number of leg to cut in gilt_hotrgplaq",
                    choices = [2,4], default = 4)
parser.add_argument("--isDiffAcc", help = "whether to draw flow at different T near Tc",
                        action = "store_true")


# read from argument parser
args = parser.parse_args()
chi = args.chi
isGilt = args.isGilt
isDiffAcc = args.isDiffAcc
gilteps = args.gilteps
scheme = args.scheme
Ngilt = args.Ngilt
legcut = args.legcut

# generate file name
# input and output file name
if scheme == "hotrg":
    figdir = "hotrg"
    chieps = "chi{:02d}".format(chi)
elif scheme == "trg":
    if isGilt:
        figdir = "gilt_trg_flow"
    else:
        figdir = "trgflow"
elif scheme == "Gilt-HOTRG":
    figdir = "gilt_hotrg{:d}{:d}_flow".format(Ngilt, legcut)
    chieps = "eps{:.0e}_chi{:02d}".format(gilteps, chi)
# read Tc if exists
savedirectory = "../out/" + figdir +  "/" + chieps
relTc = 1.0
Tcfile = savedirectory + "/Tc.pkl"
if not os.path.exists(Tcfile):
    relTc = 1.0
    print("No estimated Tc exists, set Tc = 1.")
else:
    with open(Tcfile,"rb") as f:
        Tlow, Thi = pkl.load(f)
    relTc = 1.0 * Tlow
    Tcerr = abs(Thi - Tlow) / (Tlow + Thi)
    outacc = int("{:e}".format(Tcerr)[-2:])
    print("Read the estimated Tc = {Tcval:.{acc}f}".format(Tcval = relTc,
                                                           acc = outacc))
    print("Related error of the estimate is {:.1e}".format(Tcerr))


# Plot the data file

singValFile = savedirectory + "/flowAtTc_fixSign.pkl"


if os.path.exists(singValFile):
    if scheme == "hotrg" or scheme == "trg":
        pass
        # with open(singValFile, "rb") as f:
        #     sarr, Adifflist = pkl.load(f)
    elif scheme =="Gilt-HOTRG":
        # read singular values flow and Adiff
        with open(singValFile, "rb") as f:
            sarr, Adifflist = pkl.load(f)
        # read norm flow
        Tsdata = "./data/" + figdir + "/" + chieps + "/otherTs.pkl"
        with open(Tsdata, "rb") as f:
            Anorm = pkl.load(f)[0]
        
if isDiffAcc:
    datadicFile = savedirectory + "/flowDiffAcc.pkl"
    with open(datadicFile,"rb") as f:
        datadic = pkl.load(f)
    plt.figure()
    for acc in datadic.keys():
        AnormL, AnormH = datadic[acc]
        tempMarker = next(marker)
        plt.plot(AnormL[1:],'b' + tempMarker + '-', alpha = 0.6,
                  label = r"$-10^{{-{0:d}}}$".format(acc))
        plt.plot(AnormH[1:],'k' + tempMarker + '--', alpha = 0.6,
                  label = r"$+10^{{-{0:d}}}$".format(acc))
    plt.yscale("log")
    plt.legend()
    plt.xlabel("RG step $n$")
    plt.xticks(np.arange(0,len(AnormL[1:]),5), np.arange(1, len(AnormL[1:])+1,5))
    plt.ylabel("$\Vert A^{(n)}\Vert$")
    plt.minorticks_off()
    # plt.yticks([0.2, 1],[r"$2\times 10^{-1}$",
    #                         r"$10^0$"],rotation=45)
   
    plt.savefig(savedirectory + "/AnormFlowDiffAcc.png", bbox_inches = 'tight', 
                dpi = 300)


if scheme == "Gilt-HOTRG":
    plt.figure()
    for k in range(sarr.shape[1]):
        plt.plot(sarr[:,k],"go-",alpha = 0.5)
        plt.title("Flow of spectrum $T = {:.10f}T_c, \chi = {:d}$".format(relTc,chi))
        plt.xlabel("RG step")
        plt.ylabel("Singluar values")
    plt.savefig(savedirectory + 
                "/flow_sing_fixSign.png", dpi = 300)
    
    enteps = 1e-10
    entro = entropy(sarr > 1e-8, axis = 1, base = 2)
    plt.figure()
    plt.plot(entro, "kx--", alpha = 0.6)
    plt.title("Flow of entropy of singular value spectrum")
    plt.xlabel("RG step")
    plt.ylabel("Entropy")
    plt.savefig(savedirectory + "/flow_singSpectr_fixSign.png", dpi = 300)
    
    
    plt.figure()
    plt.title("$\chi = ${:d}".format(chi))
    plt.plot(Anorm[1:],"kx--",label="Tc = {:.10f}".format(relTc))
    plt.yscale("log")
    plt.legend()
    plt.xlabel("RG step")
    plt.ylabel("$|A|$")
    plt.savefig(savedirectory + "/AnormFlow.png", dpi=300)
    
    
    plt.figure()
    plt.plot(Adifflist[:31], "ko--", alpha = 0.6)
    plt.yscale("log")
    plt.xlabel("RG step")
    plt.ylabel("$|A'-A|$")
    plt.savefig(savedirectory + 
                "/Adiff_fixSign.png", dpi = 300)






    