#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:30:42 2020

@author: brucelyu
"""

import numpy as np
import argparse
import pickle as pkl
import itertools
import matplotlib.pyplot as plt
import os

marker = itertools.cycle(('o', 'x','.','*','x', '2','+', '3','4','x',
                          '.','*','x','>','+','x','D','d','|','_'))
lines =  itertools.cycle(('-','--','-.',':'))
colors = itertools.cycle(('b','g','r','c','m','y'))
parser = argparse.ArgumentParser(
    "Draw the scaling dimensions wrt RG step given a data file")

parser.add_argument("--Ndisp", dest = "Ndisp", type = int,
                    help = "number of scaling dimensions to draw (default:7)",
                    default = 17)
parser.add_argument("--chi", dest = "chi", type = int,
                help = "horizontal and vertical bound dimension (default: 10)",
                default = 10)

parser.add_argument("--gilteps", dest = "gilteps", type = float,
                    help = "a number smaller than which we think the" +
                    "singluar values for the environment spectrum is zero" +
                    "(default: 1e-7)",
                    default = 1e-7)
parser.add_argument("--Ngilt", dest = "Ngilt", type = int,
                    help = "How many times do we perform Gilt in oneHOTRG",
                    choices = [1,2], default = 1)
parser.add_argument("--legcut", dest = "legcut", type = int,
                    help = "number of leg to cut in gilt_hotrgplaq",
                    choices = [2,4], default = 4)

args = parser.parse_args()
Ndrawn = args.Ndisp
chi = args.chi
gilteps = args.gilteps

Ngilt = args.Ngilt
legcut = args.legcut

# direction where the data is saved
figdir = "gilt_hotrg{:d}{:d}_flow".format(Ngilt, legcut)
chieps = "eps{:.0e}_chi{:02d}".format(gilteps, chi)
## file to read or save scaling dimensions
scDFile = "../out/" + figdir +  "/" + chieps + "/scDim.pkl"


with open(scDFile,"rb") as f:
    klist,scDlist = pkl.load(f)

scDlist = np.array(scDlist)
scDN = scDlist.shape[1]
exactscD = [0, 0.125 ,1 , 1.125, 1.125, 2, 2, 2, 2, 2.125, 2.125, 2.125,
            3, 3, 3, 3, 3]
plt.figure()
for i in range(min(scDN,Ndrawn)):
    fmtsty = next(marker) + next(colors)
    plt.plot(klist,scDlist[:,i],fmtsty, 
             label = "$x_i = ${:.3f}".format(exactscD[i]))
    plt.hlines(exactscD[i],klist[0],klist[-1],colors = 'k',linestyles='dashed')
    plt.title("$\chi = ${:d} and gilt $\epsilon = ${:.0e}".format(chi, gilteps))
    plt.xlabel("RG step")
    plt.ylabel("scaling dimensions")
    plt.ylim([-0.1,3.125])
    #plt.legend()
plt.savefig(os.path.splitext(scDFile)[0]+ ".png",dpi = 300)