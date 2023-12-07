"""
Created on Wed May 17 11:00:41 2023

@author: Jelle
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import scipy as sp
plt.close('all')


###############################################################################
# Setting the default figure properties for my thesis
plt.close('all')
plt.rcParams["figure.figsize"] = [6, 5]
plt.style.use('dark_background')
marker = ['o', 'x', '^', '>']

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "lualatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": 'serif',
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\RequirePackage{amsmath}",
        r"\RequirePackage{fontspec}",   # unicode math setup
        r"\setmainfont[Scale = MatchLowercase]{DejaVu Serif}",
        r"\setsansfont[Scale = MatchLowercase]{DejaVu Sans}",
        r"\setmonofont[Scale = MatchLowercase]{DejaVu Sans Mono}",
        r"\usepackage{unicode-math}",
        r"\setmathfont[Scale = MatchLowercase]{DejaVu Math TeX Gyre}", 
        r"\usepackage{siunitx}",
        r"\usepackage[version=3]{mhchem}"])}

mpl.use("pgf")
mpl.rcParams.update(pgf_with_latex)
plt.rcParams["axes.grid"] = "False"

# Sizes of specific parts
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2
plt.rcParams['lines.linewidth'] = 2

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize']= 0.75
plt.rcParams['xtick.major.pad'] = 7
plt.rcParams['ytick.major.pad'] = 7

plt.rcParams['xtick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.direction'] =  'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.direction'] =  'in'
plt.rcParams['ytick.right'] = True

plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelpad']= 7
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 15

figures = r'C:\Users\Jelle\Delft University of Technology\Jelle Lagerweij Master - Documents\General\Personal Thesis files\01 Defence Presentation\Figures'
plt.rcParams['savefig.directory'] = figures
# Done

###############################################################################

path = '../../../RPBE_Production/'
MD = ['AIMD/10ps/rdfs.npy', 'MLMD/100ns/rdfs.npy', 'MLMD/10ns/rdfs.npy']

rdf_AIMD = np.load(path+MD[0])
rdf_MLMD = np.load(path+MD[1])
r = rdf_AIMD[0, :]

rdf_exp = pd.read_csv("OO_Galamba.csv", header=None, sep=';', decimal=",")
rdf_exp = sp.interpolate.CubicSpline(rdf_exp[0], rdf_exp[1])
rdf_exp = rdf_exp(r)

rdf_exp2 = pd.read_csv("OO_Katayama.csv", header=None, sep=';', decimal=",")
rdf_exp2 = sp.interpolate.CubicSpline(rdf_exp2[0], rdf_exp2[1])
rdf_exp2 = rdf_exp2(r)

rdf_exp3 = pd.read_csv("OO_Imberti.csv", header=None, sep=';', decimal=",")
rdf_exp3 = sp.interpolate.CubicSpline(rdf_exp3[0], rdf_exp3[1])
rdf_exp3 = rdf_exp3(r)

rdf_scan = pd.read_csv("OO_Zhang.csv", header=None, sep=';', decimal=",")
rdf_scan = sp.interpolate.CubicSpline(rdf_scan[0], rdf_scan[1])
rdf_scan = rdf_scan(r)

rdf_habibi = pd.read_csv('rdf.dat', delim_whitespace=True, dtype=np.float64)
rdf_habibi = sp.interpolate.CubicSpline(rdf_habibi['Radius'],
                                        rdf_habibi['rdf__wat_wat'])
rdf_habibi = rdf_habibi(r)


plt.figure('OO')
plt.plot(r, rdf_AIMD[1, :], label='AIMD')
# plt.fill_between(r, rdf_AIMD[1, :] - rdf_AIMD[2, :], y2= rdf_AIMD[1, :] + rdf_AIMD[2, :], alpha=0.7)

plt.plot(r, rdf_MLMD[1, :], label='MLMD')
# plt.fill_between(r, rdf_MLMD[1, :] - rdf_MLMD[2, :], y2= rdf_MLMD[1, :] + rdf_MLMD[2, :], alpha=0.7)

plt.plot(r, rdf_habibi, label='TIP4P Transport')
# plt.plot(r, rdf_exp, label='Exp. NaCl')
plt.plot(r, rdf_scan, label='DeePMD SCAN')
# plt.plot(r, rdf_exp2, label='Exp. 5kbar')

plt.xlabel(r'$r$/[\si{\angstrom}]')
plt.yticks([0, 1, 2, 3], [0, 1, 2, 3])
plt.ylabel(r'$g_\text{OO}(r)$')
plt.legend()
plt.ylim(0, 3.5)
plt.xlim(2, 5)
plt.tight_layout()
# plt.savefig(figures + '/OO_KOH')

plt.figure('OO2')
plt.plot(r, rdf_AIMD[1, :], label='AIMD')
# plt.fill_between(r, rdf_AIMD[1, :] - rdf_AIMD[2, :], y2= rdf_AIMD[1, :] + rdf_AIMD[2, :], alpha=0.7)

plt.plot(r, rdf_MLMD[1, :], label='MLMD')
# plt.fill_between(r, rdf_MLMD[1, :] - rdf_MLMD[2, :], y2= rdf_MLMD[1, :] + rdf_MLMD[2, :], alpha=0.7)

plt.plot(r, rdf_exp3, label='Exp. KOH')
plt.plot(r, rdf_exp2, label='Exp. NaOH')

plt.xlabel(r'$r$/[\si{\angstrom}]')
plt.yticks([0, 1, 2, 3], [0, 1, 2, 3])
plt.ylabel(r'$g_\text{OO}(r)$')
plt.legend()
plt.ylim(0, 3.5)
plt.xlim(2, 5)
plt.tight_layout()
# plt.savefig(figures + '/OO_KOH2')

rdf_habibi = pd.read_csv('rdf.dat', delim_whitespace=True, dtype=np.float64)
rdf_habibi = sp.interpolate.CubicSpline(rdf_habibi['Radius'],
                                        rdf_habibi['rdf__wat_Oh'])
rdf_habibi = rdf_habibi(r)

plt.figure('OH-O')
plt.plot(r, rdf_AIMD[3, :], label='AIMD')
# plt.fill_between(r, rdf_AIMD[3, :] - rdf_AIMD[4, :], y2= rdf_AIMD[3, :] + rdf_AIMD[4, :], alpha=0.7)

plt.plot(r, rdf_MLMD[3, :], label='MLMD')
# plt.fill_between(r, rdf_MLMD[3, :] - rdf_MLMD[4, :], y2= rdf_MLMD[3, :] + rdf_MLMD[4, :], alpha=0.7)

plt.plot(r, rdf_habibi, label='TIP4P Transport')
plt.xlabel(r'$r$/[\si{\angstrom}]')
plt.ylabel(r'$g_\text{\ce{OH-}O}(r)$')
plt.legend()
plt.ylim(0, 5.5)
plt.xlim(2, 5)
plt.tight_layout()
# plt.savefig(figures +'/OHO_KOH')

rdf_habibi = pd.read_csv('rdf.dat', delim_whitespace=True, dtype=np.float64)
rdf_habibi = sp.interpolate.CubicSpline(rdf_habibi['Radius'],
                                        rdf_habibi['rdf__wat_K'])
rdf_habibi = rdf_habibi(r)

plt.figure('KO')
plt.plot(r, rdf_AIMD[5, :], label='AIMD')
# plt.fill_between(r, rdf_AIMD[5, :] - rdf_AIMD[6, :], y2= rdf_AIMD[5, :] + rdf_AIMD[6, :], alpha=0.7)

plt.plot(r, rdf_MLMD[5, :], label='MLMD')
# plt.fill_between(r, rdf_MLMD[5, :] - rdf_MLMD[2, :], y2= rdf_MLMD[5, :] + rdf_MLMD[6, :], alpha=0.7)
plt.plot(r, rdf_habibi, label='TIP4P Transport')
plt.yticks([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
plt.ylim(0, 4)
plt.xlim(2, 5)
plt.xlabel(r'$r$/[\si{\angstrom}]')
plt.ylabel(r'$g_\text{\ce{K+}O}(r)$')
plt.legend()
plt.tight_layout()
# plt.savefig(figures + '/KO_KOH')