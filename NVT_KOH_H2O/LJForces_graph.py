# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:20:58 2023

@author: Jelle
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants as co
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

# Fonts
# plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.grid"] = "False"

# Sizes of specific parts

plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2
plt.rcParams['lines.linewidth'] = 2

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 0.75
plt.rcParams['xtick.major.pad'] = 7
plt.rcParams['ytick.major.pad'] = 7

plt.rcParams['xtick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = True

plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelpad'] = 7
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 15

figures = r'C:\Users\Jelle\Delft University of Technology\Jelle Lagerweij Master - Documents\General\Personal Thesis files\01 Defence Presentation\Figures'
plt.rcParams['savefig.directory'] = figures
# Done

###############################################################################

sig = 3.73e-10  # m
eps = 148*co.k # J
r = np.linspace(0, 10, 100000)*1e-10

F = -(24*eps/r)*(2*np.power(sig/r, 12) - np.power(sig/r, 6))
E = 4*eps*(np.power(sig/r, 12) - np.power(sig/r, 6))
plt.figure('Energy')
plt.plot(r*1e10, E*1e21)
plt.ylim(-2.5, 4)

plt.figure('Force')
plt.plot(r*1e10, F*1e12)
plt.ylim(-15, 15)
plt.xlim(0, 10)
plt.xlabel(r'$r$/[\si{\angstrom}]')
plt.ylabel(r'$F$/[\si{\pico\N}]')
plt.savefig(figures + '\Force')