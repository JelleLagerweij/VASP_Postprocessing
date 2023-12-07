"""
Created on Wed May 17 11:00:41 2023

@author: Jelle
"""
import class_diff_hopping_hdf5 as hop
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants as co
import uncertainties as unc
from uncertainties import unumpy
plt.close('all')

# ###############################################################################
# # Setting the default figure properties for my thesis
# plt.close('all')
# plt.rcParams["figure.figsize"] = [6, 5]
# label_spacing = 1.1
# marker = ['o', 'x', '^', '>']

# # Fonts
# plt.rcParams["svg.fonttype"] = "none"   
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["axes.grid"] = "False"

# # Sizes of specific parts
# plt.rcParams['axes.labelsize'] = 'large'
# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams['xtick.major.pad'] = 7
# plt.rcParams['ytick.major.pad'] = 5
# plt.rcParams['xtick.labelsize'] = 'large'
# plt.rcParams['ytick.labelsize'] = 'large'
# plt.rcParams['lines.markersize'] = 10
# plt.rcParams['lines.markeredgewidth'] = 2
# plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['xtick.major.size'] = 5
# plt.rcParams['xtick.major.width'] = 2
# plt.rcParams['ytick.major.size'] = 5
# plt.rcParams['ytick.major.width'] = 2
# plt.rcParams['legend.fontsize'] = 'large'
# plt.rcParams['legend.frameon'] = False
# plt.rcParams['legend.labelspacing'] = 0.75
# # File properties and location
# # Done

# ###############################################################################

###############################################################################
# Setting the default figure properties for my thesis
plt.close('all')
plt.rcParams["figure.figsize"] = [6, 5]
plt.style.use('dark_background')
marker = ['o', 'x', '^', '>']

# File properties and location
# mpl.use("SVG")

# pgf_with_latex = {                      # setup matplotlib to use latex for output
#     "pgf.texsystem": "lualatex",        # change this if using xetex or lautex
#     "text.usetex": True,                # use LaTeX to write all text
#     "font.family": 'sans',
#     "pgf.rcfonts": False,    # don't setup fonts from rc parameters
#     "pgf.preamble": "\n".join([ # plots will use this preamble
#         r"\RequirePackage{amsmath}",
#         r"\RequirePackage{fontspec}",   # unicode math setup
#         r"\setmainfont[Scale = MatchLowercase]{DejaVu Serif}",
#         r"\setsansfont[Scale = MatchLowercase]{DejaVu Sans}",
#         r"\setmonofont[Scale = MatchLowercase]{DejaVu Sans Mono}",
#         r"\RequirePackage{arevmath}",
#         r"\DeclareMathSizes{12}{9.4}{7.6}{6}", 
#         r"\usepackage{siunitx}",
#         r"\usepackage[version=3]{mhchem}"])}

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


# path = r'../../RPBE_Production/MLMD/100ps_Exp_Density/'
path = r'../../RPBE_Production/AIMD/10ps/'
folder = ['i_1', 'i_2', 'i_3', 'i_4', 'i_5']

n_KOH = 1
n_H2O = 55

n_steps = 10000
# n_steps = 10000
MSDOH = np.zeros(n_steps)
MSDK = np.zeros(n_steps)
visc = np.zeros((len(folder), 2))
reaction_rate = np.zeros(len(folder))
press = np.zeros((len(folder), 2))
e_kin = np.zeros((len(folder), 2))
e_pot = np.zeros((len(folder), 2))
e_tot = np.zeros((len(folder), 2))
Temp = np.zeros((len(folder), 2))
diff_OH = np.zeros(len(folder))
diff_K = np.zeros(len(folder))
diff_H2O = np.zeros(len(folder))
OO = np.zeros((len(folder), 1984))
HO = np.zeros((len(folder), 1984))
KO = np.zeros((len(folder), 1984))
d_OO = np.zeros(len(folder))
d_HO = np.zeros(len(folder))
d_KO = np.zeros(len(folder))
n_OO = np.zeros(len(folder))
n_HO = np.zeros(len(folder))
n_KO = np.zeros(len(folder))


for i in range(len(folder)):
    Traj = hop.Prot_Hop(path+folder[i])
    reaction_rate[i], index, loc_OH = Traj.track_OH(rdf=[32, 2, 5])

    plt.figure('OH index')
    plt.plot(Traj.t*1e12, index - Traj.N_H, label='run ' + str(i+1))

    nreact_t = np.zeros(Traj.n_max)
    for j in range(1, Traj.n_max):
        nreact_t[j] = np.count_nonzero(index[1:j] != index[0:j-1])
    
    plt.figure('n_reactions')
    plt.plot(Traj.t*1e12, nreact_t, label='run ' + str(i+1))
    
# plt.figure('OH index')
# plt.xlabel('time/[ps]')
# plt.ylabel('Index of oxygen of OH-')
# plt.xlim(0, 100)
# plt.ylim(-5, 60)
# plt.legend()
# plt.savefig(figures+'/index1')

# plt.xlim(2.5, 4)
# plt.savefig(figures+'/index2')


# plt.figure('n_reactions')
# plt.xlabel('time/[ps]')
# plt.ylabel('Number of reactions')
# plt.xlim(0, 100)
# plt.ylim(0, 430)
# plt.legend()
# plt.savefig(figures+'/reax1')

# plt.xlim(0, 5)
# plt.ylim(0, 45)
# plt.savefig(figures+'/reax2')
