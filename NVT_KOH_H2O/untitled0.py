# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:59:39 2023

@author: Jelle
"""

import Class_diff_hopping_hdf5 as hop
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
import uncertainties as unc
from uncertainties import unumpy
plt.close('all')

###############################################################################
# Setting the default figure properties for my thesis
plt.close('all')
plt.rcParams["figure.figsize"] = [6, 5]
label_spacing = 1.1
marker = ['o', 'x', '^', '>']

# Fonts
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.grid"] = "False"

# Sizes of specific parts
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['xtick.major.pad'] = 7
plt.rcParams['ytick.major.pad'] = 5
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgewidth'] = 2
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['legend.fontsize'] = 'large'
# File properties and location
# plt.use("SVG")
figures = r'C:\Users\Jelle\Delft University of Technology\Jelle Lagerweij Master - Documents\General\Personal Thesis files\00 Literature study\figures'
# Done

###############################################################################

path = '../../RPBE_Production/MLMD/100ps_2/'
folder = ['i_1']#, 'i_2', 'i_3', 'i_4', 'i_5']

n_steps = 60000
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
    loc_K = Traj.track_K()
    loc_H2O = Traj.track_H2O(index)
    
    visc[i, :] = Traj.viscosity(cubicspline=10, plotting=True, padding=2e4)
    
    r, rdfs, n_conf = Traj.rdf(interpol=64, plotting=True, r_end=[4.2, 3.3, 3.5])
    n_OO[i], n_HO[i], n_KO[i] = n_conf
    OO[i, :], HO[i, :], KO[i, :] = rdfs

    d_OO[i] = r[np.argmax(OO[i, :])]
    d_HO[i] = r[np.argmax(HO[i, :])]
    d_KO[i] = r[np.argmax(KO[i, :])]

    # sanity checks
    ave_beef, ave_bees = Traj.bayes_error(plotting=True, move_ave=1500)
    press[i, :] = Traj.pressure(plotting=True, filter_width=1500, skip=100)
    e_kin[i, :] = Traj.kin_energy(plotting=True, filter_width=1500, skip=100)
    e_pot[i, :] = Traj.pot_energy(plotting=True, filter_width=1500, skip=100)
    e_tot[i, :] = Traj.tot_energy(plotting=True, filter_width=1500, skip=100)
    Temp[i, :] = Traj.temperature(plotting=True, filter_width=1500, skip=100)

    msdOH = Traj.windowed_MSD(loc_OH)
    msdK = Traj.windowed_MSD(loc_K)
    msdH2O = Traj.windowed_MSD(loc_H2O)

    t = np.arange(n_steps)

    # multiple window loglog
    plt.figure('multiple window loglog OH')
    plt.loglog(Traj.t[1:-1], msdOH[1:], label='MSD ' + folder[i])

    # multiple window loglog
    plt.figure('multiple window loglog K')
    plt.loglog(Traj.t[1:-1], msdK[1:], label='MSD ' + folder[i])
    
    # multiple window loglog
    plt.figure('multiple window loglog H2O')
    plt.loglog(Traj.t[1:-1], msdH2O[1:], label='MSD ' + folder[i])

    plt.figure('OH index')
    plt.plot(Traj.t[:-1], index - Traj.N_H, label='MSD ' + folder[i])

    n = np.arange(start=10000, stop=n_steps, step=25)
    n = np.arange(start=10000, stop=n_steps, step=100)
    t_test = Traj.t[n]
    MSD_test = msdOH[n]
    diff_OH[i] = Traj.diffusion(MSD_test, 1, t=t_test, plotting=False)
    MSD_test = msdK[n]
    diff_K[i] = Traj.diffusion(MSD_test, 1, t=t_test, plotting=False)
    MSD_test = msdH2O[n]
    diff_H2O[i] = Traj.diffusion(MSD_test, 1, t=t_test, plotting=False)

    # print('pressure = ', press[i, :],
    #       'temperature = ', Temp[i, :],
    #       'viscosity = ', visc[i, :],
    #       'D_OH =', diff_OH[i]*1e9,
    #       'D_K =', diff_K[i]*1e9,
    #       'reaction rate =', reaction_rate[i])

# rdfs= np.array([r, np.mean(OO, axis=0), np.std(OO, axis=0),
#                 np.mean(HO, axis=0), np.std(HO, axis=0),
#                 np.mean(KO, axis=0), np.std(KO, axis=0)])
# np.save(path+'rdfs.npy', rdfs)

plt.figure('multiple window loglog OH')
plt.loglog(Traj.t[1:n_steps], MSDOH[1:]/len(folder), label='Average MSD')
plt.xlabel('time in s')
plt.ylabel('MSD in Ang^2')
plt.grid()
plt.legend()
plt.xlim(Traj.t[2], Traj.t[-1])


plt.figure('multiple window loglog K')
plt.loglog(Traj.t[1:n_steps], MSDK[1:]/len(folder), label='Average MSD')
plt.xlabel('time in s')
plt.ylabel('MSD in Ang^2')
plt.grid()
plt.legend()
plt.xlim(Traj.t[2], Traj.t[-1])

plt.figure('BEEF')
plt.xlabel('time/[s]')
plt.ylabel('Bayesian error estimate of force/[eV/Ang]')
plt.xlim(Traj.t[0],Traj.t[-1])
plt.legend()

plt.figure('BEES')
plt.xlabel('time/[s]')
plt.ylabel('Bayesian error estimate of pressure/[kB]')
plt.xlim(Traj.t[0],Traj.t[-1])
plt.legend()


plt.figure('pressure')
plt.xlabel('time in/[s]')
plt.ylabel('Pressure/[Pa]')
plt.xlim(Traj.t[0], Traj.t[-1])
plt.grid()
plt.legend()

plt.figure('kinetic energy')
plt.xlabel('time in/[fs]')
plt.ylabel('kinetic energy/[J]')
plt.xlim(Traj.t[0], Traj.t[-1])
plt.grid()
plt.legend()

plt.figure('potential energy')
plt.xlabel('time in/[fs]')
plt.ylabel('potential energy/[J]')
plt.xlim(Traj.t[0], Traj.t[-1])
plt.grid()
plt.legend()

plt.figure('total energy')
plt.xlabel('time in/[fs]')
plt.ylabel('total energy/[J]')
plt.xlim(Traj.t[0], Traj.t[-1])
plt.grid()
plt.legend()

p = np.sum(unumpy.uarray(press[:, 0], press[:, 1]))/5
t = np.sum(unumpy.uarray(Temp[:, 0], Temp[:, 1]))/5
e_tot = np.sum(unumpy.uarray(e_tot[:, 0], e_tot[:, 1]))/5
e_kin = np.sum(unumpy.uarray(e_kin[:, 0], e_kin[:, 1]))/5
e_pot = np.sum(unumpy.uarray(e_pot[:, 0], e_pot[:, 1]))/5
n_oo = unc.ufloat(np.mean(n_OO), np.std(n_OO)/np.sqrt(5))
n_ho = unc.ufloat(np.mean(n_HO), np.std(n_HO)/np.sqrt(5))
n_ko = unc.ufloat(np.mean(n_KO), np.std(n_KO)/np.sqrt(5))
d_oo = unc.ufloat(np.mean(d_OO), np.std(d_OO)/np.sqrt(5))
d_ho = unc.ufloat(np.mean(d_HO), np.std(d_HO)/np.sqrt(5))
d_ko = unc.ufloat(np.mean(d_KO), np.std(d_KO)/np.sqrt(5))
t_r = unc.ufloat(np.mean(reaction_rate), np.std(reaction_rate)/np.sqrt(5)) 
t_r = 1/(1000*t_r)

viscosity = unc.ufloat(np.mean(visc[:, 0]),
                       np.std(visc[:, 1])/np.sqrt(5))
D_cor = co.k*t*2.837298/(6*np.pi*Traj.L*1e-10*viscosity)
D_H2O = unc.ufloat(np.mean(diff_H2O), np.std(diff_H2O)/np.sqrt(5)) + D_cor
D_OH = unc.ufloat(np.mean(diff_OH), np.std(diff_OH)/np.sqrt(5)) + D_cor
D_K = unc.ufloat(np.mean(diff_K), np.std(diff_K)/np.sqrt(5)) + D_cor
sigma = (1*co.eV**2)*(D_K + D_OH)/(co.k*t*(Traj.L*1e-10)**3)

print('first peak in Angstrom =', d_oo)
print('nOO =', n_oo)
print('first peak in Angstrom =', d_ko)
print('nKO =', n_ko)
print('first peak in Angstrom =', d_ho)
print('nHO =', n_ho)


print('E_tot in eV =', e_tot)
print('E_kin in eV', e_kin)
print('E_pot in eV', e_pot)
print('temperature in K =', t)
print('pressure in bar =', p)
print('reaction time in ps =', t_r)

print('viscosity in mPas (or cP) =', viscosity*1e3)
print('D_s correction in 10^-9 m^2/s =', D_cor*1e9)
print('D_s H2O in 10^-9 m^2/s =', D_H2O*1e9)
print('D_s K in 10^-9 m^2/s =', D_K*1e9)
print('D_s OH in 10^-9 m^2/s =', D_OH*1e9)
print('Electric conductivity in S/m', sigma)