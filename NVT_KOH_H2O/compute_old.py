"""
Created on Wed May 17 11:00:41 2023

@author: Jelle
"""
import Class_diff_hopping as hop
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')


path = '../../KOH_Density/RPBE_MLMD/i_1350'
# folder = ['i_2', 'i_2', 'i_3', 'i_4', 'i_5']
folder = ['']

# n = [10, 20, 30, 40, 50, 60 ,70, 80, 90, 100, 200, 300, 400, 500, 600 ,700,
#       800, 900, 1000, 2000, 3000, 4000]
# n = [10, 20, 30, 40, 50, 60 ,70, 80, 90, 100, 200, 300, 400, 500, 600 ,700,
#       800, 900, 1000, 2000, 3000, 4000, 5000, 6000 ,7000, 8000, 9000, 10000,
#       20000, 30000]

# MSDOH = np.zeros(n_steps)
# MSDK = np.zeros(n_steps)
# visc = np.zeros(len(folder))
reaction_rate = np.zeros(len(folder))
# press = np.zeros(len(folder))
# Temp = np.zeros(len(folder))

for i in range(len(folder)):
    Traj = hop.Prot_Hop(path+folder[i], 20000)
    reaction_rate[i], index, loc_O = Traj.track_OH()
    loc_K = Traj.track_K()
    # msdOH = Traj.windowed_MSD(loc_O)
    # msdK = Traj.windowed_MSD(loc_K)
    # visc[i] = Traj.viscosity(cubicspline=64, plotting=True, padding=45000)
    # rdfs = Traj.rdf(cubicspline=64, plotting=True)

    # sanity checks
    # ave_beef, ave_bees = Traj.bayes_error(plotting=True, move_ave=False)
    # press[i] = Traj.pressure(plotting=True, move_ave=100)
    # e_tot = Traj.tot_energy(plotting=True, move_ave=100)
    # e_pot = Traj.pot_energy(plotting=True, move_ave=100)
    # e_kin = Traj.kin_energy(plotting=True, move_ave=100)
    # Temp[i] = Traj.temperature(plotting=True, move_ave=100)

    # MSDOH += msdOH
    # MSDK += msdK

    # t = np.arange(n_steps)

    # # multiple window loglog
    # plt.figure('multiple window loglog OH')
    # plt.loglog(Traj.t[1:], msdOH[1:], label='MSD ' + folder[i])

    # # multiple window loglog
    # plt.figure('multiple window loglog K')
    # plt.loglog(Traj.t[1:], msdK[1:], label='MSD ' + folder[i])

    # plt.figure('OH index')
    # plt.plot(Traj.t, index - Traj.N_H, label='MSD ' + folder[i])


# plt.figure('multiple window loglog OH')
# plt.loglog(Traj.t[1:], MSDOH[1:]/len(folder), label='Average MSD')
# plt.xlabel('time in s')
# plt.ylabel('MSD in Ang^2')
# plt.grid()
# plt.legend()
# plt.xlim(Traj.t[2], Traj.t[-1])


# plt.figure('multiple window loglog K')
# plt.loglog(Traj.t[1:], MSDK[1:]/len(folder), label='Average MSD')
# plt.xlabel('time in s')
# plt.ylabel('MSD in Ang^2')
# plt.grid()
# plt.legend()
# plt.xlim(Traj.t[2], Traj.t[-1])

# plt.figure('BEEF')
# plt.xlabel('time/[s]')
# plt.ylabel('Bayesian error estimate of force/[eV/Ang]')
# plt.xlim(Traj.t[0],Traj.t[-1])
# plt.legend()

# plt.figure('BEES')
# plt.xlabel('time/[s]')
# plt.ylabel('Bayesian error estimate of pressure/[kB]')
# plt.xlim(Traj.t[0],Traj.t[-1])
# plt.legend()

# plt.figure('pressure autocorrelation')
# plt.xlabel('time shift/[s]')
# plt.ylabel('Autocorrelation of pressure/[Pa^2]')
# plt.xlim(Traj.t[0],Traj.t[-1])
# plt.legend()

# plt.figure('pressure')
# plt.xlabel('time in/[s]')
# plt.ylabel('Pressure/[Pa]')
# plt.xlim(Traj.t[0],Traj.t[-1])
# plt.grid()
# plt.legend()


# plt.figure('OH index')
# plt.xlabel('time in/[fs]')
# plt.ylabel('Index of the O of the OH-/[-]')
# plt.xlim(Traj.t[0],Traj.t[-1])
# plt.ylim(0, 50)
# plt.grid()
# plt.legend()

# plt.figure('kinetic energy')
# plt.xlabel('time in/[fs]')
# plt.ylabel('kinetic energy/[J]')
# plt.xlim(Traj.t[0],Traj.t[-1])
# plt.grid()
# plt.legend()

# plt.figure('potential energy')
# plt.xlabel('time in/[fs]')
# plt.ylabel('potential energy/[J]')
# plt.xlim(Traj.t[0],Traj.t[-1])
# plt.grid()
# plt.legend()

# plt.figure('total energy')
# plt.xlabel('time in/[fs]')
# plt.ylabel('total energy/[J]')
# plt.xlim(Traj.t[0],Traj.t[-1])
# plt.grid()
# plt.legend()


# plt.figure('temperature')
# plt.xlabel('time in/[fs]')
# plt.ylabel('temperature/[K]')
# plt.xlim(Traj.t[0],Traj.t[-1])
# plt.grid()
# plt.legend()

# plt.figure('RDF O-O')
# plt.xlabel('radial distance/[Ang]')
# plt.ylabel('$g(r)$/[-]')
# plt.xlim(0, 10)
# plt.grid()
# plt.legend()


# t_test = Traj.t[n]
# MSD_test = MSDOH[n]
# diff_OH = Traj.diffusion(MSD_test, 1, t=t_test, plotting=True)
# MSD_test = MSDK[n]
# diff_K = Traj.diffusion(MSD_test, 1, t=t_test, plotting=True)

# visc = visc.mean()*1e3
# diff_OH = diff_OH*1e9
# diff_K = diff_K*1e9
# p = press.mean()*1e-5
# T = Temp.mean()

# print('pressure = ', p, 'temperature = ', T, 'viscosity = ', visc, 'D_OH = ', diff_OH, 'D_K = ', diff_K)