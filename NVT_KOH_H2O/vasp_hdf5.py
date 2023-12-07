# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:50:56 2023

@author: Jelle
"""

from py4vasp import Calculation
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')

path = '../../KOH_Density/RPBE_MLMD/i_1350/'
folder = ['']
# folder = ['i_3/']

p_mean = np.zeros(len(folder))
T_mean = np.zeros(len(folder))
width = 0

def filter(signal, width):
    if width == 0:
        signal = signal
    else:
        signal = pd.DataFrame(signal)
        signal = signal.ewm(span=width).mean()
        signal = signal.to_numpy()
    return signal

for i in range(len(folder)):
    calc = Calculation.from_path(path+folder[i])
    stress = calc.stress[:].to_dict()['stress']
    steps = np.shape(stress)[0]
    t = np.arange(steps)
    pressure = np.mean(np.diagonal(stress, axis1=1, axis2=2), axis=1)*1e3

    energies = calc.energy[:].to_dict()
    T = energies['temperature    TEIN']
    E_kin = energies['kinetic energy EKIN']
    E_tot = energies['total energy   ETOTAL'] - energies['nose potential ES'] - energies['nose kinetic   EPS']
    E_pot = E_tot - E_kin
    
    plt.figure('temperature')
    plt.plot(t, filter(T, width), label=folder[i])
    T_mean[i] = np.mean(T[1500:])
    
    plt.figure('pressure')
    plt.plot(t, filter(pressure, width), label=folder[i])
    p_mean[i] = np.mean(pressure[1500:])

    plt.figure('energies')
    plt.plot(t, filter(E_tot, width), label='Etot '+folder[i])
    plt.plot(t, filter(E_kin, width), label='Ekin '+folder[i])
    plt.plot(t, filter(E_pot, width), label='Epot '+folder[i])

plt.figure('temperature') 
plt.legend()
plt.xlabel('time/[fs]')
plt.ylabel('temperature/[K]')
print(T_mean)


plt.figure('pressure') 
plt.legend()
plt.xlabel('time/[fs]')
plt.ylabel('pressure/[bar]')
print(p_mean)

plt.figure('energies')
plt.legend()
plt.xlabel('time/[fs]')
plt.ylabel('energy/[eV]')

