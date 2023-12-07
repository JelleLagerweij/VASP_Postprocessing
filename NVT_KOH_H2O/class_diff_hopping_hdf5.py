# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:20:40 2023

@author: Jelle
"""

import numpy as np
import pandas as pd
import scipy as sp
import re
import scipy.constants as co
import scipy.optimize as opt
import matplotlib.pyplot as plt
import freud
from py4vasp import Calculation
import uncertainties as unc

class Prot_Hop:
    """Postprocesses class for NVT VASP simulations of aqueous KOH."""

    def __init__(self, folder, T_set=325, r_bond=1.1, dt=1e-15):
        """
        Postprocesses class for NVT VASP simulations of aqueous KOH.

        It can compute the total system viscosity and the diffusion coefficient
        of the K+ and the OH-. The viscosity is computed using Green-Kubo from
        the pressure correlations and the diffusion coefficient using Einstein
        relations from the positions. The reactive OH- trajactory is computed
        using minimal and maximal bond lengths.

        Parameters
        ----------
        folder : string
            Path to folder which holds VASP outputs.
        r_bond : float, optional
            The typical bond length of the OH bond in OH- and H2O. The default
            is 1.1.
        dt : float, optional
            The timestepsize in fs of the calculation. The default is 1e-15s.

        Returns
        -------
        None.

        """
        # Make flexible lateron and integrate as optional argument in class.
        self.r_bond = r_bond  # Angstrom
        self.dt = dt
        self.folder = folder + '/'
        self.species = ['H', 'O', 'K']
        self.T_ave = T_set

        # Normal startup behaviour
        self.setting_properties()

    def setting_properties(self):
        """
        Load the XDATCAR.

        This will compute the basis set vectors, the number of total particles
        and stores the XDATCAR in a large dataframe.

        Returns
        -------
        None.

        """
        # Set which O and H belongs to OH to unknown
        self.i_O = 0
        self.i_H = 0

        # The Position Section
        self.df = Calculation.from_path(self.folder)
        data = self.df.structure[:].to_dict()

        # Getting number of Particles per species nicely packaged
        self.N = np.array([data['elements'].count(self.species[0]),
                          data['elements'].count(self.species[1]),
                          data['elements'].count(self.species[2])])

        self.N_tot = sum(self.N)
        self.N_H = self.N[self.species.index('H')]
        self.N_O = self.N[self.species.index('O')]
        self.N_K = self.N[self.species.index('K')]

        self.H = np.arange(self.N_H)
        self.O = np.arange(self.N_H, self.N_H + self.N_O)
        self.K = np.arange(self.N_H + self.N_O, self.N_H + self.N_O + self.N_K)

        # Calculating the number of OH-
        self.n_OH = -(self.N_H - 2*self.N_O)
        self.shift = np.zeros((self.n_OH, 1, 3))
        self.n_list = 10  # number of Hydrogens in the neighbor list

        self.L = data['lattice_vectors'][0, 0, 0]
        self.pos = self.L*data['positions']
        self.n_max = len(self.pos[:, 0, 0])
        self.t = np.arange(self.n_max)*self.dt
        self.find_O_i(0)

        self.stress = self.df.stress[:].to_dict()['stress']
        self.energy  = self.df.energy[:].to_dict()

    def find_O_i(self, n):
        """
        Find the index of the Oxygen beloging to the OH-.

        This function searches for the index of the Oxygen belonging to the OH-
        particles. It automatically creates a neighbor list as well as an array
        which holds the unwraped location of the real OH particle.

        Parameters
        ----------
        n : integer
            Timestep of the assesment.
        Returns
        -------
        None.

        """
        if n == 0:
            startup = True
        else:
            startup = False
            
        # Retrieve ALL particle positions.
        x = self.pos[n, :, :]
        if startup is False:
            O_i_old = self.O_i

        # Compute ALL particle distances^2 including PBC.
        r = np.broadcast_to(x, (self.N_tot, self.N_tot, 3))
        r_vect = (r - r.transpose(1, 0, 2) + self.L/2) % self.L - self.L/2
        d_sq = np.einsum('ijk, ijk->ij', r_vect, r_vect, optimize='optimal')

        # Compute which atom index belongs to the OH- Oxygen.
        d_sq = d_sq[np.ix_(self.H, self.O)]
        number = (d_sq < self.r_bond**2).sum(axis=0)
        O_i = np.where(number == 1) + self.N_H

        r_bond = self.r_bond
        while not (O_i.size == self.n_OH):
            if O_i.size > self.n_OH:
                r_bond *= 1.01
            if O_i.size < self.n_OH:
                r_bond *= 0.99
            number = (d_sq < (r_bond - 0.02)**2).sum(axis=0)
            O_i = np.where(number == 1) + self.N_H

        self.O_i = O_i.flatten()
        # Create neighborlist with this data as well. Do it here as there is
        # no need to compute distances again.
        d_sq = d_sq[:, self.O_i-self.N_H].flatten()
        self.H_n = np.sort(np.argpartition(d_sq, self.n_list)[:self.n_list])
        self.n_list_c = 0  # reset counter for neighborlist updating

        if startup is False:
            # Check which is the correct image of the particle that is reacted
            # with And update the number of boundary passes that it has had.
            # compute true displacement
            dis = (x[O_i, :] - x[O_i_old, :] + self.L/2) % self.L - self.L/2
            real_loc = x[O_i_old, :] + dis
            self.shift += real_loc - x[O_i, :]
            self.x_O_i = x[O_i, :]

    def find_H_n(self, n):
        """
        Update the neighborlist of the Hydrogen atoms.

        The neighborlist of the Hydrogen atoms makes it fast to compute if the
        OH- has hoped, however, it needs to be updated every so often.

        Parameters
        ----------
        n : integer
            Timestep at which the neighborlist is updated.

        Returns
        -------
        None.

        """
        # Retrieve particle positions O_i and ALL Hydrogens
        x_O_i = self.pos[n, self.O_i, :]
        x_H = self.pos[n, self.H, :]

        # Compute particle distances^2 between O_i and all H including PBC.
        rel_r = (x_H - x_O_i + self.L/2) % self.L - self.L/2
        d_sq = np.einsum('ij, ij -> i', rel_r, rel_r)

        # Get index of the neighborlist Hydrogen
        self.H_n = np.sort(np.argpartition(d_sq, self.n_list)[:self.n_list])
        self.n_list_c = 0  # reset counter for neighborlist updating

    def track_OH(self, N_neighbor=100, rdf=False):
        """
        Run the timeloop and tracks the OH- and store its index and location.
        
        The unwraped locations and the atomic index of the Oxygen belonging to
        the OH- are returned.

        Parameters
        ----------
        N_neighbor : integer, optional
            Number of timesteps in between an update of the neighborlist. The
            default is 100.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        # Prep things to track
        self.O_i_stored = np.zeros(self.n_max)
        self.O_loc_stored = np.zeros((self.n_max, self.n_OH, 3))
        counter = 0
        # Run the big loop
        for j in range(self.n_max):
            self.x_O_i = self.pos[j, self.O_i, :]
            self.x_H_n = self.pos[j, self.H_n, :]
            rel_r = (self.x_H_n - self.x_O_i + self.L/2) % self.L - self.L/2
            d_sq = np.einsum('ij, ij -> i', rel_r, rel_r)
            n = np.count_nonzero(d_sq < self.r_bond**2)

            if n > 1:
                # New O_i particle to find
                self.find_O_i(j)

            if self.n_list_c > N_neighbor:
                # Update neighbor list ever so often
                self.find_H_n(j)

            self.O_i_stored[j] = self.O_i
            self.O_loc_stored[j, :, :] = self.x_O_i + self.shift
            self.n_list_c += 1
            if self.O_i != self.O_i_stored[j-1]:
                counter += 1

            if rdf is not False and j % 10 == 0:
                self.rdf_compute(j, rdf)

        react_rate = counter/(self.n_max*self.n_OH)
        O_i_stored = self.O_i_stored
        O_i_loc_stored = self.O_loc_stored
        return react_rate, O_i_stored, O_i_loc_stored

    def rdf_compute(self, n, properties):
        nb = properties[0]
        rstart = properties[1]
        rstop = properties[2]
        
        # Retrieve ALL particle positions.
        x = self.pos[n, :, :]

        # Compute ALL particel distances:
        r = np.broadcast_to(x, (self.N_tot, self.N_tot, 3))
        r_vect = (r - r.transpose(1, 0, 2) + self.L/2) % self.L - self.L/2
        d = np.sqrt(np.einsum('ijk, ijk->ij', r_vect, r_vect,
                              optimize='optimal'))

        if n == 0:
            self.r = np.histogram(d, bins=nb,
                                  range=(rstart, rstop))[1]
            self.r = self.r[:-1]
            self.g_OO = np.zeros(len(self.r))
            self.g_HO = np.zeros(len(self.r))
            self.g_KO = np.zeros(len(self.r))

        # First all O_w O_w interactions as this is easier
        d_OO = d[self.O[0]:self.O[-1]+1, self.O[0]:self.O[-1]+1]
        d_OO = np.delete(d_OO, self.O_i-self.O[0], axis=0)
        d_OO = np.delete(d_OO, self.O_i-self.O[0], axis=1)
        n_OO = np.histogram(d_OO, bins=nb, range=(rstart, rstop))[0]
        rescale1 = 4*np.pi*(self.r**2)*(self.r[1]-self.r[0])  # 4pir**2 dr part
        rescale2 = (self.N_O - self.n_OH)*(self.N_O - self.n_OH - 1)
        self.g_OO += (self.L**3*n_OO)/(rescale1*rescale2*self.n_max/10)

        # Then the O_H O_w interactions
        d_HO = d[self.O_i, self.O[0]:self.O[-1]+1]
        d_HO = np.delete(d_HO, self.O_i-self.O[0], axis=1)
        n_HO = np.histogram(d_HO, bins=nb, range=(rstart, rstop))[0]
        rescale1 = 4*np.pi*(self.r**2)*(self.r[1]-self.r[0])  # 4pir**2 dr part
        rescale2 = (self.N_O - self.n_OH)*(self.n_OH)
        self.g_HO += (self.L**3*n_HO)/(rescale1*rescale2*self.n_max/10)

        # Then the K O_w interactions
        d_KO = d[self.K, self.O[0]:self.O[-1]+1]
        d_KO = np.delete(d_KO, self.O_i-self.O[0], axis=1)
        n_KO = np.histogram(d_KO, bins=nb, range=(rstart, rstop))[0]
        rescale1 = 4*np.pi*(self.r**2)*(self.r[1]-self.r[0])  # 4pir**2 dr part
        rescale2 = (self.N_O - self.n_OH)*(self.N_K)
        self.g_KO += (self.L**3*n_KO)/(rescale1*rescale2*self.n_max/10)
        
    def rdf(self, interpol=False, plotting=False, r_end=[3.4, 3.2, 3.5]):
        if interpol is False:
            r_small = self.r
            g_OO = self.g_OO
            g_HO = self.g_HO
            g_KO = self.g_KO

        else:
            r_small = np.arange(start=self.r[0], stop=self.r[-1],
                                step=(self.r[1] - self.r[0])/interpol)
            g_OO = sp.interpolate.CubicSpline(self.r, self.g_OO)
            g_OO = g_OO(r_small)
            g_HO = sp.interpolate.CubicSpline(self.r, self.g_HO)
            g_HO = g_HO(r_small)
            g_KO = sp.interpolate.CubicSpline(self.r, self.g_KO)
            g_KO = g_KO(r_small)

        # Calculating the configuration numbers
        stop_i = np.argmin(abs(r_small - r_end[0]))
        n_OO = np.trapz(g_OO[:stop_i]*(r_small[:stop_i])**2, r_small[:stop_i])
        n_OO *= 4*np.pi*((self.N_O-self.n_OH)/(self.L**3))

        stop_i = np.argmin(abs(r_small - r_end[1]))
        n_HO = np.trapz(g_HO[:stop_i]*(r_small[:stop_i])**2, r_small[:stop_i])
        n_HO *= 4*np.pi*((self.N_O-self.n_OH)/(self.L**3))

        stop_i = np.argmin(abs(r_small - r_end[2]))
        n_KO = np.trapz(g_KO[:stop_i]*(r_small[:stop_i])**2, r_small[:stop_i])
        n_KO *= 4*np.pi*((self.N_O-self.n_OH)/(self.L**3))

        if plotting is True:
            j = re.findall(r'\d+', self.folder)[-1]
            string = ' run ' + j
            plt.figure('OO')
            plt.plot(r_small, g_OO, label=string)
            plt.figure('OH-O')
            plt.plot(r_small, g_HO, label=string)
            plt.figure('KO')
            plt.plot(r_small, g_KO, label=string)

        rdfs = [g_OO, g_HO, g_KO]
        n_conf = [n_OO, n_HO, n_KO]
        return r_small, rdfs, n_conf

    def track_K(self):
        """
        Retrieves the locations of the K ions at all timepoints. It only slices
        the atom poistions of the ions with species K from the positions array.

        Returns
        -------
        K_loc : numpy array
            Returns the xyz postions of all K ions at all timesteps. The
            output shaped as, n_steps, N_K, 3, is very suitable for the msd
            code.

        """        
        # self.K_loc_stored = np.zeros((self.n_max, self.n_OH, 3))
        # for j in range(self.n_max):
        #     self.K_loc_stored[j, :, :] = self.read_XDATCAR(j, index=self.K)
        self.K_loc_stored = self.pos[:, self.K, :]
        K_loc = np.copy(self.K_loc_stored)
        return K_loc
    
    def track_H2O(self, index):
        """
        Retrieves the locations of the K ions at all timepoints. It only slices
        the atom poistions of the ions with species K from the positions array.

        Returns
        -------
        OO_loc : numpy array
            Returns the xyz postions of all water molecules all timesteps. The
            output shaped as, n_steps, N_K, 3, is very suitable for the msd
            code.

        """
        O_loc_stored = self.pos[:, self.O, :]
        index = (index - self.O[0]).astype(int)  # the atomic number O of OH-
        indexes = np.r_[0:index[0], index[0]+1:self.N_O-self.n_OH]  # the other
        indexes = indexes.astype(int)
        index_old = index[0]
        loc = np.zeros((self.n_max, self.N_O-self.n_OH - 1, 3))
        for i in range(self.n_max):
            if index[i] == index_old:
                loc[i, :, :] = O_loc_stored[i, indexes, :]
            else:
                indexes[np.where(indexes == index[i])] = index_old
                loc[i, :, :] = O_loc_stored[i, indexes, :]
                index_old = index[i]
        return loc

    def single_MSD(self):
        """
        Depreciated, self made single window MSD code.

        The MSD of a single atom and its position is returned.

        Returns
        -------
        d_sq : numpy array
            MSD per timestep from single window.

        """
        rel_r = (self.O_loc_stored - self.O_loc_stored[0, :])
        d_sq = np.einsum('ij, ij -> i', rel_r, rel_r)
        return d_sq

    def single_MSD_freud(self):
        """
        Single window MSD code refering the freud package.

        The MSD of a multiple atoms and their position is returned.

        Returns
        -------
        calculation.msd : numpy array
            MSD per timestep from single window.

        """
        # reshape the locations
        loc = self.O_loc_stored
        msd = freud.msd.MSD(mode='direct')

        # compute MSD
        calculation = msd.compute(loc)
        return calculation.msd

    def windowed_MSD(self, loc_array, Npart):
        """
        Multiple windows MSD code using the freud package.

        The MSD of multiple atoms and their position is returned.


        Parameters
        ----------
        loc_array : numpy array
            This array should hold the positions of the particles of a certain
            type for all time steps. The size has to be: (n_steps, n_atoms, 3)

        Returns
        -------
        calculation.msd : numpy array
            MSD per timestep from multiple windows.

        """
        # reshape the locations
        loc = loc_array
        msd = freud.msd.MSD(mode='window')

        # compute MSD
        calculation = msd.compute(loc)
        return calculation.msd*Npart

    def viscosity(self, plotting=False, cubicspline=False, padding=0):
        padding = int(padding)
        p_ab = np.zeros((5, len(self.t)))
        p_ab[0, :] = self.stress[:, 0, 1]
        p_ab[1, :] = self.stress[:, 0, 2]
        p_ab[2, :] = self.stress[:, 1, 2]
        p_ab[3, :] = (self.stress[:, 0, 0] + self.stress[:, 1, 1])/2
        p_ab[4, :] = (self.stress[:, 1, 1] + self.stress[:, 2, 2])/2
        p_ab = p_ab[:, padding:]
        p_ab *= 1e8  # to go from kbar to Pa

        # Calculate seperately
        factor = np.power(self.L*1e-10, 3)/(co.k*self.T_ave)
        visc = np.zeros(5)
        for i in range(len(visc)):
            data = p_ab[i, :]
            ave = np.mean(data)  # in Pa
            data -= ave
            acorr = np.correlate(data, data, 'full')[len(data)-1:]

            if padding == 0:
                t = self.t
            else:
                t = self.t[:-padding]
            
            if cubicspline is not False:
                steps = len(data)*cubicspline
                t_small = np.linspace(0, steps, num=steps)*self.dt/cubicspline
                spl_acorr = sp.interpolate.CubicSpline(t, acorr)
                acorr = spl_acorr(t_small)
                t = t_small

            if plotting is True:
                j = re.findall(r'\d+', self.folder)[-1]
                string = ' run ' + j
                plt.figure('pressure autocorrelation')
                plt.plot(t, acorr, label=string)

            integral = sp.integrate.simpson(acorr, x=t)
            visc[i] = integral*factor

        viscosity = np.array([np.mean(visc), np.std(visc)])
        return viscosity

    def diffusion(self, MSD_in, N_specie, t=False, m=False, plotting=False):
        # Settings for the margins and fit method.
        margin = 0.005  # cut away range at left and right side
        Minc = 150  # minimum number of points included in the fit
        Mmax = 200  # maximum number of points included in the fit
        er_max = 0.2  # maximum allowed error

        if t is False:
            t = self.t

        t_log = np.log10(t)
        MSD_log_in = np.log10(np.abs(MSD_in))
        ibest = 'failed'
        jbest = 'failed'
        mbest = 0

        for i in range(int(margin*len(t_log)), int((1-margin)*len(t_log))-Minc):
            for j in range(Minc, min(Mmax, int((1-margin)*len(t_log))-Minc-i)):
                if (t[i] != t[i+1]):
                    p, res, aa, aa1, aa3 = np.polyfit(t_log[i:i+j],
                                                      MSD_log_in[i:i+j], 1,
                                                      full=True)
                    mlog = p[0]
                    if (mlog > (1-er_max) and mlog < (1+er_max) and abs(mbest-1) > abs(mlog-1)):
                        mbest = mlog
                        jbest = j
                        ibest = i

        # Make sure to return NaN (not included in np.nanmean() for averaging).
        if ibest == 'failed':
            D = np.nan
            t_fit = t[0]
            fit = MSD_in[0]

        else:
            D, b = np.polyfit(t[ibest:ibest+jbest],
                              MSD_in[ibest:ibest+jbest], 1)

            # Test box size to displacement comparison.
            if np.abs(MSD_in[ibest+jbest]-MSD_in[ibest]) < m**2 and type(m) is not bool:
                print('MSD fit is smaller than simulation box',
                      MSD_in[ibest+jbest]-MSD_in[ibest], 'versus', m**2)

            t_fit = t[ibest:ibest+jbest]
            fit = D*t_fit + b

        if plotting is True:
            plt.figure('Diffusion fitting')
            plt.loglog(t, MSD_in, 'o', label='data')
            plt.loglog(t_fit, fit, '-.', label='fit')
            plt.grid()
            plt.legend()

        fact = (1e-20)/(6*N_specie)
        return D*fact

    def pressure(self, plotting=False, filter_width=0, skip=1):
        """
        Calculate the mean pressure of the simulation.

        Optionally the results are shown in a graphand with filter.

        Parameters
        ----------
        plotting : Boolean, optional
            Optional plots the pressure as function of time. The default is
            False.
        filter_width : Integer, optional
            Filter width of the moving average filter. The default is 0, which
            results in no filtering.

        Returns
        -------
        pressure : Array of 2 Floats
            The average presure and error of mean of the simulation run in bar.

        """
        # Retrieve the pressure from the dataframe. Calculate the mean of the
        # diagonal components
        press = np.mean(np.diagonal(self.stress, axis1=1, axis2=2), axis=1)*1e3
        pressure = statistics(press[0::skip])

        # plot if asked for
        if plotting is True:
            # filter if needed
            press = filter(press, filter_width)

            # set labels of plot
            i = re.findall(r'\d+', self.folder)[-1]
            string = ' run ' + i

            plt.figure('pressure')
            plt.plot(self.t*1e12, press, label=string)

        return pressure

    def bayes_error(self, beefile='BEEF.dat', plotting=False, move_ave=False):
        self.beefile = self.folder + beefile

        df = pd.read_table(self.beefile, header=None, skiprows=14,
                           delim_whitespace=True, usecols=[2, 3, 4, 5, 6, 7])
        bee = df.to_numpy()[1:, :]

        beef_max = bee[:, 1]
        beef_ave = bee[:, 2]
        beef_tresh = bee[:, 3]
        bees_max = bee[:, 4]
        bees_ave = bee[:, 5]

        if plotting is True:
            if move_ave is not False:
                beef_max = pd.DataFrame(beef_max)
                beef_max = beef_max.ewm(span=move_ave).mean()
                beef_max = beef_max.to_numpy()

                beef_ave = pd.DataFrame(beef_ave)
                beef_ave = beef_ave.ewm(span=move_ave).mean()
                beef_ave = beef_ave.to_numpy()

                beef_tresh = pd.DataFrame(beef_tresh)
                beef_tresh = beef_tresh.ewm(span=move_ave).mean()
                beef_tresh = beef_tresh.to_numpy()

                bees_max = pd.DataFrame(bees_max)
                bees_max = bees_max.ewm(span=move_ave).mean()
                bees_max = bees_max.to_numpy()

                bees_ave = pd.DataFrame(bees_ave)
                bees_ave = bees_ave.ewm(span=move_ave).mean()
                bees_ave = bees_ave.to_numpy()

            i = re.findall(r'\d+', self.folder)[-1]
            string = ' run ' + i
            plt.figure('BEEF')
            plt.plot(self.t*1e12, beef_max, label='Max BEEF' + string)
            plt.plot(self.t*1e12, beef_ave, label='Ave BEEF' + string)
            # plt.plot(self.t*1e12, beef_tresh, label='Treshold BEEF' + string)

            plt.figure('BEES')
            plt.plot(self.t*1e12, bees_max*1e3, label='Max BEES' + string)
            plt.plot(self.t*1e12, bees_ave*1e3, label='Ave BEES' + string)

        return beef_max.mean(), bees_max.mean()

    def kin_energy(self, plotting=False, filter_width=0, skip=1):
        """
        The kinetic energy function. It calculates the mean pressure of the
        simulation and optinally shows the kinetic energy in a graph, with
        an optional filter.

        Parameters
        ----------
        plotting : Boolean, optional
            Optional plots the kinetic energy as function of time. The default
            is False.
        filter_width : Integer, optional
            Filter width of the moving average filter. The default is 0, which
            results in no filtering.

        Returns
        -------
        energy : Float
            The kinetic energy of the simulation run in eV.

        """
        # Retrieve the kinetic energy from the dataframe.
        e_n = np.array(['total energy   ETOTAL',
                        'nose kinetic   EPS',
                        'nose potential ES',
                        'kinetic energy EKIN'])
        ener = self.energy[e_n[3]]
        energy = statistics(ener[0::skip])

        # plot if asked for
        if plotting is True:
            # filter if needed
            ener = filter(ener, filter_width)

            # set labels of plot
            i = re.findall(r'\d+', self.folder)[-1]
            string = ' run ' + i

            plt.figure('kinetic energy')
            plt.plot(self.t*1e12, ener, label=string)

        return energy

    def pot_energy(self, plotting=False, filter_width=0, skip=1):
        """
        The potential energy function. It calculates the mean pressure of the
        simulation and optinally shows the potential energy in a graph, with
        an optional filter.

        Parameters
        ----------
        plotting : Boolean, optional
            Optional plots the potential energy as function of time. The
            default is False.
        filter_width : Integer, optional
            Filter width of the moving average filter. The default is 0, which
            results in no filtering.

        Returns
        -------
        energy : Float
            The potential energy of the simulation run in eV.

        """
        # Retrieve the potential energ from the dataframe. This is done by
        # subtracting kin_E and the nose energies from the total energy
        e_n = np.array(['total energy   ETOTAL',
                        'nose kinetic   EPS',
                        'nose potential ES',
                        'kinetic energy EKIN'])
        ener = (self.energy[e_n[0]] - self.energy[e_n[1]]
                - self.energy[e_n[2]] - self.energy[e_n[3]])
        energy = statistics(ener[0::skip])

        # plot if asked for
        if plotting is True:
            # filter if needed
            ener = filter(ener, filter_width)

            # set labels of plot
            i = re.findall(r'\d+', self.folder)[-1]
            string = ' run ' + i

            plt.figure('potential energy')
            plt.plot(self.t*1e12, ener, label=string)

        return energy

    def tot_energy(self, plotting=False, filter_width=0, skip=1):
        """
        The total energy function. It calculates the mean pressure of the
        simulation and optinally shows the total energy in a graph, with
        an optional filter.

        Parameters
        ----------
        plotting : Boolean, optional
            Optional plots the total  energy as function of time. The default
            is False.
        filter_width : Integer, optional
            Filter width of the moving average filter. The default is 0, which
            results in no filtering.

        Returns
        -------
        energy : Float
            The total energy of the simulation run in eV.

        """
        # Retrieve the total energ from the dataframe. This is done by
        # subtracting nose energies from the total energy
        e_n = np.array(['total energy   ETOTAL',
                        'nose kinetic   EPS',
                        'nose potential ES',
                        'kinetic energy EKIN'])
        ener = self.energy[e_n[0]] - self.energy[e_n[1]] - self.energy[e_n[2]]
        energy = statistics(ener[0::skip])

        # plot if asked for
        if plotting is True:
            # filter if needed
            ener = filter(ener, filter_width)

            # set labels of plot
            i = re.findall(r'\d+', self.folder)[-1]
            string = ' run ' + i

            plt.figure('total energy')
            plt.plot(self.t*1e12, ener, label=string)

        return energy

    def temperature(self, plotting=False, filter_width=0, skip=1):
        """
        The temperature function. It calculates the mean temperature of the
        simulation and optinally shows the temperature in a graph, with
        an optional filter.

        Parameters
        ----------
        plotting : Boolean, optional
            Optional plots the temperature as function of time. The default
            is False.
        filter_width : Integer, optional
            Filter width of the moving average filter. The default is 0, which
            results in no filtering.

        Returns
        -------
        temperature : Float
            The temperature of the simulation run in eV.

        """
        # Retrieve the total energ from the dataframe. This is done by
        # subtracting nose energies from the total energy
        temp = self.energy['temperature    TEIN']
        temperature = statistics(temp[0::skip])

        # plot if asked for
        if plotting is True:
            # filter if needed
            temp = filter(temp, filter_width)

            # set labels of plot
            i = re.findall(r'\d+', self.folder)[-1]
            string = ' run ' + i

            plt.figure('temperature')
            plt.plot(self.t*1e12, temp, label=string)

        return temperature


def filter(signal, width):
    """
    Filter the signal with a moving average.

    Parameters
    ----------
    signal : array
        The array of which the moving average is needed.
    width : integer
        The width of the moving average filter.

    Returns
    -------
    signal : array
        smoothened result of same array size as input.

    """
    if width == 0:
        signal = signal
    else:
        signal = pd.DataFrame(signal)
        signal = signal.ewm(span=width).mean()
        signal = signal.to_numpy()
    return signal


def statistics(s):
    """
    Calculates the mean value and the error of this estimation.
    The error is compensated for the autocorrolation.

    Parameters
    ----------
    s : Array
        The data measured over time.

    Returns
    -------
    mean : float
        The mean value of the array.
    error : float
        the error estimation, expresed as standard deviation, compensated
        for autocorrolation.

    """
    # collect most important characterstics of array
    N = s.shape[0]  # size
    mean = s.mean()  # mean value
    var = np.var(s)  # variance of entire set
    # If no statistic behaviour exists
    if var == 0.0:
        mean, error, tao, g = mean, 0, 0, 0
    # Otherwise calculate the correct error estimation
    else:
        sp = s - mean  # deviation of the mean per index

        # Calculating the total corrolation
        corr = np.zeros(N)
        corr[0] = 1
        for n in range(1, N):
            corr[n] = np.sum(sp[n:]*sp[:-n]/(var*N))

        # To not fitt too long of a data set, the first time that the
        # corrolation drops under 0.1 is recorded and only the data before
        # that is fitted
        g = np.argmax(corr < 0.1)
        t = np.arange(2*g)
        tao = opt.curve_fit(lambda t, b: np.exp(-t/b),  t,
                            corr[:2*g], p0=(g))[0][0]
        error = np.sqrt(2*tao*s.var()/N)
    return np.array([mean, error])
