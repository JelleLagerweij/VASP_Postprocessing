# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:58:09 2023

@author: Jelle
"""

import numpy as np
import pandas as pd
import scipy as sp
import re
import scipy.constants as co

import matplotlib.pyplot as plt
import freud


class Prot_Hop:
    """Postprocesses class for NVT VASP simulations of aqueous KOH."""

    def __init__(self, folder, n_max, posfile='XDATCAR', r_bond=1.1,
                 n_OH=1, dt=1e-15):
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
        n_max : integer
            Maximum number of timesteps stored.
        posfile : string, optional
            The name of the positional output file. It has to be of type
            XDATCAR VASP 5+. The default is 'XDATCAR'.
        presfile : string, optional
            The name of the pressure tensor output file. The order of the
            pressure tensor components has to be XX, YY, ZZ, XY, YZ, ZX. The
            default is 'press_tensor.dat'.
        rdffile : string, optional
            The name of the file which holds the RDF outputs. It should be in
            the style of VASP: "Total, 1-1, 1-2, 1-3, 2-2, 2-3, 3-3".
        r_bond : float, optional
            The typical bond length of the OH bond in OH- and H2O. The default
            is 1.1.
        n_OH : integer, optional
            Number of OH- in solution. The default is 1.
        dt : float, optional
            The timestepsize in fs of the calculation. The default is 1e-15s.

        Returns
        -------
        None.

        """
        # Make flexible lateron and integrate as optional argument in class.
        self.r_bond = r_bond  # Angstrom
        self.n_OH = n_OH  # number of OH- particles in system.
        self.dt = dt

        # Normal startup behaviour
        self.folder = folder + '/'
        self.posfile = folder + '/' + posfile
        self.T_ave = 325
        self.n_max = n_max  # HAS TO BE TOTAL TIMESTEPS OF XDATCAR FILE
        self.i_O = 0
        self.i_H = 0
        self.shift = np.zeros((self.n_OH, 1, 3))
        self.n_list = 100  # number of Hydrogens in the neighbor list
        self.t = np.arange(self.n_max)*self.dt
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
        # The Position Section
        df = pd.read_csv(self.posfile, skiprows=2, nrows=5, header=None,
                         delim_whitespace=True)
        self.species = df.iloc[3].values.tolist()
        self.N = np.array(df.iloc[4]).astype(int)
        self.basis = np.array(df.iloc[0:3]).astype(float)
        self.L = np.sqrt(np.dot(self.basis[0, :], self.basis[0, :]))

        self.N_tot = sum(self.N)
        self.N_H = self.N[self.species.index('H')]
        self.N_O = self.N[self.species.index('O')]
        self.N_K = self.N[self.species.index('K')]

        self.H = np.arange(self.N_H)
        self.O = np.arange(self.N_H, self.N_H + self.N_O)
        self.K = np.arange(self.N_H + self.N_O,
                           self.N_H + self.N_O + self.N_K)

        # only read the parts of the file you need load total dataframe.
        a = np.arange(self.n_max)*(self.N_tot+1) + 7
        skip = np.arange(self.n_max + 7)
        skip[7:] = a
        self.df = pd.read_csv(self.posfile, skiprows=skip, header=None,
                              delim_whitespace=True)
        # TODO: instead of pd.read() use method that can deal with larger
        # file sizes.
        self.find_O_i(0, startup=True)

    def read_XDATCAR(self, n, index=False, startup=False):
        """
        Read specific time and atom index from dataframe with all positions.

        Parameters
        ----------
        n : integer
            The timestep that is assesed.
        index : numpy array of integers, optional
            If array, it will only read the atomic positions of given indexes.
            The default is False.
        startup : Boolean, optional
            If True it computes the number of atoms per specie and unitvectors.
            The default is False.

        Returns
        -------
        cord : numpy array
            The (3D) atomic positions of the particles index at timestep n. If
            the index is False, all particle positions at timestep n are
            returned.

        """
        if index is False:
            # If there is no index take all atoms of specific timestep
            cord = self.df.iloc[n*self.N_tot:(n+1)*self.N_tot]
        else:
            # if there is an index, only take those specific atoms of specific
            # timestep.
            ind = index + self.N_tot*n
            cord = self.df.iloc[ind]
        return cord.to_numpy()*self.L

    def find_O_i(self, n, startup=False):
        """
        Find the index of the Oxygen beloging to the OH-.

        This function searches for the index of the Oxygen belonging to the OH-
        particles. It automatically creates a neighbor list as well as an array
        which holds the unwraped location of the real OH particle.

        Parameters
        ----------
        n : integer
            Timestep of the assesment.
        startup : Boolean, optional
            If NOT false, the unwraped location is set to 0. The default is
            False.

        Returns
        -------
        None.

        """
        # Retrieve ALL particle positions.
        x = self.read_XDATCAR(n)
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
        x_O_i = self.read_XDATCAR(n, index=self.O_i)
        x_H = self.read_XDATCAR(n, index=self.H)

        # Compute particle distances^2 between O_i and all H including PBC.
        rel_r = (x_H - x_O_i + self.L/2) % self.L - self.L/2
        d_sq = np.einsum('ij, ij -> i', rel_r, rel_r)

        # Get index of the neighborlist Hydrogen
        self.H_n = np.sort(np.argpartition(d_sq, self.n_list)[:self.n_list])
        self.n_list_c = 0  # reset counter for neighborlist updating

    def track_OH(self, N_neighbor=100):
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
            self.x_O_i = self.read_XDATCAR(j, index=self.O_i)
            self.x_H = self.read_XDATCAR(j, index=self.H_n)
            rel_r = (self.x_H - self.x_O_i + self.L/2) % self.L - self.L/2
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

        return counter/(self.n_max*self.n_OH), np.copy(self.O_i_stored), np.copy(self.O_loc_stored)

    def track_K(self):
        self.K_loc_stored = np.zeros((self.n_max, self.n_OH, 3))
        for j in range(self.n_max):
            self.K_loc_stored[j, :, :] = self.read_XDATCAR(j, index=self.K)
        return np.copy(self.K_loc_stored)

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

    def windowed_MSD(self, loc_array):
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
        return calculation.msd

    def viscosity(self, presfile='press_tensor.dat', plotting=0,
                  cubicspline=False, padding=0):
        self.presfile = self.folder + presfile

        # The pressure section
        df = pd.read_csv(self.presfile, delim_whitespace=True, header=None,
                         usecols=[4, 5, 6])
        press_off_diag = df.to_numpy()*1e8

        data = np.sum(press_off_diag, axis=1)[padding:]
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
            i = re.findall(r'\d+', self.folder)[0]
            string = ' run ' + i
            plt.figure('pressure autocorrelation')
            plt.plot(t, acorr, label=string)

        integral = sp.integrate.simpson(acorr, x=t)
        factor = np.power(self.L*1e-10, 3)/(co.k*self.T_ave)  # V/(kBT)
        visc = integral*factor
        return visc

    def diffusion(self, MSD_in, N_specie, t=False, m=False, plotting=False):
        # Settings for the margins and fit method.
        margin = 0.005  # cut away range at left and right side
        Minc = 4  # minimum number of points included in the fit
        Mmax = 20  # maximum number of points included in the fit
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

    def pressure(self, presfile='press_tensor.dat', plotting=False, 
                 move_ave=False):
        self.presfile = self.folder + presfile
        df = pd.read_csv(self.presfile, delim_whitespace=True, header=None,
                         usecols=[1, 2, 3])
        press = df.to_numpy()*1e3
        pdata = np.mean(press, axis=1)
        pressure = np.mean(pdata)

        if plotting is True:
            if move_ave is not False:
                pdata = pd.DataFrame(pdata)
                pdata = pdata.ewm(span=move_ave).mean()
                pdata = pdata.to_numpy()

            i = re.findall(r'\d+', self.folder)[0]
            string = ' run ' + i

            plt.figure('pressure')
            plt.plot(self.t, pdata, label=string)
        return pressure

    def rdf(self, rdffile='PCDAT', cubicspline=False, plotting=False):
        self.rdffile = self.folder + rdffile
        with open(self.rdffile) as fp:
            n = int(fp.readlines()[6])
        with open(self.rdffile) as fp:
            dr = float(fp.readlines()[8])

        r = np.arange(n*dr, step=dr)
        rdf_Data = pd.read_table(self.rdffile, header=None,
                                 delim_whitespace=True, skiprows=12)

        rdf_HH = rdf_Data[1].to_numpy()
        rdf_OO = rdf_Data[4].to_numpy()
        rdf_KK = rdf_Data[6].to_numpy()
        rdf_OH = rdf_Data[2].to_numpy()
        rdf_OK = rdf_Data[5].to_numpy()

        if cubicspline is not False:
            r_small = np.arange(r[0], r[-1], step=dr/cubicspline)

            spl_HH = sp.interpolate.CubicSpline(r, rdf_HH)
            spl_OO = sp.interpolate.CubicSpline(r, rdf_OO)
            spl_KK = sp.interpolate.CubicSpline(r, rdf_KK)
            spl_OH = sp.interpolate.CubicSpline(r, rdf_OH)
            spl_OK = sp.interpolate.CubicSpline(r, rdf_OK)

            rdf_HH = spl_HH(r_small)
            rdf_OO = spl_OO(r_small)
            rdf_KK = spl_KK(r_small)
            rdf_OH = spl_OH(r_small)
            rdf_OK = spl_OK(r_small)
            r = r_small

        if plotting is True:
            i = re.findall(r'\d+', self.folder)[0]
            string = ' run ' + i
            plt.figure('RDF H-H')
            plt.plot(r*1e10, rdf_HH, label=string)

            plt.figure('RDF O-O')
            plt.plot(r*1e10, rdf_OO, label=string)

            plt.figure('RDF K-K')
            plt.plot(r*1e10, rdf_KK, label=string)

            plt.figure('RDF O-H')
            plt.plot(r*1e10, rdf_OH, label=string)

            plt.figure('RDF O-K')
            plt.plot(r*1e10, rdf_OK, label=string)
        return np.array([r, rdf_HH, rdf_OO, rdf_KK, rdf_OH, rdf_OK])

    def bayes_error(self, beefile='BEEF.dat', plotting=False, move_ave=False):
        self.beefile = self.folder + beefile

        df = pd.read_table(self.beefile, header=None, skiprows=14,
                           delim_whitespace=True, usecols=[2, 3, 4, 5, 6, 7])
        bee = df.to_numpy()

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

            i = re.findall(r'\d+', self.folder)[0]
            string = ' run ' + i
            plt.figure('BEEF')
            plt.plot(self.t, beef_max, label='Max BEEF' + string)
            plt.plot(self.t, beef_ave, label='Ave BEEF' + string)
            plt.plot(self.t, beef_tresh, label='Treshold BEEF' + string)

            plt.figure('BEES')
            plt.plot(self.t, bees_max, label='Max BEES' + string)
            plt.plot(self.t, bees_ave, label='Ave BEES' + string)

        return beef_max.mean(), bees_max.mean()

    def kin_energy(self, ekinfile='EK.dat', plotting=False, move_ave=False):
        self.ekinfile = self.folder + ekinfile
        df = pd.read_csv(self.ekinfile, delim_whitespace=True, header=None)
        ekin = df.to_numpy()*co.eV

        if plotting is True:
            if move_ave is not False:
                edata = pd.DataFrame(ekin)
                edata = edata.ewm(span=move_ave).mean()
                edata = edata.to_numpy()

            i = re.findall(r'\d+', self.folder)[0]
            string = ' run ' + i

            plt.figure('kinetic energy')
            plt.plot(self.t, edata, label=string)
        return np.mean(ekin)

    def pot_energy(self, epotfile='F.dat', plotting=False, move_ave=False):
        self.epotfile = self.folder + epotfile
        df = pd.read_csv(self.epotfile, delim_whitespace=True, header=None)
        epot = df.to_numpy()*co.eV

        if plotting is True:
            if move_ave is not False:
                edata = pd.DataFrame(epot)
                edata = edata.ewm(span=move_ave).mean()
                edata = edata.to_numpy()

            i = re.findall(r'\d+', self.folder)[0]
            string = ' run ' + i

            plt.figure('potential energy')
            plt.plot(self.t, edata, label=string)
        return np.mean(epot)

    def tot_energy(self, etotfile='E.dat', plotting=False, move_ave=False):
        self.etotfile = self.folder + etotfile
        df = pd.read_csv(self.etotfile, delim_whitespace=True, header=None)
        etot = df.to_numpy()*co.eV

        if plotting is True:
            if move_ave is not False:
                edata = pd.DataFrame(etot)
                edata = edata.ewm(span=move_ave).mean()
                edata = edata.to_numpy()

            i = re.findall(r'\d+', self.folder)[0]
            string = ' run ' + i

            plt.figure('total energy')
            plt.plot(self.t, edata, label=string)
        return np.mean(etot)

    def temperature(self, tempfile='Temperature.dat', plotting=False,
                    move_ave=False):
        self.tempfile = self.folder + tempfile
        df = pd.read_csv(self.tempfile, delim_whitespace=True, header=None)
        temp = df.to_numpy()

        if plotting is True:
            if move_ave is not False:
                tdata = pd.DataFrame(temp)
                tdata = tdata.ewm(span=move_ave).mean()
                tdata = tdata.to_numpy()

            i = re.findall(r'\d+', self.folder)[0]
            string = ' run ' + i

            plt.figure('temperature')
            plt.plot(self.t, tdata, label=string)
        return np.mean(temp)
