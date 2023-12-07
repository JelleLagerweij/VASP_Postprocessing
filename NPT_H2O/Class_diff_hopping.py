# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:58:09 2023

@author: Jelle
"""

import numpy as np
import pandas as pd
import freud


class Prot_Hop:
    def __init__(self, file, n_max):
        # Make flexible lateron and integrate as optional argument in class.
        self.r_bond_u = 1.1  # Angstrom
        self.r_bond_l = 1.1  # Angstrom
        self.n_OH = 1  # number of OH- particles in system.

        # Normal startup behaviour
        self.file = file
        self.n_max = n_max  # HAS TO BE TOTAL TIMESTEPS OF XDATCAR FILE
        self.i_O = 0
        self.i_H = 0
        self.shift = np.zeros((self.n_OH, 1, 3))
        self.n_list = 10  # number of Hydrogens in the neighbor list
        self.setting_properties()
    
    
    def setting_properties(self):
        df = pd.read_csv(self.file, skiprows=2, nrows=5, header=None,
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
        self.df = pd.read_csv(self.file, skiprows=skip, header=None,
                              delim_whitespace=True)
        # TODO: instead of pd.read() use method that can deal with larger
        # file sizes.
        self.find_O_i(0, startup=True)
    
    def read_XDATCAR(self, n, index=False, startup=False):
        """
        This function reads XDATCAR files and imports atomic positions

        Parameters
        ----------
        n : timestep
            DESCRIPTION.
        index : numpy array, optional
            if array, it will only read the atomic positions of given indexes.
            The default is False.
        startup : Boolean, optional
            If True it computes the number of atoms per specie and unitvectors.
            The default is False.

        Returns
        -------
        cord : TYPE
            DESCRIPTION.

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
        number = (d_sq < self.r_bond_u**2).sum(axis=0)
        O_i = np.where(number == 1) + self.N_H
        
        
        r_bond = self.r_bond_u
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
            dis = (x[O_i, :] - x[O_i_old, :] + self.L/2)%self.L - self.L/2
            real_loc = x[O_i_old, :] + dis
            self.shift += real_loc - x[O_i, :]
            self.x_O_i = x[O_i, :]
        
    def find_H_n(self, n):
        # Retrieve particle positions O_i and ALL Hydrogens
        x_O_i = self.read_XDATCAR(n, index=self.O_i)
        x_H = self.read_XDATCAR(n, index=self.H)

        # Compute particle distances^2 between O_i and all H including PBC.
        rel_r = (x_H - x_O_i + self.L/2) % self.L - self.L/2
        d_sq = np.einsum('ij, ij -> i', rel_r, rel_r)
        
        # Get index of the neighborlist Hydrogen
        self.H_n = np.sort(np.argpartition(d_sq, self.n_list)[:self.n_list])     
        self.n_list_c = 0  # reset counter for neighborlist updating

    def run_loop(self, N_neighbor=100):
        # Prep things to track
        self.O_i_stored = np.zeros(self.n_max)
        self.O_loc_stored = np.zeros((self.n_max, 3))
        self.O_loc_stored_bad = np.zeros((self.n_max, 3))
        # Run the big loop
        for j in range(self.n_max):
            self.x_O_i = self.read_XDATCAR(j, index=self.O_i)
            self.x_H = self.read_XDATCAR(j, index=self.H_n)
            rel_r = (self.x_H - self.x_O_i + self.L/2) % self.L - self.L/2
            d_sq = np.einsum('ij, ij -> i', rel_r, rel_r)
            n = np.count_nonzero(d_sq < self.r_bond_l**2)

            if n > 1:
                # New O_i particle to find
                self.find_O_i(j)
            
            if self.n_list_c > N_neighbor:
                # Update neighbor list ever so often
                self.find_H_n(j)
            
            self.O_i_stored[j] = self.O_i
            self.O_loc_stored[j, :] = self.x_O_i + self.shift
            self.n_list_c += 1
            
        return np.copy(self.O_i_stored), np.copy(self.O_loc_stored)
    
    def single_MSD(self):
        rel_r = (self.O_loc_stored - self.O_loc_stored[0, :])
        d_sq = np.einsum('ij, ij -> i', rel_r, rel_r)
        return d_sq
    
    def single_MSD_freud(self):
        # reshape the locations
        loc = np.reshape(self.O_loc_stored, (len(self.O_loc_stored), 1,
                                             len(self.O_loc_stored[1])))
        msd = freud.msd.MSD(mode='direct')
        
        # compute MSD
        calculation = msd.compute(loc)
        return calculation.msd

    def windowed_MSD_freud(self):
        # reshape the locations
        loc = np.reshape(self.O_loc_stored, (len(self.O_loc_stored), 1,
                                             len(self.O_loc_stored[1])))
        msd = freud.msd.MSD(mode='window')
        
        # compute MSD
        calculation = msd.compute(loc)
        return calculation.msd
