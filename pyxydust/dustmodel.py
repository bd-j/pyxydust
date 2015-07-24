# Python module to deal with dust emission models, primarily the
# Draine+Li models but others can be included

import os, glob
import numpy as np
import astropy.constants as constants

from sedpy import observate
from sedpy.modelgrid import *

pc = constants.pc.cgs.value
lsun = constants.L_sun.cgs.value


class DraineLi(SpecLibrary):
    """Class to store and operate on Draine & Li 2007 models
    """
    pyxydustdir, f = os.path.split(__file__)

    # Spectra in this file are in erg/s/cm^2/AA/M_sun of dust at a
    # distance of 10pc
    model_file = pyxydustdir + '/data/models/DL07.fits'
    # From model fluxes to Lsun per Msun of dust
    flux_unit = 'erg/s/cm^2/AA of 1solar mass at 10pc'
    convert_to_lsun = 10**(np.log10(4 * np.pi) + 2 * np.log10(pc * 10)) / lsun

    def __init__(self, modfile=None):
        if modfile is not(None):
            self.model_file = modfile
        self.read_from_fits_library(self.model_file)

    def spectra_from_pars(self, parstruct):
        """Wrapper on generate_spectrum that parses a pars structured
        array for the required parameters.
        """
        alpha = 2
        mdust = 1
        return self.generate_spectrum(parstruct['UMIN'], parstruct['UMAX'],
                                      parstruct['GAMMA'], parstruct['QPAH'],
                                      alpha, mdust=mdust)

    def generate_spectrum(self, umin, umax, gamma, qpah, alpha,
                          mdust=1.0, **extras):
        """Returns the IR SED of a model(s) given the model
        parameters.

        :param umin:
            The minimum of the dust heating starlight intensity
            distribution. Scalar or ndarray of shape (nobj).  In units
            of the MMP83 value.

        :param umax:
            The maximum of the dust heating starlight intensity
            distribution. Scalar or ndarray of shape (nobj). In units
            of the MMP83 value.

        :param gamma:
            Mass fraction of dust exposed to u > umin. Scalar or
            ndarray of shape (nobj).

        :param qpah:
            PAH abundance, in %. Scalar or ndarray of shape (nobj).

        :param alpha:
            Power law slope, du/dm ~ U**(-alpha).  Currently
            unsupported.

        :param mdust: (default: 1.0)
            The dust mass in units of M_sun. Scalar or ndarray of
            shape (nobj).

        :returns spectrum:
           Spectra interpolated to the requested parameter
           values. Output units are erg/s/cm^2/AA at a distance of
           10pc. ndarray of shape (nobj,nwave)
        """
        umin, umax, gamma = np.atleast_1d(umin, umax, gamma)
        qpah, alpha, mdust = np.atleast_1d(qpah, alpha, mdust)
        check = [umin.shape[0] == umax.shape[0],
                 umin.shape[0] == qpah.shape[0]]
        if False in check:
            raise ValueError("Parameter array sizes do not match")

        delta_spec = self.interpolate_to_pars(np.array([umin, qpah]).T,
                                              parnames=['UMIN', 'QPAH'],
                                              subinds=self.delta_inds,
                                              force_triangulation=True)

        pdr_spec = np.zeros(self.wavelength.shape[0])
        if np.any(gamma > 0):
            pdr_spec = self.interpolate_to_pars(np.array([umin, umax, qpah]).T,
                                                parnames=['UMIN', 'UMAX', 'QPAH'],
                                                subinds=self.pdr_inds,
                                                force_triangulation=True)

        total_spec = (1.0 - gamma) * delta_spec.T + gamma * pdr_spec.T
        return (mdust * total_spec).T

    def ubar(self, umin, umax, gamma, alpha=2):
        """Calculate the dust-mass weighted average starlight heating
        intensity.

        :param umin:
            The minimum of the dust heating starlight intensity
            distribution. Scalar or ndarray of shape (nobj).  In units
            of the MMP83 value.

        :param umax:
            The maximum of the dust heating starlight intensity
            distribution. Scalar or ndarray of shape (nobj). In units
            of the MMP83 value.

        :param gamma:
            Mass fraction of dust exposed to u > umin. Scalar or
            ndarray of shape (nobj).

        :param alpha:
            Power law slope, du/dm ~ U**(-alpha).  Currently
            unsupported.

        :returns ubar:
            the dust-mass weighted average starlight heating
            intensity, scalar or ndarray of shape (nobj).
        """
        if np.any(alpha != 2):
            raise ValueError('only alpha=2 is currently allowed')
        ubar_pdr = np.log(umax / umin) / (1 / umin - 1 / umax)
        return (1 - gamma) * umin + gamma * ubar_pdr

    def read_from_fits_library(self, filename, grain='MW3.1'):
        """Read in the dust SED library of Draine & Li 2007, stored as
        a fits binary table with spectral units of erg/s/cm^2/AA/M_sun
        of dust at a distance of 10pc. Produce index arrays to access
        delta-function and alpha=2 power law dust heating intensity
        distribtions.
        """
        if os.path.isfile(filename) is False:
            raise ValueError('File does not exist: %s', filename)
            return 0.

        parnames = ['UMIN', 'UMAX', 'QPAH', 'GRAIN']
        dat = self.read_model_from_fitsbinary(filename, parnames)
        self.wavelength, self.spectra, self.pars = dat

        # Record the indices of models corresponding to delt-functions
        # and 'pdr' U distributions
        dinds = ((self.pars['UMAX'] == self.pars['UMIN']) &
                 (self.pars['GRAIN'] == grain))
        self.delta_inds = np.squeeze(np.array(np.where(dinds)))
        pinds = ((self.pars['UMAX'] > self.pars['UMIN']) &
                 (self.pars['GRAIN'] == grain))
        self.pdr_inds = np.squeeze(np.array(np.where(pinds)))

    def library_from_dustEM():
        """If python bindings to DustEM ever happen, you could call
        it from here to fill the model recarray with a denser grid
        than Draine provides.
        """
        pass


class ModifiedBB(SpecLibrary):
    """Class to work with modified blackbodies a.k.a. greybodies
    """

    def __init__(wavelengths):
        self.wavelength = wavelengths

    def spectraFromPars(self, pars):
        """Wrapper on generateSpectrum that parses a pars structured
        array for the required parameters.
        """
        return self.generateSpectrum(pars['T_DUST'], pars['BETA'],
                                     pars['M_DUST'])

    def generate_spectrum(self, T_dust, beta, M_dust,
                          kappa_lambda0=(1.92, 350e4)):
        """Method to return the spectrum given a list or array of
        temperatures, beta, and dust masses.  Also, the normalization
        of the emissivity curve can be given as a tuple in units of
        cm^2/g and AA, default = (1.92, 350e4).  Ouput units are
        erg/s/AA/cm^2 at 10pc.  should find a way to make this an
        array
        """
        spec = np.zeros([self.wavelength.shape[0]])
        # Should vectorize
        for i, T in enumerate(T_dust):
            term = (kappa_lambda0[1] / self.wavelength)**beta[i]
            spec += M_dust[i] * self.planck(T) * term * kappa_lambda0[0]
        return spec

    def planck(self, T):
        """Return planck function B_lambda(cgs) for a given T
        """
        wave = self.wavelength*1e8  # Convert from AA to cm
        # Return B_lambda in erg/s/AA
        conv = 2 * hplank * c_cgs**2 / wave**5 * 1e8
        denom = (np.exp(hplanck * c_cgs / (kboltz * T)) - 1)
        return conv * 1 / denom
