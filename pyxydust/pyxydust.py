import os, time
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as pl

from sedpy import observate
import dustmodel
import statutils
import datacube


class Pyxydust(object):

    doresid = False

    def __init__(self, rp):
        """Initialize.

        :param rp:
            Dictionary containing a number of important parameters.
        """
        self.rp = rp
        self.load_models()
        self.set_default_params()

    def load_models(self):
        """Load the Draine & Li basis models, initialize the grid to
        hold resampled models, and load the filters
        """
        # Draine and Li Basis
        self.dl07 = dustmodel.DraineLi()
        # object to hold the model grid
        self.dustgrid = dustmodel.SpecLibrary()
        # filter objects
        self.filterlist = observate.load_filters(self.rp['fnamelist'])

    def load_data(self):
        """Read the image cube and the uncertainty cube, apply
        distance modulus, and determine 'good' pixels
        """
        dat = datacube.load_image_cube(**self.rp)
        self.data_mag, self.data_magerr, self.rp['data_header'] = dat
        dm = 5.0 * np.log10(self.rp['dist']) + 25
        self.data_mag = np.where(self.data_mag != 0.0, self.data_mag - dm, 0.0)
        self.nx, self.ny = self.data_mag.shape[0], self.data_mag.shape[1]

        gg = np.where((self.data_mag < 0) & np.isfinite(self.data_mag), 1, 0)
        # restrict to detections in all bands
        self.goodpix = np.where(gg.sum(axis=2) == len(self.rp['imnamelist']))

    def setup_output(self):
        """Create arrays to store fit output for each pixel.
        """
        self.max_lnprob = np.zeros([self.nx, self.ny]) + float('NaN')
        try:
            self.outparnames = self.rp['outparnames'] + ['LDUST', 'MDUST']
        except (KeyError):
            self.outparnames = ['LDUST', 'MDUST']
        self.parval = {}
        for parn in self.outparnames:
            shape = [self.nx, self.ny, len(self.rp['percentiles']) + 1]
            self.parval[parn] = np.zeros(shape) + float('NaN')

        # self.doresid = self.rp.get('return_residuals', False)
        try:
            self.doresid = self.rp['return_residuals']
        except (KeyError):
            self.doresid = False
        if self.doresid is True:
            self.delta_best = np.zeros([self.nx, self.ny,
                                        len(self.filterlist)]) + float('NaN')

        self.dobestfitspectrum = self.rp.get('return_best_spectrum', False)
        if self.dobestfitspectrum:
            self.best_spectrum = np.zeros([self.nx, self.ny,
                                           len(self.dl07.wavelength)]) + float('NaN')

    def fit_image(self):
        """Fit every pixel in an image.
        """
        if hasattr(self, 'max_lnprob') is False:
            self.setup_output()
        start = time.time()
        for ipix in xrange(self.goodpix[0].shape[0]):
            iy, ix = self.goodpix[0][ipix], self.goodpix[1][ipix]
            self.fit_pixel(ix, iy)
        duration = time.time() - start
        print('Done all pixels in {0:.1f} seconds'.format(duration))

    def write_output(self):
        """Write stored fit information to FITS files.
        """
        header = self.rp['data_header']

        outfile = '{outname}_CHIBEST.fits'.format(**self.rp)
        pyfits.writeto(outfile, -2 * self.max_lnprob, header=header,
                       clobber=True)

        for i, parn in enumerate(self.outparnames):
            # header.set('BUNIT',unit[i])
            outfile = '{0}_{1}_bestfit.fits'.format(self.rp['outname'], parn)
            pyfits.writeto(outfile, self.parval[parn][:,:,-1],
                           header=header, clobber=True)
            for j, percent in enumerate(self.rp['percentiles']):
                outfile = '{0}_{1}_p{2:5.3f}.fits'.format(self.rp['outname'],
                                                          parn, percent)
                pyfits.writeto(outfile, self.parval[parn][:,:,j],
                               header=header, clobber=True)
        if self.doresid:
            for i, fname in enumerate(self.rp['fnamelist']):
                outfile = '{0}_{1}_{2}.fits'.format(self.rp['outname'],
                                                    fname, 'bestfit_residual')
                pyfits.writeto(outfile, self.delta_best[:,:,i],
                               header=header, clobber=True)
        if self.dobestfitspectrum:
            outfile = '{0}_bestfit_spectrum.fits'.format(self.rp['outname'])
            pyfits.writeto(outfile, self.best_spectrum, clobber=True)


class PyxydustGrid(Pyxydust):

    def initialize_grid(self, params=None):
        """Draw grid or library parameters from prior distributions
        and build the grid.
        """
        if params is not None:
            self.params = params

        parnames = self.params.keys()
        theta = np.zeros([self.rp['ngrid'], len(parnames)])
        for j, parn in enumerate(parnames):
            pmin, pmax = self.params[parn]['min'], self.params[parn]['max']
            n = self.rp['ngrid']
            theta[:,j] = np.random.uniform(pmin, pmax, n)
            if self.params[parn]['type'] == 'log':
                theta[:,j] = 10**theta[:,j]  # deal with uniform log priors

        start = time.time()
        self.dustgrid.set_pars(theta, parnames)
        dat = self.dl07.generateSEDs(self.dustgrid.pars,self.filterlist,
                                     wave_min=self.rp['wave_min'],
                                     wave_max=self.rp['wave_max'])
        self.dustgrid.sed, self.dustgrid.lbol, tmp = dat
        umin = self.dustgrid.pars['UMIN']
        umax = self.dustgrid.pars['UMAX']
        gamma = self.dustgrid.pars['GAMMA']
        ubar = self.dl07.ubar(umin, umax, gamma)
        self.dustgrid.add_par(ubar,'UBAR')
        duration = time.time() - start
        print('Model Grid built in {0:.1f} seconds'.format(duration))

    def fit_pixel(self, ix, iy, store=True, show_cdf=False):
        """
        Determine \chi^2 of every model for a given pixel, and store
        moments of the CDF for each parameter as well as the
        bestfitting model parameters.  Optionally store magnitude
        residuals from the best fit.
        """
        obs, err = self.data_mag[iy,ix,:], self.data_magerr[iy,ix,:]
        mask = np.where((obs < 0) & np.isfinite(obs), 1, 0)

        dat = statutils.lnprob_grid(self.dustgrid, obs, err, mask)
        lnprob, ltir, dustm, delta_mag = dat
        ind_isnum = np.where(np.isfinite(lnprob))[0]
        lnprob_isnum = lnprob[ind_isnum]
        ind_max = np.argmax(lnprob_isnum)

        # this should all go to a storage method
        self.max_lnprob[iy,ix] = np.max(lnprob_isnum)
        self.delta_best[iy,ix,:] = delta_mag[ind_isnum[ind_max],:]
        if self.dobestfitspectrum:
            spec = self.dl07.spectra_from_pars(self.dustgrid.pars[ind_isnum[ind_max]])
            self.best_spectrum[iy,ix,:] = (dustm[ind_isnum[ind_max]] * spec)

        for i, parn in enumerate(self.outparnames):
            if parn == 'LDUST':
                par = np.squeeze(ltir)[ind_isnum] * self.dl07.convert_to_lsun
            elif parn == 'MDUST':
                par = np.squeeze(dustm)[ind_isnum]
            else:
                par = np.squeeze(self.dustgrid.pars[parn])[ind_isnum]

            order = np.argsort(par)
            cdf = (np.cumsum(np.exp(lnprob_isnum[order])) /
                   np.sum(np.exp(lnprob_isnum)))
            ind_ptiles = np.searchsorted(cdf, self.rp['percentiles'])
            # should linear interpolate instead of average.
            self.parval[parn][iy,ix,:-1] = (par[order[ind_ptiles-1]] +
                                            par[order[ind_ptiles]]) / 2.0
            self.parval[parn][iy,ix,-1] = par[ind_max]

    def set_default_params(self):
        """Set the default model parameter properties.
        """
        # should be list of dicts or dict of lists?  no, dict of dicts!
        qpahmax = self.dl07.par_range(['QPAH'],
                                      inds=[self.dl07.delta_inds])[0][1]
        self.params = {}
        self.params['UMIN'] = {'min':np.log10(0.1), 'max':np.log10(25),
                               'type':'log'}
        self.params['UMAX'] = {'min':3, 'max':6, 'type':'log'}
        self.params['GAMMA'] = {'min':0, 'max':1.0, 'type':'linear'}
        self.params['QPAH'] = {'min':0.47, 'max':qpahmax, 'type':'log'}


class PyxydustMCMC(Pyxydust):
    """Use emcee to do MCMC sampling of the parameter space for a
    given pixel.

    Wildly unfinished/untested
    """

    def set_default_params(self, large_number=1e15):
        """Set the default model parameter ranges.
        """
        # should be list of dicts or dict of lists?  no, dict of dicts!
        qpahmax = self.dl07.par_range(['QPAH'],
                                      inds=[self.dl07.delta_inds])[0][1]
        self.params = {}
        self.params['UMIN'] = {'min': np.log10(0.1), 'max':np.log10(25),
                               'type':'log'}
        self.params['UMAX'] = {'min': 3, 'max':6, 'type':'log'}
        self.params['GAMMA'] = {'min': 0, 'max':1.0, 'type':'linear'}
        self.params['QPAH'] = {'min': 0.47, 'max':qpahmax, 'type':'log'}
        self.params['MDUST'] = {'min':0, 'max':large_number, 'type': 'linear'}

    def fit_pixel(self, ix, iy):
        obs, err = self.data_mag[ix,iy,:], self.data_magerr[ix,iy,:]
        obs_maggies = 10**(0 - obs / 2.5)
        obs_ivar = (obs_maggies * err / 1.086)**(-2)
        mask = np.where((obs < 0) & np.isfinite(obs), 1, 0)

        sampler = self.sample(obs_maggies, obs_ivar, mask)

    def sample(self,obs_maggies, obs_ivar, mask):
        initial = self.initial_proposal()

        # get a sampler, burn it in, and reset
        sampler = emcee.EnsembleSampler(self.rp['nwalkers'],
                                        self.rp['ndim'], self.lnprob,
                                        threads=nthreads,
                                        args=[obs_maggies,obs_ivar,mask])
        pos, prob, state, blob = sampler.run_mcmc(initial, self.rp['nburn'])
        sampler.reset()

        # cry havoc
        sampler.run_mcmc(np.array(pos),self.rp['nsteps'], rstate0=state)

        return sampler

    def initial_proposal(self):
        parnames = self.lnprob.lnprob_parnames
        theta = np.zeros(len(parnames))
        for j, parn in enumerate(parnames):
            theta[:,j] = np.random.uniform(self.params[parn]['min'],
                                           self.params[parn]['max'])
        return theta

    # def model(self, umin=umin, umax=umax, gamma=gamma, mdust=mdust, alpha=2):
    #    pass

    def lnprob(self, theta, obs_maggies, obs_ivar, mask):
        lnprob_parnames = ['UMIN', 'UMAX', 'GAMMA', 'QPAH', 'MDUST']
        # ugh.  need quick dict or struct_array from list/array
        # pardict = {lnprob_parnames theta}

        # prior bounds check
        ptest = []
        for i,par in enumerate(lnprob_parnames):
            ptest.append(pardict[par] >= self.params[par]['min'])
            ptest.append(pardict[par] <= self.params[par]['max'])
            if self.params[par]['type'] == 'log':
                pardict[par] = 10**pardict[par]

        if False in ptest:
            # set lnp to -infty if parameters out of prior bounds
            lnprob = -np.infty
            lbol = -1

        else:
            # model sed (in AB absolute mag) for these parameters
            sed, lbol = model(**pardict)
            sed_maggies = 10**(0 - sed / 2.5)

            # probability
            chi2 = ((sed_maggies - obs_maggies)**2) * obs_ivar
            inds = np.where(mask > 0)
            lnprob = -0.5 * chi2[inds].sum()

        return lnprob, [lbol]
