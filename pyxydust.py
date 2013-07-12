import os, time
import numpy as np
import pyfits
import dustmodel
import statutils
import datacube


class Pyxydust(object):

    def __init__(self, rp):
        self.rp = rp
        self.set_default_params()
        self.load_models()

    def load_models(self):
        self.dl07 = dustgrid.DraineLi()  #Draine and Li Basis
        self.dustgrid = dustgrid.SpecLibrary() #object to hold the model grid
        self.filterlist = observate.load_filters(self.rp['fnamelist']) #filter objects
        
    def load_data(self):
        self.data_mag, self.data_magerr, self.rp['data_header'] = datacube.load_image_cube(**self.rp)
        dm = 5.0*np.log10(self.rp['dist'])+25
        self.data_mag = np.where(self.data_mag != 0.0, self.data_mag-dm, 0.0)
        self.nx, self.ny = self.data_mag.shape[0], self.data_mag.shape[1]

        gg = np.where((self.data_mag < 0) & np.isfinite(self.data_mag),1,0)
        self.goodpix = np.where(gg.sum(axis = 2) == len(self.rp['imnamelist'])) #restrict to detections in all bands

    def write_output(self):

        header = self.rp['data_header']

        outfile= '{outname}_CHIBEST.fits'.format(**self.rp)
        pyfits.writeto(outfile,self.max_lnprob*(-2),header=header,clobber=True)

        for i, parn in enumerate(self.outparnames):
            #    header.set('BUNIT',unit[i])
            outfile= '{0}_{1}_bestfit.fits'.format(self.rp['outname'],parn)
            pyfits.writeto(outfile,self.parval[parn][:,:,-1],header=header,clobber=True)
            for j, percent in enumerate(self.rp['precentiles']):
                outfile= '{0}_{1}_p{2:5.3f}.fits'.format(self.rp['outname'],parn,percent) 
                pyfits.writeto(outfile,self.parval[parn][:,:,j],header=header,clobber=True)


class PyxydustGrid(Pyxydust):

    def initialize_grid(self):
        theta = np.zeros([self.rp['ngrid'],npar])
        for j, parn in enumerate(parnames) :
            theta[:,j] = np.random.uniform(self.params[parn]['min'],self.params[parn]['max'],self.rp['ngrid'])
            if self.params[parn]['type'] == 'log':
                theta[:,j]=10**theta[:,j] #deal with uniform log priors

        start = time.time()
        self.dustgrid.set_pars(theta,parnames)
        self.dustgrid.seds, self.dustgrid.lbol, tmp = self.dl07.generateSEDs(self.dustgrid.pars,self.filterlist,
                                                                             wave_min=self.rp['wave_min'],
                                                                             wave_max=self.rp['wave_max'])
        ubar = dl07.ubar(dustgrid.pars['UMIN'],dustgrid.pars['UMAX'],dustgrid.pars['GAMMA'])
        self.dustgrid.add_par(ubar,'UBAR')
        duration=time.time()-start
        print('Model Grid built in {0:.1f} seconds'.format(duration))

    def fit_image(self):
        start = time.time()
        for ipix in xrange(self.goodpix[0].shape[0]):
            iy, ix  = self.goodpix[0][ipix], self.goodpix[1][ipix]
            self.fit_pixel(ix,iy)
        duration =  time.time()-start
        print('Done all pixels in {0:.1f} seconds'.format(duration) )

    def fit_pixel(self, store = True, show_cdf = False):
        obs, err = self.data_mag[iy,ix,:], self.data_magerr[iy,ix,:]
        mask = np.where((obs < 0) & np.isfinite(obs), 1, 0)
    
        lnprob , ltir, dustm, delta_mag = statutils.lnprob_grid(self.dustgrid, obs, err, mask)
        ind_isnum = np.where(np.isfinite(lnprob))[0]
        lnprob_isnum = lnprob[ind_isnum]

        #this should all go to a storage method
        self.max_lnprob[iy,ix] = np.max(lnprob_isnum)
        self.delta_image[iy,ix,:] = delta_mag
        for i, parn in enumerate(self.outparnames):
            if parn == 'LDUST':
                par = np.squeeze(ltir)[ind_isnum]
            elif parn == 'MDUST':
                par = np.squeeze(dustm)[ind_isnum]
            else:
                par = np.squeeze(self.dustgrid.pars[parn])[ind_isnum]
                
            order = np.argsort(par)
            cdf = np.cumsum(np.exp(lnprob_isnum[order])) / np.sum(np.exp(lnprob_isnum))
            ind_ptiles= np.searchsorted(cdf,self.rp['percentiles']) 
            ind_max=np.argmax(lnprob_isnum)
            self.parval[parn][iy,ix,:-1] = (par[order[ind_ptiles-1]] +par[order[ind_ptiles]])/2.0 # should linear interpolate instead of average.
            self.parval[parn][iy,ix,-1] = par[ind_max]        

    def setup_output(self):
        self.max_lnprob = np.zeros([self.nx,self.ny])+float('NaN')
        try:
            self.outparnames = self.rp['outparnames']+['LDUST','MDUST']
        except (KeyError):
            self.outparnames = ['LDUST','MDUST']
        self.parval ={}
        for parn in self.outparnames:
            self.parval[parn] = np.zeros([self.nx,self.ny,len(self.rp['percentiles'])+1])+float('NaN')
        
        if self.doresid is True:
            self.delta_best = np.zeros([self.nx,self.ny,ln(self.filterlist)])+float('NaN')

    def set_default_params(self):
        #should be list of dicts or dict of lists?  no, dict of dicts!
        qpahmax = self.dl07.par_range(['QPAH'], inds = [dl07.delta_inds])[0][1]
        self.params = {}
        self.params['UMIN'] = {'min': np.log10(0.1), 'max':np.log10(25), 'type':'log'}
        self.params['UMAX'] = {'min': np.log10(1e4), 'max':1e6, 'type':'log'}
        self.params['GAMMA'] = {'min': 1e-4, 'max':0, 'type':'log'}
        self.params['QPAH'] = {'min': 0.47, 'max':qpahmax, 'type':'log'}
        pass
