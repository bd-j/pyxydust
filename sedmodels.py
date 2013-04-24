#Python module to deal with dust emission models,
#primarily the Draine+Li models but others can be included

import pyfits
import numpy as np
import scipy.spatial
import os
import astropy.constants as constants

import observate

lsun = constants.L_sun.cgs.value
pc = constants.pc.cgs.value
hplanck = constants.h.cgs.value
c_cgs = constants.c.cgs.value
kboltz = constants.k_B.cgs.value

class ModelGrid(object):
    """Class to deal with grids of SED models.  primary attributes are
    pars: a structured array of parameter values (ngrid)
    sed:  a numpy array of broadband seds (ngrid,nfilter)
    filter_names: a list of filter names corresponding to the nfilter axis of sed
    lbol: a numpy array of bolometric luminosities (ngrid)
    spectra: a numpy array of spectra  (ngrid, nwave) optional
    """

    maxmod=1e4

    def __init__(self,pars=None):
        ngrid = 0
        pars = None
        sed = None
        if pars is not None :
            self.setParameters(pars)                
    
    def generateSEDs(self,Library, filterlist, pars = None, wave_min = 90, wave_max = 1e7, keepspec = False):
	maxmod=1e4
        if pars is not None :
            self.setParameters(pars)
        self.filter_names = [f.name for f in filterlist]

	#split big model grids to avoid memory constraints
	self.sed = np.zeros([self.ngrid,len(self.filter_names)],dtype=float)
	self.lbol = np.zeros(self.ngrid,dtype=float)
	i=1
	while (i*maxmod <= self.ngrid):
	    s1, s2 = (i-1)*maxmod, np.min([i*maxmod-1,self.ngrid])
	    spec = Library.spectraFromPars(self.pars[s1:s2])
	    self.sed[s1:s2,:] = observate.getSED(Library.wavelength,spec,filterlist)
	    self.lbol[s1:s2] = Library.convert_to_lsun*observate.Lbol(Library.wavelength,spec,wave_min,wave_max)
	    i+=1
	#spec = Library.spectraFromPars(self.pars)
	#self.sed = observate.getSED(Library.wavelength,spec,filterlist)
	#self.lbol = Library.convert_to_lsun*observate.Lbol(Library.wavelength,spec,wave_min,wave_max)

        if keepspec : self.spectra=spec

    def parNames(self):
        return self.pars.dtype.names

    def parRange(self, parname, inds=None):
        prange=[]
        for p in parname:
            range_list = [ np.min(self.pars[inds][p]),np.max(self.pars[inds][p]) ]
            prange.append(range_list)
        return prange

    def addParameter(self,value,name, dtype='<f8'):
        pass

    def setPars(self, pars,parnames):
        self.pars = self.arrayToStruct(pars,parnames)
        self.ngrid = self.pars.shape[0]

    def arrayToStruct(self, values,fieldnames, types=['<f8']):
        """turn a numpy array of floats into a structurd array. fieldnames can be a list or
        string array of parameter names with length nfield.
        Assumes pars is a numpy array of shape (nobj,nfield)
        """
        values=np.atleast_2d(values)
        if values.shape[-1] != len(fieldnames):
            if values.shape[0] == len(fieldnames):
                values=values.T
            else:
                raise ValueError('models.ModelGrid.arrayToStruct: array and fieldnames do not have consistent shapes!')
        nobj=values.shape[0]
        
        #set up the list of tuples describing the fields.  Assume each parameter is a float
        fieldtuple=[]
        for i,f in enumerate(fieldnames):
            if len(types) > 1 :
                tt =types[i]
            else: tt=types[0]
            fieldtuple.append((f,tt))
        #create the dtype and structured array                    
        dt=np.dtype(fieldtuple)
        struct=np.zeros(nobj,dtype=dt)
        for i,f in enumerate(fieldnames):
            struct[f]=values[...,i]
        return struct


class SpecLibrary(ModelGrid):
    """Class to operate on spectral libraries. Methods are provided to interpolate the
    available model spectra (stored as a structured array and a spectral array) to a certain
    set of parameters.  Subclasses are used to return the actual model spectrum given
    a set of model parameters.
    """
    
    flux_unit = 'erg/s/cm^2/AA of 1solar mass at 10pc' 
        
    def __init__(self):
        pass
        
    def interpolateTo(self, target_points, parnames=None, itype='dt',subinds=None ):
        """Method to obtain the model spectrum for a given set of parameter values via
        interpolation of the model grid. The interpolation weights are determined
        from barycenters of a Delaunay triangulation or nLinear interpolation.
        
        The input is an array of target model parameters, optionally a string list of the
        fields containing the corresponding library model parameters
            target_points - ndarray of shape (ntarg,ndim) of ndim desired model parameters.
                            Can also be a structured array with fields named for the model parameters
            subinds - ndarray of indices of the model library structure to use in interpolation.
                      allows for only portions of the library to be used
            parnames  - string list of the names of the model library parameters
        The output has shape (ntarg, nwave)
        """

        #Note: should (but doesn't yet) allow for grid (and point) to be scaled in any or
        #all dimensions.  The DT is *not* invariant under scaling.
        #Note: should (but doesn't yet) check that point is in the grid
        #deal with recarray input
        
        if parnames is None:
            parnames = target_points.dtype.names #if the target point(s) is already a struct array use the field names
            targets = np.array(target_points.tolist())
        else:
            targets = target_points
        targets=np.atleast_2d(targets)
            
        #pull the grid points out of the model record data and make an (nmodel,ndim) array of
        #model grid parameter values.  need to loop to make sure order is correct
        model_points=[]
        for pname in parnames:
            model_points.append(np.squeeze(self.pars[subinds][pname]))
        model_points = np.array(model_points).transpose() #(nmod, ndim)
        
        inds, weights = self.weightsDT(model_points,targets)
        
        #weight using broadcasting, then sum the weighted spectra and return (nwave,ntarg)
        return (( weights* (self.spectra[subinds[inds]].transpose(2,0,1)) ).sum(axis=2)).T


    def weightsDT(self,model_points,target_points):
        """ The interpolation weights are determined from barycenter coordinates
        of the vertices of the enclosing Delaunay triangulation. This allows for
        the use of irregular Nd grids. see also weightsLinear.
            model_points - array of shape (nmod, ndim)
            target_points - array of shape (ntarg,ndim)
            output inds and weights - arrays of shape (npts,ndim+1)
        """
        
        ndim = target_points.shape[-1]
        #delaunay triangulate and find the encompassing (hyper)triangle(s) for the desired point
        dtri = scipy.spatial.Delaunay(model_points)
        #output triangle_inds is an (ntarg) array of simplex indices
        triangle_inds = dtri.find_simplex(target_points)
        #and get model indices of (hyper)triangle vertices. inds has shape (ntarg,ndim+1)
        inds = dtri.vertices[triangle_inds,:]
        #get the barycenter coordinates through matrix multiplication and dimensional juggling
        bary = np.dot( dtri.transform[triangle_inds,:ndim,:ndim],
                       (target_points-dtri.transform[triangle_inds,ndim,:]).reshape(-1,ndim,1) )
        oned = np.arange(triangle_inds.shape[0])
        bary = np.atleast_2d(np.squeeze(bary[oned,:,oned,:])) #ok.  in np 1.7 can add an axis to squeeze
        last = 1-bary.sum(axis=-1) #the last bary coordinate is 1-sum of the other coordinates
        weights = np.hstack((bary,last[:,np.newaxis]))

        #loop implementation of the above
        #npts = triangle_inds.shape[0]
        #bary = np.zeros([npts,ndim+1])
        #for i in xrange(npts):
        #    bary[i,:-1]= np.dot( dtri.transform[triangle_inds[0],:ndim,:ndim],
        #                       (target_points-dtri.transform[triangle_inds[i],ndim,:])
            
        return inds, weights 
               
        
class DraineLi(SpecLibrary):
    """DraineLi:: Class to store and operate on Draine & Li 2007 models
    """

    #spectra in this file are in erg/s/cm^2/AA/M_sun of dust at a distance of 10pc
    model_file = os.getenv('pyxydust')+'/data/models/DL07.fits'
    #from model fluxes to Lsun per Msun of dust
    flux_unit = 'erg/s/cm^2/AA of 1solar mass at 10pc' 
    convert_to_lsun = 10**( np.log10(4.0*np.pi)+2*np.log10(pc*10)-np.log10(lsun) ) 

    def __init__(self,modfile=None):
        if modfile is not(None) :
            self.model_file=modfile
        self.readFitsModelLibrary(self.model_file)


    def spectraFromPars(self,parstruct):
        """ wrapper on generateSpectrum that parses a pars structured array for the required parameters
        """
        alpha = 2
        mdust = 1
        return self.generateSpectrum(parstruct['UMIN'], parstruct['UMAX'],
                                     parstruct['GAMMA'], parstruct['QPAH'],alpha,mdust=mdust)

    def generateSpectrum(self,umin,umax,gamma,qpah,alpha,mdust = 1):
        """Returns the IR SED of a model given the model parameters.
        calling sequence: DraineLi.generateSpectrum(umin, umax, gamma, qpah, alpha, mdust)
        Output units are erg/s/cm^2/AA at a distance of 10pc.
        mdust defaults to 1 solar mass. output shape is (nobj,nwave)
        """

        if False in [umin.shape[0] == umax.shape[0],umin.shape[0] == qpah.shape[0]] :
            raise ValueError("DraineLi.generateSpectrum: parameter array sizes do not match")
        
        delta_spec = self.interpolateTo( np.array([umin,qpah]).T,
                                         parnames = ['UMIN','QPAH'], subinds = self.delta_inds )
        pdr_spec = 0
        if np.any(gamma > 0):
            pdr_spec = self.interpolateTo( np.array([umin,umax,qpah]).T,
                                           parnames = ['UMIN','UMAX','QPAH'], subinds = self.pdr_inds )
        return (mdust*(delta_spec.T*(1.0-gamma)+pdr_spec.T*gamma)).T
                
    
    def ubar(self,umin,umax,gamma,alpha=2):
        if np.any(alpha != 2):
            raise ValueError('only alpha=2 is currently allowed')
        return (1-gamma)*(umin)+gamma*np.log(umax/umin)/(1/umin-1/umax)

    def readFitsModelLibrary(self,filename,grain='MW3.1'):
        """Read in the dust SED library of Draine & Li 2007, stored as a
        fits binary table with spectral units of erg/s/cm^2/AA/M_sun of dust
        at a distance of 10pc. Produce index arrays to access delta-function
        and alpha=2 power law dust heating intensity distribtions.
        """
        
        if os.path.isfile( filename ) is False :
            raise ValueError('File does not exist: %s',filename)
            return 0.
    
        fits = pyfits.open( filename )
        #parse the FITS recarray and assign ModelGrid parameter, spectra, and wavelength attributes
        self.wavelength = fits[1].data[0]['WAVE']
        parnames=['UMIN','UMAX','QPAH','GRAIN']
        lib_pars=[]
        for pname in parnames:
            lib_pars.append(np.squeeze(fits[1].data[pname]))
        lib_pars = np.array(lib_pars).transpose() #(nmod, ndim)
        self.pars=self.arrayToStruct(lib_pars,parnames,types=['<f8','<f8','<f8','S10'])
        self.spectra=fits[1].data['F_LAMBDA'] #(nmod,nwave)
        fits.close()
        #record the indices of models corresponding to delt-functions and 'pdr' U distributions
        self.delta_inds = np.squeeze(np.array(np.where(np.logical_and(
            (self.pars['UMAX'] == self.pars['UMIN']), (self.pars['GRAIN'] == grain) ) )))
        self.pdr_inds = np.squeeze(np.array(np.where(np.logical_and(
            (self.pars['UMAX'] > self.pars['UMIN']),(self.pars['GRAIN'] == grain) ) )))

    def libraryFromDustEM():
        """if python bindings to DustEM ever happen, you could call it from here
        to fill the model recarray with a dnser grid than Draine provides
        """
        pass


class ModifiedBB(SpecLibrary):
    """ModifiedBB:: Class to work with modified blackbodies a.k.a. greybodies
    """

    def __init__(wavelengths):
        self.wavelength=wavelengths

    def spectraFromPars(self,pars):
        """ wrapper on generateSpectrum that parses a pars structured array for the required parameters
        """            
        return self.generateSpectrum(pars['T_DUST'], pars['BETA'], pars['M_DUST'] )

        
    def generateSpectrum(self, T_dust, beta, M_dust, kappa_lambda0=(1.92, 350e4)):
        """Method to return the spectrum given a list or array of temperatures,
        beta, and dust masses.  Also, the normalization of the emissivity curve can
        be given as a tuple in units of cm^2/g and AA, default = (1.92, 350e4).
        Ouput units are erg/s/AA/cm^2 at 10pc.  should find a way to make this an array
        """

        spec=np.zeros([self.wavelength.shape[0]])
        #should vectorize
        for i,T in enumerate(T_dust) : 
            spec += M_dust[i] * self.planck(T) * (kappa_lambda0[1]/self.wavelength)**beta[i] * kappa_lambda0[0]
        return spec
    
    def planck(self, T) :
        """ Return planck function B_lambda(cgs) for a given T
        """
        wave = self.wavelength*1e8 #convert from AA to cm
        #return B_lambda in erg/s/AA
        return 2*hplank*(c_cgs**2)/(wave**5) * 1/(np.exp( hplanck*c_cgs/(kboltz*T) )-1) * 1e8
        
class BaSeL():
    pass

############ A method for determining nLinear interpolation weights.  unfinished ########

#def weightsLinear(self,model_points,point):
#    """ The interpolation weights are determined from the distances to the nearest
#    grid points in each dimension.  There will be 2**ndim indices and weight products,
#    corresponding to the vertices of the (hyper)-square.  Therefore, this routine gets
#    nasty in high-dimensional spaces.  stay out of them. Requires rectilinearly gridded models.
#    see also scipy.ndimage.interpolate.map_coordinates """
#    #the n-Linear interpolation as defined here *is* invariant under rescaling of any dimension
#            
#    ndim=point.shape[0]
#        #vectorize?  allow multiple points and/or remove loop over dimensions.  need to write
#        #down the math that will speed this up. or turn into a ufunc
#    for idim in xrange(ndim):
#            #unique sorted model grid point values and
#            #distance from point (n_uniqueval)
#        model_vals = np.unique(model_thisdim)
#        dp = point[idim] - model_vals
#
#            #get the weight of the lower bounding model grid point.  The weight
#            #will be zero if grid point is higher than target (dp < 0) or grid point 
#            #one grid step or more less than target (dp/dbin >= 1) 
#        dbin = np.append(model_vals[1:]-model_vals[:-1],-1) #distance from grid point to next highest
#        w1 = (1-dp/dbin) * (dp >= 0) *  (dp/dbin < 1)
#            #if w1<0 that means target was above the whole grid, use nearest grid (which has w1<0)
#        w1 = np.where(w1 >= 0, w1,1) 
#
#            #get the weight of the upper bounding model grid point by
#            #reversing signs of above  The weight
#            #will be zero if grid point is lower than target (dp < 0) or grid point 
#            #one grid step or more more than target (dp/dbin >= 1)
#        dp = dp*(-1)
#        dbin = np.append(-1,model_vals[1:]-model_vals[:-1]) #distance from grid point to next lowest
#        w2 = (1-dp/dbin) * (dp > 0) *  (dp/dbin < 1)
#            #if w2<0 that means target was lower than the whole grid, use nearest grid (which has w1<0)
#        w2 = np.where(w2 >= 0, w2,1) 

#            #index into the model grid point values (nmod)
#        model_index = np.digitize(np.squeeze(model_points[idim,:]), model_vals)
        
#    bins=np.unique(model_points[idim,:])
#    ind=np.digitize(point[idim],bins)

#IDW weighting below:
#dists = np.sqrt( ( (dtri.points[inds]-point)**2 ).sum(1) )
#weights = 1.0/dists
#interpolate only if you need to.  should probably check before interpolation is run
#(or triangulation) but this is a failsafe
#exact = np.where(dists == 0.)
#if (exact[0].shape[0] > 0):
#    weights = np.array([1.0])
#    inds = inds[exact[0]]
#need to renormalize

#def testWeighting():
#    m=np.array([0.,1,2,3])
#    t=1.4
#    #assert (self.weightsLinear(m,t) is [1,2] ,[0.6,0.4])
#    t=1
#    #assert (self.weightsLinear(m,t) is [1] ,[1])
#    m=np.array([0,1,1,2,3])
#        
#    raise ValueError('No test written')
#    pass
