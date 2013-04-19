#Python module to deal with dust emission models,
#primarily the Draine+Li models but others can be included

import pyfits
import numpy as np
import scipy.spatial
import os
import astropy.constants as constants

lsun = constants.L_sun.cgs.value
pc = constants.pc.cgs.value
hplanck = constants.h.cgs.value
c_cgs = constants.c.cgs.value
kboltz = constants.k_B.cgs.value

class SpecLibrary(object):
    """Class to operate on spectral libraries. Methods are provided to interpolate the
    available model spectra (stored as a rec array or FITS rec array) to a certain
    set of parameters.  Subclasses are used to return the actual model spectrum given
    a set of model parameters """

    def __init__(self):
        pass

    def interpolateTo(self, target_points, fieldnames=None, itype='dt',subinds=None, flux_field_name='F_LAMBDA'):
        """Method to obtain the model spectrum for a given set of parameter values
        via interpolation of the model grid. The interpolation weights are
        determined from barycenters of a Delaunay triangulation or nLinear interpolation
        
        The input is an array of target model parameters, optionally a string list of the
        fields containing the corresponding library model parameters

            target_points - ndarray of shape (ntarg,ndim) of ndim desired model parameters.
                            Can also be a recarray with fields named for the model parameters
            subinds - ndarray of indices of the model library structure to use in interpolation.
                      allows for only portions of the library to be used
            fields  - string list of the field names of the model library parameters """
        
        #Note: should (but doesn't yet) check that point is in the grid
        #deal with recarray input
        
        if fieldnames is None:
            fields = target_points.dtype.names #if the target point(s) is already a recarray use the field names
            targets = np.array(target_points.tolist())
        else:
            targets = target_points
        if targets.ndim is 1:
            targets=targets[np.newaxis,:]
        
        #if len(fields) != targets.shape(-1) :
        #    raise ValueError('#Fields != number of dimensions of reqested target coordinates')
            
        #pull the grid points out of the model record data and make an (nmodel,ndim) array of
        #model grid parameter values.  need to loop to make sure order is correct
        model_points=[]
        for fname in fieldnames:
            model_points.append(np.squeeze(self.model_lib[subinds][fname]))
        model_points = np.array(model_points).transpose() #(nmod, ndim)

        if itype is 'dt' :
            inds, weights = self.weightsDT(model_points,targets)
        else:
            raise ValueError('Only DT weighting available currently')
            #inds, weights = self.weightsLinear(model_points, point)

        #weight using broadcasting, them sum the weighted spectra and return (nwave,ntarg)
        return ( weights* (self.model_lib[subinds[inds]][flux_field_name].transpose(2,0,1)) ).sum(axis=2)


    def weightsDT(self,model_points,target_points):
        """ The interpolation weights are determined from barycenter coordinates
        of the vertices of the enclosing Delaunay triangulation. This allows for
        the use of irregular Nd grids. see also weightsLinear.
          model_points - array of shape (nmod, ndim)
          target_points - array of shape (ntarg,ndim)
          output inds and weights - harrays of shape (npts,ndim+1)"""

        #Note: should (but doesn't yet) allow for grid (and point) to be scaled in any or
        #all dimensions.  The DT is *not* invariant under scaling.
        
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
            
        return inds, weights 
               
        
class DraineLi(SpecLibrary):
    """DraineLi:: Class to store and operate on Draine & Li 2007 models"""

    #spectra in this file are in erg/s/cm^2/AA/M_sun of dust at a distance of 10pc
    model_file=os.getenv('pyxydust')+'/data/models/DL07.fits'
    
    #from model fluxes to Lsun per Msun of dust
    convert_to_lsun = 10**( np.log10(4.0*np.pi)+2*np.log10(pc*10)-np.log10(lsun) ) 

    def __init__(self,modfile=None):
        if modfile is not(None) :
            self.model_file=modfile
        self.readFitsModelLibrary(self.model_file)

    def generateSpectrum(self,umin,umax,gamma,qpah,alpha,mdust = 1):
        """Returns the IR SED of a model given the model parameters.
        calling sequence: DraineLi.generateSpectrum(umin, umax, gamma, qpah, alpha, mdust)
        Output units are erg/s/cm^2/AA at a distance of 10pc.
        mdust defaults to 1 solar mass"""
        
        delta_spec = self.interpolateTo( np.array([umin,qpah]).T,
                                         fieldnames = ['UMIN','QPAH'], subinds = self.delta_inds )
        pdr_spec = 0
        if (gamma > 0):
            pdr_spec = self.interpolateTo( np.array([umin,umax,qpah]).T,
                                           fieldnames = ['UMIN','UMAX','QPAH'], subinds = self.pdr_inds )
                    
        return mdust*(delta_spec*(1-gamma)+pdr_spec*gamma)
        

    def readFitsModelLibrary(self,filename,grain='MW3.1'):
        """Read in the dust SED library of Draine & Li 2007, stored as a
        fits binary table with spectral units of erg/s/cm^2/AA/M_sun of dust
        at a distance of 10pc. Produce index arrays to access delta-function
        and alpha=2 power law dust heating intensity distribtions."""
        
        if os.path.isfile( filename ) is False :
            raise ValueError('File does not exist: %s',filename)
            return 0.
    
        fits = pyfits.open( filename )
        self.model_lib = fits[1].data
        self.wavelength = fits[1].data[0]['WAVE']
        self.delta_inds = np.squeeze(np.array(np.where(
            np.logical_and( (self.model_lib['UMAX'] == self.model_lib['UMIN']),
                            (self.model_lib['GRAIN'] == grain) ) )))
        self.pdr_inds = np.squeeze(np.array(np.where(
                        np.logical_and( (self.model_lib['UMAX'] > self.model_lib['UMIN']),
                                        (self.model_lib['GRAIN'] == grain) ) )))
        #self.delta_model_lib = self.model_lib[delta_inds]
        #self.pdr_model_lib = self.model_lib[pdr_inds]
        fits.close()

    def libraryFromDustEM():
        """if python bindings to DustEM ever happen, you could call it from here
        to fill the model recarray with a dnser grid than Draine provides"""
        pass


class ModifiedBB(SpecLibrary):
    """ModifiedBB:: Class to work with modified blackbodies a.k.a. greybodies """

    def __init__(wavelengths):
        self.wavelength=wavelengths
        
    def generateSpectrum(self, T_dust, beta, M_dust, kappa_lambda0=(1.92, 350e4)):
        """Method to return the spectrum given a list or array of temperatures,
        beta, and dust masses.  Also, the normalization of the emissivity curve can
        be given as a tuple in units of cm^2/g and AA, default = (1.92, 350e4).
        Ouput units are erg/s/AA/cm^2 at 10pc.  should find a way to make this an array"""

        spec=np.zeros([self.wavelength.shape[0]])
        for i,T in enumerate(T_dust) : 
            spec += M_dust[i] * self.planck(T) * (kappa_lambda0[1]/self.wavelength)**beta[i] * kappa_lambda0[0]
        return spec
    
    def planck(self, T) :
        """ Return planck function B_lambda(cgs) for a given T """
        wave = self.wavelength*1e8 #convert from AA to cm
        #return B_lambda in erg/s/AA
        return 2*hplank*(c_cgs**2)/(wave**5) * 1/(np.exp( hplanck*c_cgs/(kboltz*T) )-1) * 1e8
        


############ A method for detremining nLinear interpolation weights.  unfinished ########
def weightsLinear(self,model_points,point):
    """ The interpolation weights are determined from the distances to the nearest
    grid points in each dimension.  There will be 2**ndim indices and weight products,
    corresponding to the vertices of the (hyper)-square.  Therefore, this routine gets
    nasty in high-dimensional spaces.  stay out of them. Requires rectilinearly gridded models.
    see also scipy.ndimage.interpolate.map_coordinates """
    #the n-Linear interpolation as defined here *is* invariant under rescaling of any dimension
            
    ndim=point.shape[0]
        #vectorize?  allow multiple points and/or remove loop over dimensions.  need to write
        #down the math that will speed this up. or turn into a ufunc
    for idim in xrange(ndim):
            #unique sorted model grid point values and
            #distance from point (n_uniqueval)
        model_vals = np.unique(model_thisdim)
        dp = point[idim] - model_vals

            #get the weight of the lower bounding model grid point.  The weight
            #will be zero if grid point is higher than target (dp < 0) or grid point 
            #one grid step or more less than target (dp/dbin >= 1) 
        dbin = np.append(model_vals[1:]-model_vals[:-1],-1) #distance from grid point to next highest
        w1 = (1-dp/dbin) * (dp >= 0) *  (dp/dbin < 1)
            #if w1<0 that means target was above the whole grid, use nearest grid (which has w1<0)
        w1 = np.where(w1 >= 0, w1,1) 

            #get the weight of the upper bounding model grid point by
            #reversing signs of above  The weight
            #will be zero if grid point is lower than target (dp < 0) or grid point 
            #one grid step or more more than target (dp/dbin >= 1)
        dp = dp*(-1)
        dbin = np.append(-1,model_vals[1:]-model_vals[:-1]) #distance from grid point to next lowest
        w2 = (1-dp/dbin) * (dp > 0) *  (dp/dbin < 1)
            #if w2<0 that means target was lower than the whole grid, use nearest grid (which has w1<0)
        w2 = np.where(w2 >= 0, w2,1) 


            #index into the model grid point values (nmod)
        model_index = np.digitize(np.squeeze(model_points[idim,:]), model_vals)
        
    bins=np.unique(model_points[idim,:])
    ind=np.digitize(point[idim],bins)

def testWeighting(self):
    m=np.array([0.,1,2,3])
    t=1.4
    #assert (self.weightsLinear(m,t) is [1,2] ,[0.6,0.4])
    t=1
    #assert (self.weightsLinear(m,t) is [1] ,[1])
    m=np.array([0,1,1,2,3])
        
    raise ValueError('No test written')
    pass
