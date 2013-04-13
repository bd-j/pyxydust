#Python module to deal with dust emission models,
#primarily the Draine+Li models but others can be included

import pyfits
import numpy as np
import scipy.spatial
import os
import astropy.constants

lsun=astropy.constants.L_sun.cgs.value
pc=astropy.constants.pc.cgs.value

class DraineLi(object):
    """DraineLi:: Class to store and operate on Draine & Li 2007 models"""

    #this should not be hard-coded
    #spectra in this file are in erg/s/cm^2/AA/M_sun of dust at a distance of 10pc
    model_file=os.getenv('pyxydust')+'/data/models/DL07.fits'
    convert_to_lsun = 10**( np.log10(4.0*np.pi)+2*np.log10(pc*10)-np.log10(lsun) ) #from model fluxes to Lsun per Msun of dust

    def __init__(self):
        self.readFitsModelLibrary(self.model_file)

    def generateSpectrum(self,umin,umax,gamma,qpah,alpha):
        """Returns the IR SED of a model given the model parameters.
        calling sequence: DraineLi.generateSpectrum(umin, umax, gamma, qpah, alpha)
        Output units are erg/s/cm^2/AA/M_sun of dust at a distance of 10pc."""
        
        delta_spec = self.interpolateDT( self.delta_model_lib,
                                         ['UMIN','QPAH'],np.array([umin,qpah]).reshape(2) )
        if (gamma > 0):
            pdr_spec = self.interpolateDT( self.pdr_model_lib,
                                           ['UMIN','UMAX','QPAH'],np.array([umin,umax,qpah]).reshape(3) )
        else:
            pdr_spec = 0
            
        out_model = delta_spec*(1-gamma)+pdr_spec*gamma
        return out_model

    def interpolateDT(self,model_recs,rec_fields,point):
        """Method to obtain the dust spectrum for a given set of parameter values
        via interpolation of the model grid. The interpolation weights are
        determined from inverse distances to the vertices of the enclosing
        Delaunay triangulation. This allows for the use of irregular Nd
        grids.  For regular grids it might be better to use bilinear/n-linear
        interpolation methods.

        The input is a pyfits Fits record array containing the model library, a string
        list of the fields containing the grid parameters, and a numpy array
        giving the desired parameters."""
        #Note: should (but doesn't yet) allow for grid (and point) to be scaled in any or all dimensions
        #Note: should (but doesn't yet) check that point is in the grid
    
        #pull the grid points out of the record data and make
        #an (npoints,ndim) array of model grid parameter values
        model_points=[]
        for f in rec_fields:
            model_points.append(model_recs.field(f)) 
        model_points = np.array(model_points).transpose()

        #now delaunay triangulate and find the encompassing
        #(hyper)triangle for the desired point
        dt = scipy.spatial.Delaunay(model_points)
        triangle = dt.find_simplex(point)
        
        #and get model indices of (hyper)triangle vertices and distances (inverse weights)
        #of those vertices
        inds = dt.vertices[triangle,:]
        dists = np.sqrt( ( (dt.points[inds]-point)**2 ).sum(1) )
        weights = 1.0/dists
        
        #interpolate only if you need to.  should probably check before interpolation is run
        #(or triangulation) but this is a failsafe
        exact=np.where(dists == 0.)
        if (exact[0].shape[0] > 0):
            spec = model_recs[inds[exact[0]]].field('F_LAMBDA')
        else:
            spec = ( weights* (model_recs[inds].field('F_LAMBDA').transpose()) ).sum(1)
            spec = spec/( weights.sum() ) #renormalize
        return spec

    def readFitsModelLibrary(self,filename,grain='MW3.1'):
        """Read in the dust SED library of Draine & Li 2007, stored as a
        fits binary table with spectral units of erg/s/cm^2/AA/M_sun of dust
        at a distance of 10pc. Produce separate libraries for delta-function
        and alpha=2 power law dust heating intensity distribtions."""
        
        if os.path.isfile( filename ) == False :
            raise ValueError('File does not exist: %s',filename)
            return 0.
    
        fits = pyfits.open( filename )
        self.model_lib = fits[1].data
        self.wavelength = fits[1].data[0].field('WAVE')
        delta_inds = np.where( np.logical_and( (self.model_lib.field('UMAX') == self.model_lib.field('UMIN')),
                               (self.model_lib.field('GRAIN') == grain) ) )
        pdr_inds = np.where( np.logical_and( (self.model_lib.field('UMAX') > self.model_lib.field('UMIN')),
                             (self.model_lib.field('GRAIN') == grain) ) )
        self.delta_model_lib = self.model_lib[delta_inds]
        self.pdr_model_lib = self.model_lib[pdr_inds]
        fits.close()


class MBB():
    """MBB:: Class to work with modified blackbodies"""
    pass

