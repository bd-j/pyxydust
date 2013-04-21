import numpy as np
import observate

class ModelGrid(object):

    def __init__(self,pars=None):
        ngrid = 0
        pars = None
        sed = None
        if pars is not None :
            self.setParameters(pars)        
        
    
    def generateSEDs(self,Library, filterlist, pars = None, wave_min = 90, wave_max = 1e7, keepspec = False):

        if pars is not None :
            self.setParameters(pars)
        #self.spec = np.array([self.nmod,Library.wavelength.shape[0]])
        self.filter_names = [f.name for f in filterlist]
        
        spec = Library.spectraFromPars(self.pars).T
        print(spec.shape)
        self.sed = observate.getSED(Library.wavelength,spec,filterlist)
        self.lbol = Library.convert_to_lsun*observate.Lbol(Library.wavelength,spec,wave_min,wave_max)

        if keepspec : self.spec=spec

    def setParameters(self, pars,parnames):
        self.pars = self.arrayToStruct(pars,parnames)
        self.nmod = self.pars.shape[0]
        
        
    def arrayToStruct(self, values,fieldnames):
        #parnames can be a list or string array of parameter names with length nfield
        #assumes pars is a numpy array of shape (nobj,nfield)
        values=np.atleast_2d(values)
        if values.shape[-1] != len(fieldnames):
            if values.shape[0] == len(fieldnames):
                values=values.T
            else:
                raise ValueError('ModelGrid.parsToStruct: array and fieldnames do not have consistent shapes!')
        nobj=values.shape[0]
        
        #set up the list of tuples describing the fields.  Assume each parameter is a float
        fieldtuple=[]
        for f in fieldnames:
            fieldtuple.append((f,'<f8'))
        #create the dtype and structured array                    
        dt=np.dtype(fieldtuple)
        struct=np.zeros(nobj,dtype=dt)
        for i,f in enumerate(fieldnames):
            struct[f]=values[...,i]
        return struct
