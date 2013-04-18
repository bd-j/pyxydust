
class ModelGrid(object):

    def __init__():
        self.nmod = 0
        self.pars = None

    def generateSEDs(self,Library, filters, pars = None, wave_min = 90, mave_max = 1e7):
        if pars != None :
            self.setPars(pars)
        #self.spec = np.array([self.nmod,Library.wavelength.shape[0]])
        self.sed = np.array([self.nmod,len(filters)])
        for i in xrange(self.nmod):
            spec = Library.generateSpectrum(self.pars[i]) #fix the passing of parameters
            self.sed[i,:] = observate.getSED(Library.wavelength,spec,filters)

            inds=np.where(np.logical_and(Library.wavelength < wave_max, Library.wavelength >= wave_min))
            self.lbol[i] = dl07.convert_to_lsun*np.trapz(spec[inds],dl07.wavelength[inds]) 

    def setPars(self,pars,parnames):
        self.pars = self.parsToStruct(pars,parnames)
        self.nmod = self.pars.shape[0]
        
    def parsToStruct(pars,parnames):
        #pars can be a list or string array of parameter names
        #assumes pars is a numpy array of shape (nmod,npar)
        nmod=pars.shape[0]
        #set up the list of tuples describing the fields.  Assume each parameter is a float
        partuple=[]
        for p in parnames:
            partuple.append((p,'<f8'))
        #create the dtype and structured array                    
        dt=np.dtype(partuple)
        parstruct=np.zeros(nmod,dtype=dt)
        for i,p in enumerate(parnames):
            parstruct[p]=pars[i,:]
        return parstruct
