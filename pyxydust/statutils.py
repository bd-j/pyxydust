import numpy as np


#####
##### function to obtain likelihood ####
#####

def lnprob_grid(grid, obs, err, mask):
    #linearize fluxes.  
    inds = np.where(mask > 0)
    inds=inds[0]
    mod_maggies = 10**(0-grid.sed[...,inds]/2.5)
    obs_maggies = 10**(0-obs[inds]/2.5)
    obs_ivar = (obs_maggies[inds]*err[inds]/1.086)**(-2)

    #best scale for these parameters.  sums are over the bands
    mdust = np.squeeze(( (mod_maggies*obs_maggies*obs_ivar).sum(axis=-1) ) /
                       ( ( (mod_maggies**2.0)*obs_ivar ).sum(axis=-1) ))
    lbol = grid.lbol*mdust
        
    #probability with dimensional juggling to get the broadcasting right
    chi2 =  (( (mdust*mod_maggies.T).T - obs_maggies)**2)*obs_ivar
    lnprob = np.squeeze(-0.5*chi2.sum(axis=-1))
    delta_mag = 0-2.5*np.log10((mdust*mod_maggies.T).T/obs_maggies)
    #clean NaNs
    
    
    return lnprob, lbol, mdust, delta_mag


#not working yet....
def cdf_moment(par,lnprob,percentiles,save=False,plot=False):
    order = np.argsort(par)
    cdf = np.cumsum(np.exp(lnprob[order])) / np.sum(np.exp(lnprob))
    ind_ptiles= np.searchsorted(cdf,percentiles)
    ind_max=np.argmax(lnprob_isnum)


    return np.concatenate(par[order[ind_ptiles]],par[ind_max])


