import pylab

def plot_all_samples2d(sourcename):
    figure(1,(8.0,8.0))
    for i in xrange(npar):
        for j in xrange(i):
            subplot(npar,npar,i*j+1)
            plotsamples2d(i,j)
    tightlayout(0.1)
    savefig('%s_param_dist.png' %sourcename)
    
def plotsamples2d(i,j):


    # plot prior samples: complicated because of 10**umin and umax>umin
    #    if (k >=0) and (h >=0) :
    #    plot( np.random.uniform(par_range[k,0],par_range[h,1],nwalkers),
    #          np.random.uniform(par_range[k,0],par_range[h,1],nwalkers),
    #          linestyle='none', marker='o', color='blue', mec='blue',
    #        alpha=.5, label='Prior', zorder=-100)

    #figure(1).clear()
    #axes=gca()
    xlabel(par_names[i], fontsize=3)
    ylabel(par_names[j], fontsize=3)#,rotation=0
    plot( allpars[i,:], allpars[j,:],
          linestyle='none', marker='o', color='red', mec='red',
        alpha=.5, label='Posterior', zorder=-99)
    legend()
    if par_log[i] == 1 : xscale('log')
    if par_log[j] == 1 : yscale('log')

    #import scipy.stats
    #gkde = scipy.stats.gaussian_kde([allpars[i,:], allpars[j,:]])
    #x,y = np.mgrid[0:40:.05, 0:1.1:.05]
    #z = array(gkde.evaluate([x.flatten(),y.flatten()])).reshape(x.shape)
    #contour(x, y, z, linewidths=1, alpha=.5, cmap=cm.Greys)

    
    #axis([-1, 41, 10, 1e7])
    
