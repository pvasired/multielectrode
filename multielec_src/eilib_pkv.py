#This module provides a set of functions for interacting with EI files 

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#plt.switch_backend('Agg')
import glob
import pdb

#global definitions--------------------
corrColDict = {}
corrColDict['b'] = '#1f77b4'
corrColDict['r'] = '#d62728'
corrColDict['g'] = '#2ca02c'
corrColDict['y'] = '#bcbd22'
corrColDict['c'] = '#17becf'
corrColDict['k'] = 'k'

def getCellLocs(eiarr, thr):

	ncell = len(eiarr)
	locs = np.zeros((ncell,2))

	for c in range(ncell):
		ei = np.amin(eiarr[c,0], axis=1)
		if np.amin(ei) < thr: 
			els = ei.argsort()[:3]
			elvals = sorted(ei)[:3]
			if isadjacent512(els[0],els[1]) & isadjacent512(els[1],els[2]):
				cent = np.zeros(2)
				for i in range(len(elvals)):
					cent += getElecCoords512(els[i])*elvals[i]
				locs[c] = cent/sum(elvals)
			else: #JANKY!!!!!!!!!!!!	Because just uses center elec
				locs[c] = getElecCoords512(els[0])

	return locs

def getLocCentroid(locs):
	return np.mean(locs,axis=0)

def isadjacent512(e1,e2):
	adjmat = loadmat('/Volumes/Lab/Users/Sasi/pystim/files/adj_mat_512.mat')
	if e2+1 in adjmat['adj_mat_512'][e1,0]: return True
	else: return False

def loadCoords512():
	coords = loadmat('/Volumes/Lab/Users/Sasi/pystim/files/512Coords.mat')
	return coords['coords']

def load_Harray_coords(eipath):

    # Loads file with coordinates: currently 'data000.txt'.
    try:
        coords = np.loadtxt(eipath+'/'+'data000.txt')
    except FileNotFoundError:
        print("Electrode coordinates file not found in path or path",\
                "doesn't exist.")
        raise(FileNotFoundError)

    # Transforms the coordinates to microns and excludes first entry because 
    # it is centroid (0,0).
    return (coords[1:,:] * 17.5) 

def loadAdjMat512():
    adjmat = loadmat('/Volumes/Lab/Users/Sasi/pystim/files/adj_mat_512.mat')
    return adjmat['adj_mat_512']

def loadAdjMat519():
    adjmat = loadmat('/home/agogliet/gogliettino/pystim/files/adj-elecs-tools/adj_mat_519.mat')
    return adjmat['adj_mat_519']

def loadCoords519():
	coords = loadmat('/Volumes/Lab/Users/Sasi/pystim/files/519Coords.mat')
	return coords['coords']

def loadChipElecs512():
    coords = loadmat('/Volumes/Lab/Users/Sasi/pystim/files/elecs_by_chip.mat')
    x = coords['elecs_by_chip']
    out = []
    for i in range(x.shape[1]):
        foo = []
        for j in range(x[0,i].shape[1]):
            foo.append(x[0,i][0,j])
        out.append(foo)
    return out

def getElecCoords512(el):
    #coords = loadCoords512()['coords']
	coords = loadCoords512()
	return coords[el,:]

def getLocDist(loc1, loc2):
    return np.sqrt(abs(loc1[0] - loc2[0])**2 + abs(loc1[1] - loc2[1])**2)

def getEIStrength(eiarr):
    out = []
    for e in range(len(eiarr)): 
        out.append(np.amin(erarr[e,0]))

def getAverageMainSpikeAmp(eis):
    cnt = 0
    foo = 0
    for n in range(len(eis)):
        foo += np.min(np.min(eis[n,0],axis=1))
        cnt += 1
    return foo/cnt

def getMainSpikeAmp(eis,nidx):
    return np.min(np.min(eis[nidx,0],axis=1))

def getMainElec(eis,nidx):
    return np.argmin(np.min(eis[nidx,0],axis=1))

def getMedianMainSpikeAmp(eis):
    foo = []
    for n in range(len(eis)):
        foo.append(np.min(np.min(eis[n,0],axis=1)))
    return np.median(foo)

def getSomaLoc(eis, elec_coords, thr=-3):
    ei = np.amin(eis, axis=1)
    elecinds = np.nonzero(ei < thr)[0]

    #Filter by waveform (triphasic or biphasic)
    somalist = []
    weightlist = []
    for e in elecinds:
        foo = eis[e,:].flatten()
        if np.argmax(foo) > np.argmin(foo):
            somalist.append(e)
            weightlist.append(abs(np.min(foo)))
    elecinds = np.array(somalist)
    weightlist = np.array(weightlist)/sum(weightlist)

    #Get cell coordinates
    cell_coords_x = [elec_coords[j,0] for j in elecinds]
    cell_coords_y = [elec_coords[j,1] for j in elecinds]

    #Compute average position
    xcoord = np.dot(cell_coords_x, weightlist)
    ycoord = np.dot(cell_coords_y, weightlist)
    return (xcoord, ycoord)

def getEI(eis, neuron_index):
    ei = np.amin(eis[neuron_index,0], axis=1)
    return ei

def getEIofPat(eis, neuron_index, pattern):
    ei = np.amin(eis[neuron_index,0], axis=1)
    return ei[pattern]

def getNumSpikes(spikes, neuron_index):
    return spikes[neuron_index,0].shape[0]


def getSomaLoc2(eis, neuron_index, elec_coords, adj_mat,numElecs=7):
    #ei = np.amin(eis[neuron_index,0], axis=1)
    ei = np.amin(eis[neuron_index,:,:],axis=1)

    #Get elec list
    main_elec = np.argmin(ei)
    adj_elecs = (adj_mat[main_elec,0] -1).flatten()

    #Filter by number elecs desired
    weightlist = []
    included_elecs = []
    for e in adj_elecs:
        #foo = eis[neuron_index,0][e,:].flatten()
        foo = eis[neuron_index,e,:]
        weightlist.append(abs(np.min(foo)))
    included_elecs = list(np.array(adj_elecs)[np.argsort(weightlist)])[-(numElecs-1):]
    weightlist = list(np.sort(weightlist)[-(numElecs-1):])
    #weightlist.append(abs(np.min(eis[neuron_index,0][main_elec,:].flatten())))
    weightlist.append(abs(np.min(eis[neuron_index,main_elec,:])))
    included_elecs.append(main_elec)
    #included_elecs = np.concatenate((np.array([main_elec]), adj_elecs))

    #Take weighted sum
    #weightlist = []
    #for e in included_elecs:
    #    foo = eis[neuron_index,0][e,:].flatten()
    #    weightlist.append(abs(np.min(foo)))
    weightlist = np.array(weightlist)/sum(weightlist)

    #Get cell coordinates
    cell_coords_x = [elec_coords[j,0] for j in included_elecs]
    cell_coords_y = [elec_coords[j,1] for j in included_elecs]

    #Compute average position
    xcoord = np.dot(cell_coords_x, weightlist)
    ycoord = np.dot(cell_coords_y, weightlist)
    return (xcoord, ycoord)

def getElecCellPos(eis, neuron_index, elec, adj_mat):
    coords = loadCoords512()
    ei = np.amin(eis[neuron_index,0], axis=1)
    main_elec = np.argmin(ei)
    adj_elecs = (adj_mat['adj_mat_512'][main_elec,0] -1).flatten()
    if elec == main_elec:
        return ['s',0]
    elif elec in adj_elecs:
        return ['a',getAdjElecDir(main_elec,elec)/60]#NORMALIZED WITH 60
    else:
        elec_loc = coords[elec,:]
        soma_loc = getSomaLoc2(eis, neuron_index, coords, adj_mat)
        dist = getLocDist(elec_loc, soma_loc)
        return ['d',dist/60]#NORMALIZED WITH 60 


def getAdjElecDir(elec1, elec2):
    coords = loadCoords512()
    c1 = coords[elec1,:]
    c2 = coords[elec2,:]
    return c1 - c2
#def plotEiStim(eis, neuron_index, stim_elec, ax, saveName='', thr=-1):
def plotEiStim(ei, stim_elecs, ax, coords, size_fac=-25,thr=-1,stim_sf=240,stim_mew=3,ei_color='b',stim_color='y',alpha=1,label=None):
    #Data
    #coords = loadCoords512()
    xs = coords[:,0]
    ys = coords[:,1]
    ##EI
    #ei = np.amin(eis[neuron_index,0], axis=1)
    goodInds = np.nonzero(ei<thr)[0]
    eixs = coords[goodInds,0]
    eiys = coords[goodInds,1]
    #Plotting
    #fig = plt.figure(figsize=(18,12))
    #ax = fig.add_subplot(1,1,1)
    ##Plot elecs
    #ax.scatter(xs, ys, facecolors='none', edgecolors='b', s=110) #DEFAULT
    ax.scatter(xs, ys, facecolors='k', edgecolors='k', s=5,alpha=.5)
    ##Plot EI
    #ax.scatter(eixs,eiys, facecolors='b', edgecolors='b', s=list(ei[goodInds]*-55)) #DEFAULT
    ax.scatter(eixs,eiys, facecolors=ei_color, edgecolors=ei_color, s=list(ei[goodInds]*size_fac),alpha=alpha,label=label)
    ##Plot stim elec
    for stim_elec in stim_elecs:
        stim_elec_coords = coords[stim_elec-1]
        ax.scatter(stim_elec_coords[0],stim_elec_coords[1], facecolors=stim_color, edgecolors=stim_color, marker='x',s=stim_sf,linewidth=stim_mew)
    ##Cosmetics
    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.set_xlim([min(xs)-100, max(xs)+100])
    ax.set_ylim([min(ys)-100, max(ys)+100])
    #Add electrode labels
    #if saveName:
    #    plt.savefig(saveName)
    #plt.close('all')
    #return fig, ax

def plotEiMarked(ei, marked_elecs, ax, coords, size_fac=-25,thr=-1,stim_sf=240,stim_mew=3,myalpha=1,myfc='b',mymarkedfc='y',mymarkedalpha=0.6,same_size=False,same_size_sf=40,noArray=False,zorder=2):
    #MARKED ELECS given in terms of pattern - 1, so electrode index
    xs = coords[:,0]
    ys = coords[:,1]
    #stim_elec_coords = coords[stim_elec-1]
    marked_elec_xs = coords[marked_elecs,0]
    marked_elec_ys = coords[marked_elecs,1]
    ##EI
    #ei = np.amin(eis[neuron_index,0], axis=1)
    goodInds = np.nonzero(ei<thr)[0]
    eixs = coords[goodInds,0]
    eiys = coords[goodInds,1]
    if not noArray:
        ax.scatter(xs, ys, facecolors='k', edgecolors='k', s=5,alpha=.5,zorder=0)
    #ax.scatter(eixs,eiys, facecolors='b', edgecolors='b', s=list(ei[goodInds]*-55)) #DEFAULT
    #ax.scatter(eixs,eiys, facecolors=myfc, edgecolors='b', s=list(ei[goodInds]*size_fac),alpha=myalpha)
    if same_size:
        ax.scatter(eixs,eiys, facecolors=myfc, edgecolors='none',s=same_size_sf,alpha=myalpha,zorder=1)
    else:
        ax.scatter(eixs,eiys, facecolors=myfc, edgecolors='none',s=list(ei[goodInds]*size_fac),alpha=myalpha,zorder=1)
    ##Plot stim elec
    #ax.scatter(stim_elec_coords[0],stim_elec_coords[1], facecolors='y', edgecolors='y', marker='x',s=stim_sf,linewidth=stim_mew)
    if same_size:
        ax.scatter(marked_elec_xs,marked_elec_ys, facecolors=mymarkedfc, edgecolors='b', s=same_size_sf,alpha=mymarkedalpha,zorder=zorder)
    else:
        #ax.scatter(marked_elec_xs,marked_elec_ys, facecolors=mymarkedfc, edgecolors='b', s=list(ei[marked_elecs]*size_fac),alpha=mymarkedalpha)
        ax.scatter(marked_elec_xs,marked_elec_ys, facecolors=mymarkedfc, edgecolors='none', s=list(ei[marked_elecs]*size_fac),alpha=mymarkedalpha,zorder=zorder)
    ##Cosmetics
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
    ax.set_xlim([min(xs)-100, max(xs)+100])
    ax.set_ylim([min(ys)-100, max(ys)+100])

    #return marked for annotation purposes
    return marked_elec_xs,marked_elec_ys

def plotEiMarkedSquared(ei, marked_elecs, ax, coords, size_fac=-25,thr=-1,stim_sf=240,stim_mew=3,myalpha=1,myfc='b',mymarkedfc='y',mymarkedalpha=0.6,same_size=False,same_size_sf=40,power=2,noArray=False,marked_marker=None,marked_size=0):
    #MARKED ELECS given in terms of pattern - 1, so electrode index
    xs = coords[:,0]
    ys = coords[:,1]
    #stim_elec_coords = coords[stim_elec-1]
    marked_elec_xs = coords[marked_elecs,0]
    marked_elec_ys = coords[marked_elecs,1]
    ##EI
    #ei = np.amin(eis[neuron_index,0], axis=1)
    goodInds = np.nonzero(ei<thr)[0]
    eixs = coords[goodInds,0]
    eiys = coords[goodInds,1]
    if not noArray:
        ax.scatter(xs, ys, facecolors='k', edgecolors='k', s=5,alpha=.5,zorder=0)
    #ax.scatter(eixs,eiys, facecolors='b', edgecolors='b', s=list(ei[goodInds]*-55)) #DEFAULT
    #ax.scatter(eixs,eiys, facecolors=myfc, edgecolors='b', s=list(ei[goodInds]*size_fac),alpha=myalpha)
    if same_size:
        ax.scatter(eixs,eiys, facecolors=myfc, edgecolors='none',s=same_size_sf,alpha=myalpha,zorder=1)
    else:
        ax.scatter(eixs,eiys, facecolors=myfc, edgecolors='none',s=list(np.array(np.abs((ei[goodInds]),dtype=object)**power)*-size_fac),alpha=myalpha,zorder=1)
    ##Plot stim elec
    #ax.scatter(stim_elec_coords[0],stim_elec_coords[1], facecolors='y', edgecolors='y', marker='x',s=stim_sf,linewidth=stim_mew)
    if same_size:
        ax.scatter(marked_elec_xs,marked_elec_ys, facecolors=mymarkedfc, edgecolors='b', s=same_size_sf,alpha=mymarkedalpha,marker=marked_marker)
    else:
        if marked_size == 0: #determine size the same way as for other ei dots
            ax.scatter(marked_elec_xs,marked_elec_ys, facecolors=mymarkedfc, edgecolors='b', s=list(np.array(np.abs((ei[marked_elecs]),dtype=object)**power)*-size_fac),alpha=mymarkedalpha,marker=marked_marker)
        else:
            ax.scatter(marked_elec_xs,marked_elec_ys, facecolors=mymarkedfc, edgecolors='b', s=marked_size,alpha=mymarkedalpha,marker=marked_marker)
    ##Cosmetics
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
    ax.set_xlim([min(xs)-100, max(xs)+100])
    ax.set_ylim([min(ys)-100, max(ys)+100])

def plotMarkedOnly(ei,marked_elecs,ax,coords,size_fac=-25,mymarkedalpha=0.6,mymarkedfc='y',same_size=False,same_size_sf=50):
    marked_elec_xs = coords[marked_elecs,0]
    marked_elec_ys = coords[marked_elecs,1]
    if same_size:
        ax.scatter(marked_elec_xs,marked_elec_ys, facecolors=mymarkedfc, edgecolors='b', s=same_size_sf,alpha=mymarkedalpha)
    else:
        ax.scatter(marked_elec_xs,marked_elec_ys, facecolors=mymarkedfc, edgecolors='b', s=list(ei[marked_elecs]*size_fac),alpha=mymarkedalpha)

    return marked_elec_xs,marked_elec_ys

def plotEiCols(ei,vals,thr,fig,ax,coords,size_fac=-25,cmap=cm.jet,myfc='b',vmin=0,vmax=0,noArray=False,myalpha=1,set_size=1,cbarLab='',cbarLabSize=22,labelpad=20,clip=False,hSize=100,yClip=False,lims=(0,0,0,0)):
    try:
        if vmin == vmax: #default
            vmin = np.min(vals)
            vmax = np.max(vals)
        vFlg = True
    except TypeError: #if you give a color instead
        vFlg = False
    xs = coords[:,0]
    ys = coords[:,1]
    goodInds = np.nonzero(ei<thr)[0]
    eixs = coords[goodInds,0]
    eiys = coords[goodInds,1]
    #Plot array
    if not noArray:
        ax.scatter(coords[:,0], coords[:,1], facecolors='k', edgecolors='k', s=2.5,alpha=.5,zorder=0)
    #Plot EI
    if set_size == 1:
        mys = ei[goodInds]*size_fac
    elif set_size == .5:
        mys = np.sqrt(ei[goodInds]*size_fac)
    else:
        mys = set_size
    if vFlg:
        im = ax.scatter(eixs,eiys, facecolors=myfc, edgecolors='none',s=mys,alpha=myalpha,zorder=1,c=vals[goodInds],vmin=vmin,vmax=vmax,cmap=cmap)
    else:
        im = ax.scatter(eixs,eiys, facecolors=myfc, edgecolors='none',s=mys,alpha=myalpha,zorder=1,c=vals[goodInds])

    ##Cosmetics
    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    if all(i==0 for i in lims): #no manual lims were given
        if clip:
            xLimLow = min(eixs)-hSize
            xLimHigh = max(eixs)+hSize
            yLimLow = min(eiys)-hSize
            yLimHigh = max(eiys)+hSize
        elif yClip:
            xLimLow = min(xs)-hSize
            xLimHigh = max(xs)+hSize
            yLimLow = min(eiys)-hSize
            yLimHigh = max(eiys)+hSize
        else:
            xLimLow = min(xs)-hSize
            xLimHigh = max(xs)+hSize
            yLimLow = min(ys)-hSize
            yLimHigh = max(ys)+hSize
    else: #manual lims were given
        xLimLow, xLimHigh, yLimLow, yLimHigh = lims
    ax.set_xlim([xLimLow, xLimHigh])
    ax.set_ylim([yLimLow, yLimHigh])
    ax.axis('off')

    if cbarLab:
        cbar = fig.colorbar(im, orientation='horizontal',ax=ax)
        #cbar.ax.tick_params(labelsize=15) 
        cbar.ax.tick_params(labelsize=cbarLabSize) 
        #cbar.set_label(cbarLab, rotation=270,fontsize=cbarLabSize,labelpad=labelpad)
        cbar.set_label(cbarLab,fontsize=cbarLabSize,labelpad=labelpad)

    return im,(xLimLow,xLimHigh,yLimLow,yLimHigh)

def plotEiColsRatio(eiFull,thr,fig,ax,coords,size_fac=-25,cmap=cm.jet,myfc='b',noArray=False,myalpha=1,set_size=1,clip=False,hSize=100,yClip=False,lims=(0,0,0,0),uppBound=1.6,lowBound=0.05,legendFlg=False):
    
    cDict = {'soma':'b','axon':'r','mixed':'y','dendrite':'c','error':'k'}
    ei = np.min(eiFull, axis =1)
    xs = coords[:,0]
    ys = coords[:,1]
    goodInds = np.nonzero(ei<thr)[0]
    eixs = xs[goodInds]
    eiys = ys[goodInds]

    #Plot array
    if not noArray:
        ax.scatter(coords[:,0], coords[:,1], facecolors='k', edgecolors='k', s=2.5,alpha=.5,zorder=0)
    #Plot EI
    if set_size == 1:
        mys = ei[goodInds]*size_fac
    elif set_size == .5:
        mys = np.sqrt(ei[goodInds]*size_fac)
    else:
        mys = set_size
    ##get colors
    vals = []
    compts = []
    for ee,eiVal in enumerate(ei):
        if eiVal<thr:
            compts.append(axonorsomaRatio(eiFull[ee,:],uppBound=uppBound,lowBound=lowBound))
            vals.append(cDict[compts[-1]])
    vals = np.array(vals)
    compts = np.array(compts)

    for compt in cDict.keys():
        comptInds = np.where(compts==compt)
        im = ax.scatter(eixs[comptInds],eiys[comptInds], facecolors=myfc, edgecolors='none',s=mys[comptInds],alpha=myalpha,zorder=1,
                c=[corrColDict[i] for i in vals[comptInds]],label=compt)

    ##Cosmetics
    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    if all(i==0 for i in lims): #no manual lims were given
        if clip:
            xLimLow = min(eixs)-hSize
            xLimHigh = max(eixs)+hSize
            yLimLow = min(eiys)-hSize
            yLimHigh = max(eiys)+hSize
        elif yClip:
            xLimLow = min(xs)-hSize
            xLimHigh = max(xs)+hSize
            yLimLow = min(eiys)-hSize
            yLimHigh = max(eiys)+hSize
        else:
            xLimLow = min(xs)-hSize
            xLimHigh = max(xs)+hSize
            yLimLow = min(ys)-hSize
            yLimHigh = max(ys)+hSize
    else: #manual lims were given
        xLimLow, xLimHigh, yLimLow, yLimHigh = lims
    ax.set_xlim([xLimLow, xLimHigh])
    ax.set_ylim([yLimLow, yLimHigh])
    ax.axis('off')

    #legend
    if legendFlg:
        ax.legend()


    return im,(xLimLow,xLimHigh,yLimLow,yLimHigh)

def plotEiColsHollow(ei,vals,thr,fig,ax,coords,size_fac=-25,cmap=cm.jet,myfc='b',noArray=False,myalpha=1,set_size=1,cbarLab='',cbarLabSize=22,labelpad=20,clip=False,hSize=100,yClip=False,lims=(0,0,0,0),xBound=0):
    xs = coords[:,0]
    ys = coords[:,1]
    goodInds = np.nonzero(ei<thr)[0]
    eixs = coords[np.nonzero(ei<thr)[0],0]
    eiys = coords[np.nonzero(ei<thr)[0],1]
    #Plot array
    if not noArray:
        ax.scatter(coords[:,0], coords[:,1], facecolors='k', edgecolors='k', s=2.5,alpha=.5,zorder=0)
    #Plot EI
    if set_size == 1:
        mys = ei[goodInds]*size_fac
    elif set_size == .5:
        mys = np.sqrt(ei[goodInds]*size_fac)
    else:
        mys = set_size

    if not xBound:
        im = ax.scatter(eixs,eiys, facecolors='none', edgecolors='k',s=mys,alpha=myalpha,zorder=1)
    else:
        im = ax.scatter([i for i in eixs if i > xBound],[i for ii,i in enumerate(eiys) if eixs[ii] > xBound],
                facecolors='none', edgecolors='k',s=[i for ii,i in enumerate(mys) if eixs[ii] > xBound], alpha=myalpha,zorder=1)

    ##Cosmetics
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
    if all(i==0 for i in lims): #no manual lims were given
        if clip:
            xLimLow = min(eixs)-hSize
            xLimHigh = max(eixs)+hSize
            yLimLow = min(eiys)-hSize
            yLimHigh = max(eiys)+hSize
        elif yClip:
            xLimLow = min(xs)-hSize
            xLimHigh = max(xs)+hSize
            yLimLow = min(eiys)-hSize
            yLimHigh = max(eiys)+hSize
        else:
            xLimLow = min(xs)-hSize
            xLimHigh = max(xs)+hSize
            yLimLow = min(ys)-hSize
            yLimHigh = max(ys)+hSize
    else: #manual lims were given
        xLimLow, xLimHigh, yLimLow, yLimHigh = lims
    ax.set_xlim([xLimLow, xLimHigh])
    ax.set_ylim([yLimLow, yLimHigh])
    ax.axis('off')

    if cbarLab:
        cbar = fig.colorbar(im, orientation='horizontal',ax=ax)
        #cbar.ax.tick_params(labelsize=15) 
        cbar.ax.tick_params(labelsize=cbarLabSize) 
        #cbar.set_label(cbarLab, rotation=270,fontsize=cbarLabSize,labelpad=labelpad)
        cbar.set_label(cbarLab,fontsize=cbarLabSize,labelpad=labelpad)

    return im,(xLimLow,xLimHigh,yLimLow,yLimHigh)


def plotEiColsOnly(ei,vals,thr,fig,ax,coords,size_fac=-25,cmap=cm.jet,myfc='b',vmin=0,vmax=0,noArray=False,myalpha=1,set_size=1,cbarLab='',cbarLabSize=22,labelpad=20,clip=False,hSize=100,yClip=False,lims=(0,0,0,0)):
    try:
        if vmin == vmax: #default
            vmin = np.min(vals)
            vmax = np.max(vals)
        vFlg = True
    except TypeError: #if you give a color instead
        vFlg = False
    xs = coords[:,0]
    ys = coords[:,1]
    goodInds = np.nonzero(ei<thr)[0]
    eixs = coords[np.nonzero(ei<thr)[0],0]
    eiys = coords[np.nonzero(ei<thr)[0],1]
    #Plot array
    if not noArray:
        ax.scatter(coords[:,0], coords[:,1], facecolors='k', edgecolors='k', s=2.5,alpha=.5,zorder=0)
    #Plot EI
    if set_size == 1:
        mys = ei[goodInds]*size_fac
    elif set_size == .5:
        mys = np.sqrt(ei[goodInds]*size_fac)
    else:
        mys = set_size
    if vFlg:
        #im = ax.scatter(eixs,eiys, facecolors=myfc, edgecolors='none',s=mys,alpha=myalpha,zorder=1,c=vals,vmin=vmin,vmax=vmax,cmap=cmap)
        im = ax.scatter([i for ii,i in enumerate(eixs) if vals[ii] < np.max(vals)],
                [i for ii,i in enumerate(eiys) if vals[ii] < np.max(vals)],
                facecolors=myfc, edgecolors='none',s=[i for ii,i in enumerate(mys) if vals[ii] < np.max(vals)],
                alpha=myalpha,zorder=1,
                c=[i for i in vals if i < np.max(vals)],vmin=vmin,vmax=vmax,cmap=cmap)
    else:
        im = ax.scatter(eixs,eiys, facecolors=myfc, edgecolors='none',s=mys,alpha=myalpha,zorder=1,c=vals)

    ##Cosmetics
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
    if all(i==0 for i in lims): #no manual lims were given
        if clip:
            xLimLow = min(eixs)-hSize
            xLimHigh = max(eixs)+hSize
            yLimLow = min(eiys)-hSize
            yLimHigh = max(eiys)+hSize
        elif yClip:
            xLimLow = min(xs)-hSize
            xLimHigh = max(xs)+hSize
            yLimLow = min(eiys)-hSize
            yLimHigh = max(eiys)+hSize
        else:
            xLimLow = min(xs)-hSize
            xLimHigh = max(xs)+hSize
            yLimLow = min(ys)-hSize
            yLimHigh = max(ys)+hSize
    else: #manual lims were given
        xLimLow, xLimHigh, yLimLow, yLimHigh = lims
    ax.set_xlim([xLimLow, xLimHigh])
    ax.set_ylim([yLimLow, yLimHigh])
    ax.axis('off')

    if cbarLab:
        cbar = fig.colorbar(im, orientation='horizontal',ax=ax)
        #cbar.ax.tick_params(labelsize=15) 
        cbar.ax.tick_params(labelsize=cbarLabSize) 
        #cbar.set_label(cbarLab, rotation=270,fontsize=cbarLabSize,labelpad=labelpad)
        cbar.set_label(cbarLab,fontsize=cbarLabSize,labelpad=labelpad)

    return im,(xLimLow,xLimHigh,yLimLow,yLimHigh)


def plotArrayOnly(ax, coords, size=5, myalpha=1, myfc='k', myec='k'):
    xs = coords[:,0]
    ys = coords[:,1]
    ax.scatter(xs, ys, facecolors=myfc, edgecolors=myec, s=size,alpha=myalpha)
    ax.set_xlim([min(xs)-100, max(xs)+100])
    ax.set_ylim([min(ys)-100, max(ys)+100])

def plotArrayOnlyCols(fig,ax, coords, cols, cmap=cm.jet, size=50, myalpha=1,cbarLab='',vmin=None,vmax=None,cbarFlg=True,linewidths=5):
    xs = coords[:,0]
    ys = coords[:,1]
    im = ax.scatter(xs, ys, c=cols, s=size, alpha=myalpha, cmap=cmap,vmin=vmin,vmax=vmax,linewidths=linewidths)
    ax.set_xlim([min(xs)-100, max(xs)+100])
    ax.set_ylim([min(ys)-100, max(ys)+100])
    ax.set_xticks([])
    ax.set_yticks([])
    if cbarFlg:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbarLab, rotation=270)

    return im


def plotEiStimNoStim(minei, coords, thr=-1):
    #Data
    #coords = loadCoords512()
    xs = coords[:,0]
    ys = coords[:,1]
    ##EI
    #ei = np.amin(eis[neuron_index,0], axis=1)
    goodInds = np.nonzero(minei<thr)[0]
    eixs = coords[goodInds,0]
    eiys = coords[goodInds,1]
    #Plotting
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1)
    ##Plot elecs
    ax.scatter(xs, ys, facecolors='none', edgecolors='b', s=110)
    ##Plot EI
    ax.scatter(eixs,eiys, facecolors='b', edgecolors='b', s=list(minei[goodInds]*-40),alpha=.75)
    ##Cosmetics
    ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
    ax.set_xlim([min(xs)-100, max(xs)+100])
    ax.set_ylim([min(ys)-100, max(ys)+100])
    #Return handle
    return fig,ax

def getSomaLoc3(eis, neuron_index, elec_coords, adj_mat): #UNDER CONSTRUCTION!!
    ei = np.amin(eis[neuron_index,0], axis=1)

    #Get elec list
    main_elec = np.argmin(ei)
    adj_elecs = (adj_mat['adj_mat_512'][main_elec,0] -1).flatten()
    included_elecs = np.concatenate((np.array([main_elec]), adj_elecs))

    #Take weighted sum of top three elecs only
    weightlist = []
    for e in included_elecs:
        foo = eis[neuron_index,0][e,:].flatten()
        weightlist.append(abs(np.min(foo)))
    weightlist = np.array(weightlist)/sum(weightlist)

    #Get cell coordinates
    cell_coords_x = [elec_coords[j,0] for j in included_elecs]
    cell_coords_y = [elec_coords[j,1] for j in included_elecs]

    #Compute average position
    xcoord = np.dot(cell_coords_x, weightlist)
    ycoord = np.dot(cell_coords_y, weightlist)
    return (xcoord, ycoord)



def isCloseToMidpoint(eis, neuron_index, electrodes, elec_coords, adj_mat, thr=-8, distThr = 30):
    #Get electrode positions
    c1 = elec_coords[electrodes[0] - 1]
    c2 = elec_coords[electrodes[1] - 1]

    #Get midpoint
    midx = (c1[0] + c2[0])/2
    midy = (c1[1] + c2[1])/2

    #Get soma location
    #(somax,somay) = getSomaLoc(eis, neuron_index, elec_coords, adj_mat, thr=thr)
    (somax,somay) = getSomaLoc2(eis, neuron_index, elec_coords, adj_mat)

    #See how close this is to midpoint and return if below thresh
    return getLocDist((midx,midy), (somax,somay)) < distThr

def isCloseToMidline(eis, neuron_index, electrodes, elec_coords, adj_mat, thr=-7, distThr = 300, midDistThr = 15):
    #'midDistThr' refers to tolerance of soma distance on either side of the midline here
    #Determined by visual inspection from here: http://www.falstad.com/dotproduct/
    #'distThr' is placed here to ensure the soma is not so far away as to make the stimulation noisy due to 
    #Electrically anisotropic tissue medium
    #Get electrode positions
    c1 = elec_coords[electrodes[0] - 1]
    c2 = elec_coords[electrodes[1] - 1]

    #Get midpoint
    midx = (c1[0] + c2[0])/2
    midy = (c1[1] + c2[1])/2

    #Get soma location
    #(somax,somay) = getSomaLoc(eis, neuron_index, elec_coords, adj_mat, thr=thr)
    (somax,somay) = getSomaLoc2(eis, neuron_index, elec_coords, adj_mat)

    #Get distance from midline to soma loc
    ##Get dot product of midpoint to point 
    vec1 = np.array([(c1[0]-c2[0]),(c1[1]-c2[1])])
    vec1norm = np.linalg.norm(vec1)
    vec2 = np.array([(midx - somax), (midy - somay)])
    vec2norm = np.linalg.norm(vec2)
    dp = abs(np.dot(vec1, vec2))
    ##Get angle
    angle = np.pi/2 - np.arccos(dp/(vec1norm*vec2norm))
    ##Get distance to midline
    midDist = np.sin(angle)*vec2norm

    #See how close this is to midline and return if below thresh
    return (midDist < midDistThr) and (getLocDist((midx,midy), (somax,somay)) < distThr)


#def axonorsoma(wave, mixed_diff=3): #SCRAPPED in favor of 10% threshold
def axonorsoma(wave, mixed_diff_perc=.1):

    #Get index and value of first (positive) max
    maxind1 = np.argsort(wave)[-1]
    maxval1 = np.sort(wave)[-1]
                        
    #Get index and value of (negative) min {only one}
    minind = np.argmin(wave)
    minval = np.min(wave)

    #If positive value is larger than negative value, automatically classify as dendrite
    ##ASSUMES SIGNAL STARTS AT 0, which is true for Rig 1 system but doesn't generalize
    if maxval1 > abs(minval): return 'dendrite'
                                        
    #Check on the other side of minimum peak for other max
    if minind > maxind1:
        maxind2 = np.argsort(wave[minind:])[-1]
        maxval2 = np.sort(wave[minind:])[-1]
    else:
        try:
            maxind2 = np.argsort(wave[:minind])[-1]
            maxval2 = np.sort(wave[:minind])[-1]
        except IndexError: #captures really weird waveforms where minind is first
            return 'messy'
    #If max values are close together, classify as 'mixed'
    #print(str(maxval1) + ' ' + str(maxval2) + ' ' + str(minval ) + ' ' + str(mixed_diff_perc))
    if np.abs(maxval1 - maxval2) < np.abs(minval)*mixed_diff_perc: return 'mixed'

    #If max peak occurs before min peak, classify as axon. Else soma
    if maxind1 < minind: return 'axon'
    else: return 'soma'

def axonorsomaRatio(wave,uppBound=1.6,lowBound=0.05):
    try:
        #Get index and value of (negative) min {only one}
        minind = np.argmin(wave)
        minval = np.min(wave)

        #Get max vals to either side of min
        maxvalLeft = np.max(wave[0:minind])
        maxvalRight = np.max(wave[minind:])

        if np.abs(minval) < max(maxvalLeft,maxvalRight):
            rCompt = 'dendrite'
        else:
            if maxvalRight == 0:
                ratio = 0
            else:
                ratio = maxvalLeft/maxvalRight
            if ratio > uppBound:
                rCompt = 'axon'
            elif ratio < lowBound: #FUDGED
                rCompt = 'soma'
            else:
                rCompt = 'mixed'
    except ValueError:
        rCompt = 'error' #wave is abnormally shaped (usually has min at leftmost or rightmost point)

    return rCompt

def calcRatio(wave):
    try:
        #Get index and value of (negative) min {only one}
        minind = np.argmin(wave)
        minval = np.min(wave)

        #Get max vals to either side of min
        maxvalLeft = np.max(wave[0:minind])
        maxvalRight = np.max(wave[minind:])

        if maxvalRight == 0:
            ratio = 0
        else:
            ratio = maxvalLeft/maxvalRight

    except ValueError:
        ratio = 0 #good value to set?? #wave is abnormally shaped (usually has min at leftmost or rightmost point)

    return ratio

def axonorsoma_findPosDiff(wave):

    #Get index and value of (negative) min {only one}
    minind = np.argmin(wave)
    minval = np.min(wave)

    #Get index and value of first (positive) max on left side of min
    try:
        maxind1 = np.argsort(wave[:minind])[-1]
        maxval1 = np.sort(wave[:minind])[-1]
    except IndexError: #captures really weird waveforms where minind is first
        return 0

    #Get index and value of second (positive) max on right side of min
    maxind2 = np.argsort(wave[minind:])[-1]
    maxval2 = np.sort(wave[minind:])[-1]

    return maxval1 - maxval2


    #If positive value is larger than negative value, automatically classify as dendrite
    ##ASSUMES SIGNAL STARTS AT 0, which is true for Rig 1 system but doesn't generalize
    ##Give this one 0 difference since doesn't mean anything
    if maxval1 > abs(minval): return 0

#New soma loc comp
def getSomaLocWave(eis,cell,elec_coords,mixed_diff_perc=.1,eithr=-2):
    ei = np.amin(eis[cell,0], axis=1)
    goodInds = np.nonzero(ei<eithr)[0]
    soma_elecs = []
    failed_waves = []

    for goodInd in goodInds:
        wave = eis[cell,0][goodInd,:].flatten() 
        try: 
            if axonorsoma(wave,mixed_diff_perc=mixed_diff_perc) in ['soma']: soma_elecs.append(goodInd)
        except IndexError:
            failed_waves.append(wave)

    #Compute centroid
    weightlist = ei[soma_elecs]
    weightlist = np.array(weightlist)/sum(weightlist)
    #Get cell coordinates
    soma_elec_coords_x = [elec_coords[j,0] for j in soma_elecs]
    soma_elec_coords_y = [elec_coords[j,1] for j in soma_elecs]
    #Compute average position
    xcoord = np.dot(soma_elec_coords_x, weightlist)
    ycoord = np.dot(soma_elec_coords_y, weightlist)

    return (xcoord,ycoord),soma_elecs,failed_waves

def getSomaLocWaveRatio(eis,cell,elec_coords,uppBound=1.6,lowBound=0.05,eithr=-2):
    ei = np.amin(eis[cell,:,:],axis=1)
    goodInds = np.nonzero(ei<eithr)[0]
    soma_elecs = []
    failed_waves = []

    for goodInd in goodInds:
        #wave = eis[cell,0][goodInd,:].flatten() 
        wave = eis[cell,goodInd,:]

        try: 
            if axonorsomaRatio(wave,uppBound=uppBound,lowBound=lowBound) in ['soma']: soma_elecs.append(goodInd)
        except IndexError:
            failed_waves.append(wave)

    #Compute centroid
    weightlist = ei[soma_elecs]
    weightlist = np.array(weightlist)/sum(weightlist)
    #Get cell coordinates
    soma_elec_coords_x = [elec_coords[j,0] for j in soma_elecs]
    soma_elec_coords_y = [elec_coords[j,1] for j in soma_elecs]
    #Compute average position
    xcoord = np.dot(soma_elec_coords_x, weightlist)
    ycoord = np.dot(soma_elec_coords_y, weightlist)

    return (xcoord,ycoord),soma_elecs,failed_waves

def getSomaLocWaveRatioLean(ei,elec_coords,uppBound=1.6,lowBound=0.05,eithr=-2):
    goodInds = np.nonzero(np.amin(ei,axis=1)<eithr)[0]
    soma_elecs = []
    failed_waves = []

    for goodInd in goodInds:
        wave = ei[goodInd,:].flatten() 
        try: 
            if axonorsomaRatio(wave,uppBound=uppBound,lowBound=lowBound) in ['soma']: soma_elecs.append(goodInd)
        except IndexError:
            failed_waves.append(wave)

    #Compute centroid
    weightlist = np.min(ei[soma_elecs,:],axis=1)
    weightlist = np.array(weightlist)/sum(weightlist)
    #Get cell coordinates
    soma_elec_coords_x = [elec_coords[j,0] for j in soma_elecs]
    soma_elec_coords_y = [elec_coords[j,1] for j in soma_elecs]
    #Compute average position
    xcoord = np.dot(soma_elec_coords_x, weightlist)
    ycoord = np.dot(soma_elec_coords_y, weightlist)

    return (xcoord,ycoord),soma_elecs,failed_waves

def getAxonDir(mypoint, eis, neuron_index, elec_coords, adj_mat, thr=-1,mixedDiff=3):
    ei_wave = eis[neuron_index,0] #NOT flattened, contains time data
    ei = np.amin(ei_wave, axis = 1)

    main_elec = np.argmin(ei)
    main_elec_coords = elec_coords[main_elec,:]

    #Get list of electrodes above threshold and filter
    elecinds = np.nonzero(ei < thr)[0]

    #Figure out which has axon waveform
    axinds = []
    if list(elecinds):
        for e in elecinds:
            if axonorsoma(ei_wave[e,:],mixed_diff=mixedDiff) == 'axon':
                axinds.append(e)

    #Get list of cell elec coordinates
    cell_coords_x = [elec_coords[j,0] for j in axinds]
    cell_coords_y = [elec_coords[j,1] for j in axinds]

    #Get end points wrt main elec
    som_dists = []
    for a in range(len(axinds)):
        som_dists.append(getLocDist(main_elec_coords,[cell_coords_x[a],cell_coords_y[a]]))
    try:
        maxind = np.argmax(som_dists)
        minind = np.argmin(som_dists)
    except ValueError:
        #print(som_dists)
        return 2 #beyond normalized

    #Get NORMALIZED dot product between vectors to 1 - 'mypoint' and 2 - min to max
    vec1 = np.array(mypoint) - np.array([cell_coords_x[minind],cell_coords_y[minind]])
    vec2 = np.array([cell_coords_x[maxind],cell_coords_y[maxind]]) - np.array([cell_coords_x[minind],cell_coords_y[minind]])
    return np.dot(vec1/np.linalg.norm(vec1),vec2/np.linalg.norm(vec2))


def getAxonFit(eis, neuron_index, elec_coords, adj_mat, noise, num_sigmas=.5, pdeg = 4, distThr = 175):

    #ei = np.amin(eis[neuron_index,0], axis=1)
    ei = np.amin(eis[neuron_index,:,:],axis=1)
    c = elec_coords

    if False: #old method
        main_elec = np.argmin(ei)
        #adj_elecs = (adj_mat['adj_mat_512'][main_elec,0] -1).flatten()
        adj_elecs = (adj_mat[main_elec,0] -1).flatten()
        excluded_elecs = np.concatenate((np.array([main_elec]), adj_elecs))
        #Get list of electrodes above threshold and filter
        #elecinds = np.nonzero(ei < thr)[0]

        # More robust method.
        elecinds = np.argwhere(np.abs(ei) > num_sigmas * noise).flatten()
        elecinds = [j for j in elecinds if j not in excluded_elecs]

    #new method!
    #somaLoc,_,_ = getSomaLocWaveRatio(eis,neuron_index,elec_coords)
    somaLoc = getSomaLoc2(eis, neuron_index, elec_coords, adj_mat,numElecs=7)

    #Get list of electrodes above threshold and filter by distance from soma center
    #elecinds = np.nonzero(ei < thr)[0]

    # More robust method.
    elecinds = np.argwhere(np.abs(ei) > num_sigmas * noise).flatten()
    elecinds = [j for j in elecinds if getLocDist(c[j,:],somaLoc) > distThr]

    #Get list of cell elec coordinates
    cell_coords_x = [c[j,0] for j in elecinds]
    cell_coords_y = [c[j,1] for j in elecinds]

    #Determine whether x or y direction has larger range
    try: 
        if abs(max(cell_coords_x) - min(cell_coords_x)) > abs(max(cell_coords_y) - min(cell_coords_y)):
            dirflg = 'x'
        else:
            dirflg = 'y'
    except ValueError:
        return (0,0,0,0)

    #Get list of ei values for these coordinates
    cell_ei = np.abs([ei[j] for j in elecinds])

    #Fit polynomial
    if dirflg == 'x':
        fitx = cell_coords_x
        fity = cell_coords_y
    else:
        fitx = cell_coords_y
        fity = cell_coords_x

    p = np.poly1d(np.polyfit(fitx, fity, pdeg, w=cell_ei))
    pder = np.polyder(p)

    return(p, pder, dirflg, fitx) #also return x coords in range

def runsInBetween(eis, neuron_index, electrodes, elec_coords, adj_mat, disttolerance = 7, dottolerance = .25, thr = -1, pdeg = 4):
    #Tolerance means on either side of midpoint

    #Get axon fit
    (p, pder, dirflg, fitx) = getAxonFit(eis, neuron_index, elec_coords, adj_mat, thr = thr, pdeg = pdeg)
    if not p: return False
    
    #Get coordinates of electrodes
    c1 = elec_coords[electrodes[0] - 1]
    c2 = elec_coords[electrodes[1] - 1]

    #Switch x and y if axon is dependent on y
    if dirflg == 'y':
        foo = c1[0]
        c1[0] = c1[1]
        c1[1] = foo
        foo = c2[0]
        c2[0] = c2[1]
        c2[1] = foo

    minx = min([c1[0],c2[0]])
    miny = min([c1[1],c2[1]])
    maxx = max([c1[0],c2[0]])
    maxy = max([c1[1],c2[1]])


    ##1. Check if axon is in between the two electrodes
    #Get midpoint of two electrodes 
    midx = abs(c1[0] - c2[0])/2 + minx
    midy = abs(c1[1] - c2[1])/2 + miny

    #Get axon point closest to midpoint by generating finely spaced points
    numpts = 200
    #axonx = np.linspace(minx,maxx,numpts)
    axonx = np.linspace(minx-10,maxx+10,numpts) #TEMPORARY?
    axony = p(axonx)

    distchkflg = False
    bestdist = 100000
    for i in range(numpts):
        dist = np.sqrt(abs((midy - axony[i])**2) + abs((midx - axonx[i])**2)) 
        if dist < disttolerance:
            distchkflg = True
            #Store closest point
            if dist < bestdist:
                closestx = axonx[i]
                bestdist = dist

    #plt.scatter([c1[0],c2[0]],[c1[1],c2[1]])
    #plt.plot(axonx,axony)
    #plt.axis('equal')
    #plt.show()

    #print(bestdist)
    #If distance criteria was not met, get outta here!
    if not distchkflg: return False

    ##2. Check if axon is perpendicular
    slope = pder(closestx)
    vec1 = np.array([(c1[0]-c2[0]),(c1[1]-c2[1])])
    vec2 = np.array([1,slope])
    dp = abs(np.dot(vec1/np.linalg.norm(vec1), vec2/np.linalg.norm(vec2)))

    #print(dp)
    
    if dp < dottolerance:
        #plt.scatter([c1[0],c2[0]],[c1[1],c2[1]])
        #plt.plot(axonx,axony)
        #plt.axis('equal')
        #plt.show()
        return True
    else:
        return False

#def getEIStats(eipath):
    #Takes eipath or eis
    ##Check if string or array
    #if eipath
    #eis = st.loadEis(eipath)
    
    

def loadEis(ei_path): #copied from stimlib, beware of name conflicts!
    eis = loadmat(ei_path)['datarun'][0,0]['ei']['eis'][0,0]
    return eis

def loadSpikes(ei_path):
    spikes = loadmat(ei_path)['datarun'][0,0]['spikes']
    return spikes

def loadEiCellIds(ei_path): #copied from stimlib, beware of name conflicts!           
    cellIds = list(loadmat(ei_path)['datarun'][0,0]['cell_ids'][0,:])
    return cellIds

def loadCelltypes(ct_path):

    foo = loadmat(ct_path)['celltypes']
    celltypes = [i[0] for i in foo[0,:]]
    return foo

def getClass(celltypes, cell_index):
    celltype = celltypes[cell_ind].lower()
    ct_color = 'w'
    if 'on' in celltype and 'parasol' in celltype: ct_color = 'b'
    if 'off' in celltype and 'parasol' in celltype: ct_color = 'g'
    if 'on' in celltype and 'midget' in celltype: ct_color = 'r'
    if 'off' in celltype and 'midget' in celltype: ct_color = 'y'
    return ct_color
    #Sort of arbitrary color coding, maybe use named tuple? 

def loadNoise(noisepath):
    noise = []
    with open(noisepath,'r') as f:
        for row in f:
            noise.append(float(row.strip()))
    return noise


def plotElecArray():
    coords = loadCoords512()
    xs = coords[:,0]
    ys = coords[:,1]
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(1,1,1)
    plt.scatter(xs, ys, facecolors='none', edgecolors='b', s=110)
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
    ax.set_xlim([min(xs)-100, max(xs)+100])
    ax.set_ylim([min(ys)-100, max(ys)+100])
    #Add electrode labels
    for cc,c in enumerate(coords):
        ax.text(c[0],c[1],str(cc+1), fontweight='bold')
    plt.savefig('elecarray.png')
    plt.close('all')

def plotElecArray_chip(chipNumber):
    coords = loadCoords512()
    chipElecs = loadChipElecs512()
    myElecs = chipElecs[chipNumber-1]
    #Make facecolors vec
    fcVec = ['none' for i in range(512)]
    for e in myElecs:
        fcVec[e-1] = 'r'
    xs = coords[:,0]
    ys = coords[:,1]
    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(1,1,1)
    plt.scatter(xs, ys, facecolors=fcVec, edgecolors='b', s=110)
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
    ax.set_xlim([min(xs)-100, max(xs)+100])
    ax.set_ylim([min(ys)-100, max(ys)+100])
    #Add electrode labels
    for cc,c in enumerate(coords):
        ax.text(c[0],c[1],str(cc+1), fontweight='bold')
    plt.savefig('elecarray_chip' + str(chipNumber) + '.png')
    plt.close('all')

def plothistline(ax,data,bins=None,normed=False,line_alpha=1,hist_alpha=0,linewidth=5,label=None,line_color=None):
    n, binsout, patches = ax.hist(data,bins=bins,normed=normed,alpha=hist_alpha)
    bwidth = np.diff(binsout)[0]/2
    l = ax.plot(binsout[1:]-bwidth,n,linewidth=linewidth,alpha=line_alpha,label=label,color=line_color)

def loadNoise(noisePath):
    noises = []
    with open(noisePath,'r') as f:
        for row in f:
            noises.append(float(row.strip()))

    return noises[1:] #don't return first ttl channel

def loadBundleThrs(bThrRegexp,piece,data,ceilVal=0):
    #bThrRegexp = '/Volumes/Analysis/'+piece+'Bundle_PulkitAlg_Sasi_aug19_*'
    pulDir = glob.glob(bThrRegexp)
    pulkitBund = loadmat(pulDir[0]+ '/SafeZone_Indices.mat')['SafeZone_Indices'].flatten()
    ##find any amps in dataset
    amps = np.array([0,0])
    for cell in data[piece].keys(): 
        for elec in range(512):
            try:
                amps = data[piece][cell]['stim'][str(elec+1)]['curve'][1,:]
                break
            except KeyError:
                continue
        if np.any(amps): break

    bThrs = []
    for i in pulkitBund:
        if i > len(amps): #max
            #bThrs.append(6) #JANKY
            bThrs.append(ceilVal) #to identify 
        else:
            bThrs.append(amps[i-1])

    
    return bThrs

def findMisalignedEI(ei,eiValThr,somaBound=-1,dendBound=-50):
    skipFlg = False
    for elec in range(ei.shape[0]):
        wave = ei[elec,:]
        if np.min(wave) < -eiValThr: #only judge real waves
            testRatio = calcRatio(wave)
            if testRatio < somaBound and testRatio > dendBound: #big are dendrites
                skipFlg = True

    return skipFlg

def find_nearest_idx(array, value): #for general convenience
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    #return array[idx]
    return idx
