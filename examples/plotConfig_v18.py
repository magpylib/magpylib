# -*- coding: utf-8 -*-

#%% IMPORTS
from matplotlib import rc
import matplotlib.spines
from gxPackage_v03 import CTRlegendInstances
import matplotlib.pyplot as plt



#%% SETTINGS

# FS SETTINGS---------------------------------
FS = 18
rc('font', size=FS)
titleFS = FS
labelFS = FS-2
tickFS = FS-2


# FONT SETTINGS-------------------------------
# set LATEX NATIVE font
if False:
	matplotlib.rc('font', family='serif', serif='cm10')
	rc('text', usetex=True)
	matplotlib.rcParams['text.latex.preamble']=[
        r'\usepackage{amsfonts}'
        r'\usepackage{amsmath}']


# COLOR SETTINGS -----------------------------
frameCOL = '.4'
tickCOL = '.2'
axFaceCOL = '.99'
gridCol_major = '.93'
gridCol_minor = '.96'


# LW Settings--------------------------------
frameLW = .6
gridLW_major = .6
gridLW_minor = .4
tickLW = .6


# LABEL padding------------------------------
changeXpad = False
xpad = 20
changeYpad = False
ypad = 20


# LEGENDS------------------------------------
legProps = dict(fancybox=False,shadow=False, labelspacing = .1,handletextpad=0.2,columnspacing=0.6)
legTitleFS = FS-5
legFramelinewidth = 0.8
legFrameColor = '.6'
legFramealpha = 0.5
legFaceColor = (.99,.99,.99,.7)
'''
loc=None, numpoints=None, markerscale=None, scatterpoints=None, scatteryoffsets=None,
prop=None, fontsize=None, borderpad=None, labelspacing=None, handlelength=None,
handleheight=None, handletextpad=None, borderaxespad=None, columnspacing=None,
ncol=1, mode=None, fancybox=None, shadow=None, title=None, framealpha=None,
bbox_to_anchor=None, bbox_transform=None, frameon=None, handler_map=None
'''

# COLORBAR ----------------------------------
cbProps = dict(fsTix=FS-2, fs=FS)





#%%plotstyle function
def plotstyle(AXS_with_STYLE, AXS_with_GRID): #plotstyle is packed as a function that must be called after completing the plot

    for ax in AXS_with_STYLE:

        #set axes facecolor
        ax.patch.set_facecolor(axFaceCOL)

        #adjust frame
        for child in ax.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color(frameCOL)
                child.set_linewidth(frameLW)

        #set tick styles
        for tick in ax.xaxis.get_ticklabels()+ax.yaxis.get_ticklabels():
            tick.set_color(tickCOL)
            tick.set_fontsize(tickFS)

        #label padding
        if changeXpad:
            ax.xaxis.labelpad = xpad
        if changeYpad:
            ax.yaxis.labelpad = ypad

        #set fontsizes of title
        ax.title.set_size(titleFS)
        ax.xaxis.label.set_fontsize(labelFS)
        ax.yaxis.label.set_fontsize(labelFS)

        #restyle the tick lines
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(5)
            line.set_markeredgewidth(tickLW)
            line.set_color(frameCOL)
    
    
    #set grid and axes styles
    for ax in AXS_with_GRID:
        ax.grid(which='major',lw=gridLW_major,color=gridCol_major,ls='-')
        ax.minorticks_on()
        ax.grid(which='minor',lw=gridLW_minor,color=gridCol_minor,ls='-')
        #ax.axhline(0, color = '.7', lw=gridLW, zorder=0)
        #ax.axvline(0, color = '.7', lw=gridLW, zorder=0)


    #more legend properties
    for leg in CTRlegendInstances:
        leg.get_frame().set_linewidth(legFramelinewidth)
        leg.get_frame().set_edgecolor(legFrameColor)
        leg.get_frame().set_facecolor(legFaceColor)
        leg.get_title().set(fontsize=legTitleFS)

    #make annotations draggable
    for child in ax.get_children():
        if isinstance(child, plt.Annotation):
            child.draggable()

