# -*- coding: utf-8 -*-

#%% IMPORTS
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.pyplot import Line2D

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from numpy import array

from matplotlib.colors import LinearSegmentedColormap


#%% BASE COLORS
def rgb(r,g,b):
    return (float(r)/255,float(g)/255,float(b)/255)
WHITE = rgb(255,255,255)
BLACK = rgb(0,0,0)
PINK = rgb(255,20,147)
DDYELLOW, DYELLOW, YELLOW, LYELLOW, LLYELLOW = rgb(215,155,0), rgb(240,190,0), rgb(255,225,0), rgb(255,255,50), rgb(255,255,130)
DDRED, DRED, RED, LRED, LLRED = rgb(170,0,0), rgb(212,0,0), rgb(255,0,0), rgb(255,85,85), rgb(255,160,160)
DDORANGE, DORANGE, ORANGE, LORANGE, LLORANGE = rgb(190,75,0), rgb(212,85,0), rgb(255,102,0), rgb(255,153,85), rgb(255,190,150)
DDBROWN, DBROWN, BROWN, LBROWN, LLBROWN = rgb(85,34,0), rgb(110,50,20), rgb(150,75,37), rgb(192,96,48), rgb(211,141,95)
DDGREEN, DGREEN, GREEN, LGREEN, LLGREEN = rgb(0,80,0), rgb(0,128,0), rgb(0,170,0), rgb(0,255,0), rgb(120,255,120)
DDBLUE, DBLUE, BLUE, LBLUE,LLBLUE = rgb(0,0,110), rgb(0,0,200), rgb(15,70,255), rgb(30,144,255), rgb(75,180,255)
DDOLIVE, DOLIVE, OLIVE, LOLIVE, LLOLIVE = rgb(64,64,0), rgb(96,96,0), rgb(128,128,0), rgb(175,175,30),rgb(205,222,135)
DDCYAN, DCYAN, CYAN, LCYAN, LLCYAN = rgb(0,150,150), rgb(0,200,200), rgb(0,255,255), rgb(130,255,255), rgb(200,255,255)
DDPINK, DPINK, PINK, LPINK, LLPINK = rgb(150,0,70), rgb(200,20,100), rgb(255,20,147), rgb(255,100,175), rgb(255,170,204)
DDTEAL, DTEAL, TEAL, LTEAL, LLTEAL = rgb(0,80,80), rgb(0,120,120), rgb(0,169,157), rgb(0,197,205), rgb(0,245,255)
DDPURPLE, DPURPLE, PURPLE, LPURPLE, LLPURPLE = rgb(93,71,139), rgb(130,58,180), rgb(160,48,240), rgb(180,82,205), rgb(224,102,255)
COL = ['cyan', 'teal', 'blue', 'maroon', 'red', 'fuchsia', 'purple', 'green', 'lime', 'olive', 'orange', 'gold', 'silver', 'gray', 'black']




#%% COLOR MAPS
def createCM(name, *args):
    N = len(args)
    data = [[(1.*i/(N-1),args[i][jj],args[i][jj]) for i in range(N)] for jj in range(3)]
    dat = { 'red': data[0],
            'green': data[1],
            'blue': data[2]}
    return LinearSegmentedColormap(name, dat)

RAINBOW = plt.cm.rainbow # matplotlib stuff pylint: disable=no-member
JET = plt.cm.jet # matplotlib stuff pylint: disable=no-member
SPEKTRUM = createCM('spektrum', DPURPLE,RED,DORANGE,YELLOW,LOLIVE,LGREEN,CYAN,BLUE)
LSPEKTRUM = createCM('spektrum', PURPLE,LRED,ORANGE,LYELLOW,LLOLIVE,LLGREEN,LCYAN,LBLUE)
SPEKTRUM_r = createCM('spektrum',BLUE,CYAN,LGREEN,LOLIVE,YELLOW,DORANGE,RED, DPURPLE)
LSPEKTRUM_r = createCM('spektrum',LBLUE,LCYAN,LLGREEN,LLOLIVE,LYELLOW,ORANGE,LRED,PURPLE)

RG = createCM('RG', (1,0,0), (0,1,0))
GR = createCM('RG', (0,1,0), (1,0,0))
RB = createCM('RB', (1,0,0), (0,0,1))
BR = createCM('BR', (0,0,1), (1,0,0))
GB = createCM('GB', (0,1,0), (0,0,1))
BG = createCM('BG', (0,0,1), (0,1,0))
RGB = createCM('RGB', LRED,GREEN,LBLUE)




#%% LEGEND
CTRlegendInstances = [] # global list of all legends for property adjustment

#legend class
class CTRlegend():
    def __init__(self,**kwargs):
        self.kwa = kwargs
        self.signatures = []
        self.labels = []
        #print('CTRlegend created')

    def addLine(self,label,**kwargsLine):
        self.labels.append(label)
        self.signatures.append(Line2D([],[],**kwargsLine))
    def show(self,ax):
        leg = ax.legend(self.signatures,self.labels,**self.kwa)#'upper center', bbox_to_anchor=self.pos, fancybox=True, shadow=True, ncol=self.ncol,prop={'size':self.fs})
        leg.draggable(state=True, use_blit=True)
        global CTRlegendInstances
        CTRlegendInstances.append(leg)
        ax.add_artist(leg)

#vertical color bar
def CTRcbVert(ax,cmap = mpl.cm.winter,vals = [-1,1], fsTix = 12, label='', fs = 14):  # matplotlib stuff pylint: disable=no-member
    norm99 = mpl.colors.Normalize(vmin=vals[0],vmax=vals[1])
    cbb99 = mpl.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm99,orientation='vertical')
    plt.setp(plt.getp(cbb99.ax.axes,'yticklabels'),fontsize=fsTix)
    cbb99.set_label(label,fontsize=fs)

#horizontal color bar
def CTRcbHor(ax,cmap = mpl.cm.winter,vals = [-1,1], fsTix = 12, label='', fs = 14): # matplotlib stuff pylint: disable=no-member
    norm99 = mpl.colors.Normalize(vmin=vals[0],vmax=vals[1])
    cbb99 = mpl.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm99,orientation='horizontal')
    plt.setp(plt.getp(cbb99.ax.axes,'xticklabels'),fontsize=fsTix)
    cbb99.set_label(label,fontsize=fs)


#%% 3D Graphics

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class setup3Dplot():
    def __init__(self,size):
        fig = plt.figure(figsize=(7,6),facecolor=WHITE, dpi=130)
        fig.subplots_adjust(left=.1, bottom=.3, right=.88,top=.9, wspace=.5,hspace=.5)
        self.ax = fig.gca(projection='3d')
        self.ax.set(aspect=1.,xlim =(-size,size),ylim =(-size,size),zlim =(-size,size))


    def plotPolyCollection(self,pollies,COLS):
        pollies = array(pollies,dtype = float)
        coll = Poly3DCollection(pollies,facecolors=COLS,lw=0.5)
        self.ax.add_collection3d(coll)

    def plotVector(self,v1,v2,color,**kwargs):
        aa = Arrow3D([v1[0],v2[0]],[v1[1],v2[1]],[v1[2],v2[2]], mutation_scale=15, arrowstyle="-|>", color=color, shrinkA=0, shrinkB=0,**kwargs)
        self.ax.add_artist(aa)

    def plotPoint(self,v1,**kwargs):
        self.ax.plot([v1[0]],[v1[1]],[v1[2]],**kwargs)