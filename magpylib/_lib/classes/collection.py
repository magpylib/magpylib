######### Type hint definitions ########
# These aren't type hints, but look good 
# in Spyder IDE. Pycharm recognizes it.
from typing import Tuple
x=y=z=0.0 # Position Vector
#######################################
#%% IMPORTS

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import array,amax, linspace, pi, sin, cos, finfo
from magpylib._lib.classes.magnets import Box,Cylinder,Sphere
from magpylib._lib.classes.currents import Line, Circular
from magpylib._lib.classes.moments import Dipole
from magpylib._lib.utility import isSource, drawCurrentArrows, drawMagAxis, drawDipole
from magpylib._lib.mathLibPrivate import angleAxisRotation, fastNorm3D
from magpylib._lib.mathLibPublic import rotatePosition


class Collection():
    """
    Create a collection of :mod:`magpylib.source` objects for common manipulation.
    
    Parameters
    ----------
    sources : source objects
        python magic variable passes source objects to the collection at initialization.
    
    Attributes
    ----------
    sources : list of source objects
        List of all sources that have been added to the collection.
        
    Example
    -------
        >>> from magpylib import source, Collection
        >>> pm1 = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1])
        >>> pm2 = source.magnet.Cylinder(mag=[0,0,1000],dim=[1,1])
        >>> pm3 = source.magnet.Sphere(mag=[0,0,1000],dim=1)
        >>> col = Collection(pm1,pm2,pm3)
        >>> B = col.getB([1,0,1])
        >>> print(B)
        [9.93360625e+01 1.76697482e-14 3.12727683e+01]
    """
    
    def __init__(self,*sources,marker=[x,y,z]):
        
        assert len(marker) == 3, "Position vector for marker is not 3"
        assert all(type(p)==int or type(p)==float for p in marker), "Position vector for marker has non-int or non-float types."
        assert all(isSource(a) for a in sources), "Non-source object in Collection initialization"

        self.sources = []
        self.markers = marker
        for s in sources:
            if type(s) == Collection:
                self.sources.extend(s.sources)
            else:
                self.sources += [s]

    def popSource(self,source=None,index=-1):
        """
        Pop a source from the sources list. 
        
        Parameters
        ----------

        source : source object 
            [Optional] Remove the inputted source from the list

        index : int
            [Optional] Remove a source at the given index position. Default: Last position.
            Has no effect is source kwarg is used
        
        Return
        ------

        Popped source object.

        Raises
        ------

        ValueError
            Will be thrown if you attempt to remove a source that is not in the Collection.
        
        AssertionError
            Will be thrown if inputted index kwarg type is not type int

        Example
        -------

            >>> from magpylib import Collection, source
            >>> s = source.magnet.Sphere(mag=[1,2,3],dim=1,pos=[3,3,3])
            >>> s2 = source.magnet.Sphere(mag=[1,2,3],dim=2,pos=[-3,-3,-3])
            >>> m = source.moment.Dipole(moment=[1,2,3],pos=(0,0,0))
            >>> c = Collection(s,s2,m)
            >>> print(c.sources)
            [<magpylib._lib.classes.magnets.Sphere object at 0xa31eafcc>, 
            <magpylib._lib.classes.magnets.Sphere object at 0xa31ea1cc>, 
            <magpylib._lib.classes.moments.Dipole object at 0xa31ea06c>]
            >>> c.popSource(source=s)
            >>> print(c.sources)
            [<magpylib._lib.classes.magnets.Sphere object at 0xa31ea1cc>, 
            <magpylib._lib.classes.moments.Dipole object at 0xa31ea06c>]
            >>> c.popSource(source=s2)
            >>> print(c.sources)
            [<magpylib._lib.classes.moments.Dipole object at 0xa31ea06c>]
            >>> c.popSource()
            >>> print(c.sources)
            []
            


        """
        assert type(index) == int, "Index type in pop must be integer"

        if source is None:
                return self.sources.pop(index)
        else:

            try:
                self.sources.remove(source)
            except ValueError as e: # Give a more helpful message.
                raise type(e)(str(e) + ' - ' + str(type(source)) + ' not in list for popSource')
            return source

    def marker(self,markerPos=[x,y,z]):
        """
        Set/Update mirror marker position for display system. Default: [0,0,0]
        
        Parameters
        ----------
        markerPos : int/float list
            Position for mirrored markers (the default is [0,0,0], which shows only one marker at the center)
        
        Raise
        -----
        AssertionError
            List is not a position vector made of integers or floats.

        """
        assert len(markerPos) == 3, "Position vector for marker is not 3"
        assert all(type(p)==int or type(p)==float for p in markerPos), "Position vector for marker has non-int or non-float types."
        self.markers=markerPos

    def addSource(self,source):
        """
        This method adds the argument source object to the collection.
        
        Parameters
        ----------
        source : source object
            adds the source object `source` to the collection.
        
        Returns
        -------
        None
            
        Example
        -------
        >>> from magpylib import source, Collection
        >>> pm1 = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1])
        >>> pm2 = source.magnet.Cylinder(mag=[0,0,1000],dim=[1,1])
        >>> pm3 = source.magnet.Sphere(mag=[0,0,1000],dim=1)
        >>> col = Collection(pm1)
        >>> print(col.getB([1,0,1]))
          [4.29223532e+01 1.76697482e-14 1.37461635e+01]
        >>> col.addSource(pm2)
        >>> print(col.getB([1,0,1]))
          [7.72389756e+01 1.76697482e-14 2.39070726e+01]
        >>> col.addSource(pm3)
        >>> print(col.getB([1,0,1]))
          [9.93360625e+01 1.76697482e-14 3.12727683e+01]
        """        
        self.sources += [source]

    def merge(self,collection):
        """
        Merge another Collection object's list with this Collection's list.

        Note
        ----

        The reference to previous Collections are maintained. Modifying their objects
        will modify them in the merged Collection.

        Parameter
        ---------
        
        Collection of objects to be merged with this Collection.

        Returns
        -------

        Self

        Raise
        -----

        AssertionError
            If input is not a Collection
        
        Example
        -------

        >>> from magpylib import Collection, source
        >>> s0 = source.magnet.Sphere(mag=[1,2,3],dim=1,pos=[1,3,1])
        >>> b0 = source.magnet.Box(mag=[1,2,3],dim=[1,1,1],pos=[3,1,3])
        >>> c0 = source.magnet.Cylinder(mag=(1,1,1),dim=[2,3])
        >>> s = source.magnet.Sphere(mag=[1,2,3],dim=1,pos=[3,3,3])
        >>> s2 = source.magnet.Sphere(mag=[1,2,3],dim=2,pos=[-3,-3,-3])
        >>> m = source.moment.Dipole(moment=[1,2,3],pos=(-1,-1,-1))
        >>> col1 = Collection(s,s2,m)
        >>> print(col1.sources)
            [<magpylib._lib.classes.magnets.Sphere object at 0xa2c626cc>, 
            <magpylib._lib.classes.magnets.Sphere object at 0xa2c6246c>, 
            <magpylib._lib.classes.moments.Dipole object at 0xa2c62d0c>]
        >>> col2 = Collection(s0,b0,c0)
        >>> print(col2.sources)
            [<magpylib._lib.classes.magnets.Sphere object at 0xa2ce416c>, 
            <magpylib._lib.classes.magnets.Box object at 0xa2c6270c>, 
            <magpylib._lib.classes.magnets.Cylinder object at 0xa2c6258c>]
        >>> col1.merge(col2)
        >>> print(col1.sources)
            [<magpylib._lib.classes.magnets.Sphere object at 0xa2c626cc>, 
            <magpylib._lib.classes.magnets.Sphere object at 0xa2c6246c>, 
            <magpylib._lib.classes.moments.Dipole object at 0xa2c62d0c>, 
            <magpylib._lib.classes.magnets.Sphere object at 0xa2ce416c>, 
            <magpylib._lib.classes.magnets.Box object at 0xa2c6270c>, 
            <magpylib._lib.classes.magnets.Cylinder object at 0xa2c6258c>]
        """
        assert type(collection) == Collection, "Attempted Collection merge with non-Collection type."
        self.sources.extend(collection.sources)

        return self


    def getB(self,pos):
        """
        This method returns the magnetic field vector generated by the whole
        collection at the argument position `pos` in units of [mT]
        
        Parameters
        ----------
        pos : vec3 [mm]
            Position where magnetic field should be determined.
        
        Returns
        -------
        magnetic field vector : arr3 [mT]
            Magnetic field at the argument position `pos` generated by the
            collection in units of [mT].
        """
        Btotal = sum([s.getB(pos) for s in self.sources])
        return Btotal


    def move(self,displacement):
        """
        This method moves each source in the collection by the argument vector `displacement`. 
        Vector input format can be either list, tuple or array of any data
        type (float, int).
        
        Parameters
        ----------
        displacement : vec3 - [mm]
            Displacement vector
            
        Returns
        -------
        None
            
        Example
        -------
        >>> from magpylib import source, Collection
        >>> pm1 = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1])
        >>> pm2 = source.magnet.Cylinder(mag=[0,0,1000],dim=[1,1])
        >>> print(pm1.position,pm2.position)
          [0. 0. 0.] [0. 0. 0.]
        >>> col = Collection(pm1,pm2)
        >>> col.move([1,1,1])
        >>> print(pm1.position,pm2.position)
          [1. 1. 1.] [1. 1. 1.]
        """
        for s in self.sources:
            s.move(displacement)
    
    
    def rotate(self,angle,axis,anchor='self.position'):
        """
        This method rotates each source in the collection about `axis` by `angle`. The axis passes
        through the center of rotation anchor. Scalar input is either integer or
        float. Vector input format can be either list, tuple or array of any
        data type (float, int).
        
        Parameters
        ----------
        angle  : scalar [deg]
            Angle of rotation in units of [deg]
        axis : vec3
            Axis of rotation
        anchor : vec3
            The Center of rotation which defines the position of the axis of rotation.
            If not specified all sources will rotate about their respective center.
            
        Returns
        -------
        None
            
        Example
        -------
        >>> from magpylib import source, Collection
        >>> pm1 = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1])
        >>> pm2 = source.magnet.Cylinder(mag=[0,0,1000],dim=[1,1])
        >>> print(pm1.position, pm1.angle, pm1.axis)
          [0. 0. 0.] 0.0 [0. 0. 1.]
        >>> print(pm2.position, pm2.angle, pm2.axis)
          [0. 0. 0.] 0.0 [0. 0. 1.]
        >>> col = Collection(pm1,pm2)
        >>> col.rotate(90, [0,1,0], anchor=[1,0,0])
        >>> print(pm1.position, pm1.angle, pm1.axis)
          [1. 0. 1.] 90.0 [0. 1. 0.]
        >>> print(pm2.position, pm2.angle, pm2.axis)
          [1. 0. 1.] 90.0 [0. 1. 0.]
        """
        for s in self.sources:
            s.rotate(angle,axis,anchor=anchor)
         
    
    def displaySystem(self,suppress=False,direc=False):
        """
        Runs plt.show() and Returns a matplotlib figure identifier and shows the collection display in an interactive plot.
        


        WARNING
        -------
        As a result of an inherent problem in matplotlib the 
        Poly3DCollections z-ordering fails when bounding boxes intersect.
        


        Parameters
        ----------
        suppress : bool
            If True, only return Figure information, do not show. Interactive mode must be off.
            Default: False.


        >>> ## Suppress matplotlib.pyplot.show() 
        >>> ## and returning figure from showing up
        >>> from matplotlib import pyplot 
        >>> pyplot.ioff()
        >>> figureData = Collection.displayFigure(suppress=True)

                
        Return    
        ------
        matplotlib Figure object
            graphics object is displayed through plt.show()
            
        Example
        -------
        >>> from magpylib import source, Collection
        >>> pm1 = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1],pos=[-1,-1,-1],angle=45,axis=[0,0,1])
        >>> pm2 = source.magnet.Cylinder(mag=[0,0,1000],dim=[2,2],pos=[0,-1,1],angle=45,axis=[1,0,0])
        >>> pm3 = source.magnet.Sphere(mag=[0,0,1000],dim=3,pos=[-2,1,2],angle=45,axis=[1,0,0])
        >>> C1 = source.current.Circular(curr=100,dim=6)
        >>> col = Collection(pm1,pm2,pm3,C1)
        >>> col.displaySystem()
        """ 
        fig = plt.figure(dpi=80,figsize=(8,8))
        ax = fig.gca(projection='3d')
        
        #count magnets
        Nm = 0
        for s in self.sources:
            if type(s) is Box or type(s) is Cylinder or type(s) is Sphere:
                Nm += 1
        cm = plt.cm.hsv # Linter complains about this but it is working pylint: disable=no-member
        #select colors
        colors = [cm(x) for x in linspace(0,1,Nm+1)]
        
        ii = -1
        SYSSIZE = finfo(float).eps ## Machine Epsilon for moment
        dipolesList=[]
        magnetsList=[]
        currentsList=[]
        for s in self.sources:
            if type(s) is Box:
                ii+=1 #increase color counter
                P = s.position
                D = s.dimension/2
                #create vertices in canonical basis
                v0 = array([D,D*array([1,1,-1]),D*array([1,-1,-1]),D*array([1,-1,1]),
                                   D*array([-1,1,1]),D*array([-1,1,-1]),-D,D*array([-1,-1,1])])
                #rotate vertices + displace
                v = array([ angleAxisRotation(s.angle,s.axis,d)+P for d in v0])
                #create faces
                faces = [[v[0],v[1],v[2],v[3]],
                         [v[0],v[1],v[5],v[4]],
                         [v[4],v[5],v[6],v[7]],
                         [v[2],v[3],v[7],v[6]],
                         [v[0],v[3],v[7],v[4]],
                         [v[1],v[2],v[6],v[5]]]
                # plot
                boxf = Poly3DCollection(faces, facecolors=colors[ii], linewidths=0.5, edgecolors='k', alpha=1)
                ax.add_collection3d(boxf)
                #check system size
                maxSize = amax(abs(v))
                if maxSize > SYSSIZE:
                    SYSSIZE = maxSize

                if direc is True:
                    s.color=colors[ii]
                    magnetsList.append(s)
            elif type(s) is Cylinder:
                ii+=1 #increase color counter
                P = s.position
                R,H = s.dimension/2
                
                resolution = 20
                
                #vertices
                phis = linspace(0,2*pi,resolution)
                vertB0 = array([[R*cos(p),R*sin(p),-H] for p in phis])
                vertT0 = array([[R*cos(p),R*sin(p),H] for p in phis])
                #rotate vertices+displacement
                vB = array([ angleAxisRotation(s.angle,s.axis,d)+P for d in vertB0])
                vT = array([ angleAxisRotation(s.angle,s.axis,d)+P for d in vertT0])
                #faces
                faces = [[vT[i],vB[i],vB[i+1],vT[i+1]] for i in range(resolution-1)]
                faces += [vT,vB]
                #plot
                coll = Poly3DCollection(faces, facecolors=colors[ii], linewidths=0.5, edgecolors='k', alpha=1)
                ax.add_collection3d(coll)
                #check system size
                maxSize = max([amax(abs(vB)),amax(abs(vT))])
                if maxSize > SYSSIZE:
                    SYSSIZE = maxSize
                
                if direc is True:
                    s.color=colors[ii]
                    magnetsList.append(s)
                
            elif type(s) is Sphere:
                ii+=1 #increase color counter
                P = s.position
                R = s.dimension/2
                
                resolution = 12
                
                #vertices
                phis = linspace(0,2*pi,resolution)
                thetas = linspace(0,pi,resolution)
                vs0 = [[[R*cos(phi)*sin(th),R*sin(phi)*sin(th),R*cos(th)] for phi in phis] for th in thetas]
                #rotate vertices + displacement
                vs = array([[ angleAxisRotation(s.angle,s.axis,v)+P for v in vss] for vss in vs0])
                #faces
                faces = []
                for j in range(resolution-1):
                    faces += [[vs[i,j],vs[i+1,j],vs[i+1,j+1],vs[i,j+1]] for i in range(resolution-1)]
                #plot
                boxf = Poly3DCollection(faces, facecolors=colors[ii], linewidths=0.5, edgecolors='k', alpha=1)
                ax.add_collection3d(boxf)
                #check system size
                maxSize = amax(abs(vs))
                if maxSize > SYSSIZE:
                    SYSSIZE = maxSize

                if direc is True:
                    s.color=colors[ii]
                    magnetsList.append(s)
                    
            elif type(s) is Line:
                P = s.position
                vs0 = s.vertices
                #rotate vertices + displacement
                vs = array([ angleAxisRotation(s.angle,s.axis,v)+P for v in vs0])
                #plot
                ax.plot(vs[:,0],vs[:,1],vs[:,2],lw=1,color='k')
                #check system size
                maxSize = amax(abs(vs))
                if maxSize > SYSSIZE:
                    SYSSIZE = maxSize

                if direc is True:
                    currentsList.append(s)

            elif type(s) is Circular:
                P = s.position
                R = s.dimension/2
                
                resolution = 20
                
                #vertices
                phis = linspace(0,2*pi,resolution)
                vs0 = array([[R*cos(p),R*sin(p),0] for p in phis])
                #rotate vertices + displacement
                vs = array([ angleAxisRotation(s.angle,s.axis,v)+P for v in vs0])
                #plot
                ax.plot(vs[:,0],vs[:,1],vs[:,2],lw=1,color='k')
                #check system size
                maxSize = amax(abs(vs))
                if maxSize > SYSSIZE:
                    SYSSIZE = maxSize
                    
                if direc is True:
                    currentsList.append(s)

        
            elif type(s) is Dipole:
                P = rotatePosition(s.position,s.angle,s.axis)
                maxSize = amax(abs(P))
                if maxSize > SYSSIZE:
                    SYSSIZE = maxSize

                dipolesList.append(s)


        m = self.markers
        if all(val==0 for val in m):
            ax.scatter(m[0],m[1],m[2],s=20,marker='x')
        else:
            ax.scatter(m[0],m[1],m[2],s=20,marker='x')
            ax.scatter(-m[0],-m[1],-m[2],s=20,marker='x')
            maxSize = amax(abs(max(m)))
            if maxSize > SYSSIZE:
                SYSSIZE = maxSize
        

        for d in dipolesList:
            drawDipole( d.position,d.moment,
                        d.angle,d.axis,
                        SYSSIZE,plt)

        if direc is True: ### Draw the Magnetization axes and current directions
            drawCurrentArrows(currentsList,SYSSIZE,plt)
            drawMagAxis(magnetsList,SYSSIZE,plt)

        for tick in ax.xaxis.get_ticklabels()+ax.yaxis.get_ticklabels()+ax.zaxis.get_ticklabels():
            tick.set_fontsize(12)
        ax.set_xlabel('x[mm]', fontsize=12)
        ax.set_ylabel('y[mm]', fontsize=12)
        ax.set_zlabel('z[mm]', fontsize=12)
        ax.set(
            xlim=(-SYSSIZE,SYSSIZE),
            ylim=(-SYSSIZE,SYSSIZE),
            zlim=(-SYSSIZE,SYSSIZE),
            aspect=1
            )
        plt.tight_layout()

        if suppress is False:
            plt.show()

        return plt.gcf()
        