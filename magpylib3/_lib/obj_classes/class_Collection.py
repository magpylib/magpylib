"""Collection class code"""

from magpylib3._lib.math.utility import format_src_input, check_duplicates
from magpylib3._lib.fields.field_BHwrapper import getB, getH
from magpylib3._lib.graphics import display

class Collection:
    """ Group multiple sources in one Collection.
    
    ### Properties
    - sources (list): Ordered list of sources in the Collection

    ### Methods
    - move: move all sources simultaneously
    - rotate: rotate all sources (about the same anchor)
    - display: display the Collection graphically
    - getB: compute B-field of Collection
    - getH: compute H-field of Collection
    - add_sources: add sources to the Collection (no copy)
    - remove_source: remove a source from the Collection (no copy)

    ### Returns:
    - (Collection object)

    ### Info
    Collections have +/- operations defined to add and remove sources.
        (no copy is made)

    Collections are iterable, iterating through self.sources

    Collections have getitem defined, col[i] returns self.sources[i]
    """

    def __init__(self, *sources):

        self.sources = sources


    # sources properties --------------------------------------------
    @property
    def sources(self):
        """ list of sources in Collection
        """
        return self._sources


    @sources.setter
    def sources(self, sources):
        """ set sources in the Collection, arbitrary input
        """
        # format input
        src_list = format_src_input(sources)
        # check and eliminate duplicates
        src_list = check_duplicates(src_list)
        # set attributes
        self._sources = src_list


    # dunders -------------------------------------------------------
    def __add__(self, source):
        """ add sources or collections
        """
        self.add(source)
        return self
    

    def __sub__(self, source):
        """ remove a source from the Collection
        """
        self.remove(source)
        return self
    

    def __iter__(self):
        """ make Collection iterable
        """
        yield from self._sources
    

    def __getitem__(self,i):
        """ provide getitem property
        """
        return self._sources[i]


    # methods -------------------------------------------------------
    def add(self,*sources):
        """ Add arbitrary sources to Collection.

        ### Args:
        - sources (sequence of sources and collections): add arbitrary 
            sequences of sources and collections to the Collection.
            The new sources will be added at the end of self.sources.
            Duplicates will be eliminated.

        ### Returns:
        - self
        """

        # format input
        src_list = format_src_input(sources)
        # combine with original src_list
        src_list = self._sources + src_list
        # check and eliminate duplicates
        src_list = check_duplicates(src_list)
        # set attributes
        self._sources = src_list
        return self
    

    def remove(self,source):
        """ Remove source from collection

        ### Args:
        - source (source): remove source from Collection

        ### Returns:
        - self
        """
        self._sources.remove(source)
        return self
    

    def getB(self, pos_obs, **kwargs):
        """ Compute B-field of Collection at observer positions.

        ### Args:
        - pos_obs (N1 x N2 x ... x 3 vec): single position or set of 
            observer positions in units of mm.

        ### Returns:
        - (N1 x N2 x ... x 3 ndarray): B-field at observer positions
            in units of mT.
        """
        return getB(self._sources, pos_obs=pos_obs, sumup=True, **kwargs)
    

    def getH(self, pos, **kwargs):
        """ Compute H-field of Collection at observer positions.

        ### Args:
        - pos_obs (N1 x N2 x ... x 3 vec): single position or set of 
            observer positions in units of mm.

        ### Returns:
        - (N1 x N2 x ... x 3 ndarray): H-field at observer positions
            in units of kA/m.
        """
        return getH(self._sources, pos, sumup=True, **kwargs)


    def display(self, **kwargs):
        """ 
        Display Collection graphically. kwargs of top level display()
            function.
        """
        display(self, **kwargs)


    def move(self, displacement):
        """
        Translate all sources in Collection by the argument vector.

        ### Args:
        - displacement (vec3): displacement vector in units of mm.

        ### Returns:
        - self
        """
        for s in self:
            s.move(displacement)
        return self


    def rotate(self, rot, anchor=None):
        """ 
        Rotate all sources in Collection.

        ### Args:
        - rot (rotation input): Can either be a pair (angle, axis) with 
            angle a scalar given in [deg] and axis an arbitrary 3-vector, 
            or a scipy..Rotation object.
        - anchor (vec3): The axis of rotation passes through the anchor point. 
            By default (anchor=None) the objects will rotate about their own 
            center.

        ### Returns:
        - self
        """
        for s in self:
            s.rotate(rot, anchor)
        return self
