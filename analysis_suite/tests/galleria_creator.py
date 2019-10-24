"""
Functions that generate fake galleria inside wells

Contours created using the code by TheImportanceOfBeingErnest on stack over flow:
https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib

I use this contour to create a mask which I can then generate random pixel intensities
and add to fake plate wells
"""

from skimage import feature
import scipy.ndimage.morphology as morph
from scipy.special import binom
import matplotlib.pyplot as plt
import numpy as np

def well_with_galleria(well, galleria_pixel = 5000):
    """
    Generates fake galleria inside a well

    Parameters
    ------
    well : ndarray
        An array which represents the well the galleria needs to be added to
    galleria_pixel : float
        The pixel value to generate random values around to represent the galleria
    """
    # First get the shape of the well
    well_height = well.shape[0]
    well_width = well.shape[1]

    # Create a well mask and generate contours for the galleria outline
    well_mask = np.zeros((well_height, well_width))
    x, y = create_galleria(rad = 0.95, edgy = 0.5, num_points = 2)

    # Need to convert the coordinates to the correct range as currently between 0 and 1
    for x_coord, y_coord in zip(x, y):
        # First remove any negative pixels
        if x_coord < 0:
            x_coord = 0
        if y_coord < 0:
            y_coord = 0

        # Now scale the coordinates to the size of the array representing the well
        new_x = int((x_coord * well_width))
        new_y = int((y_coord * well_height))
        if new_x >= well_width:
            new_x = well_width - 1
        if new_y >= well_height:
            new_y = well_height - 1
        try:
            well_mask[new_y, new_x] = 1
        except:
            pass

    # Use canny edge filter to create a boolean mask, dilate this to create one whole area
    # then fill this area to create the mask
    edges = feature.canny(well_mask)
    edges2 = morph.binary_dilation(edges, iterations=3)
    well_mask_filled = morph.binary_fill_holes(edges2)

    # create an array of randomly generated pixel values the same size as the well
    whole_well_bright = np.random.normal(loc=1, scale=0.05, size=(well_height, well_width)) * galleria_pixel

    # Use the mask to set the equivalent area in the well to the pixel values from the bright well
    well[well_mask_filled == 1] = whole_well_bright[well_mask_filled == 1]

    return well

def create_galleria(rad = 0.2, edgy = 0.05, num_points = 3):
    """
    Generates contours that represent the galleria using a stackoverflow method as mentioned
    in page docstring

    Parameters
    ------
    rad : float
        is a number between 0 and 1 to steer the distance of
              control points.
    edgy : float
        is a parameter which controls how "edgy" the curve is,
               edgy=0 is smoothest
    num_points : int
        The number of points to generate curve lines between

    Returns
    ------
    x : list
        list of x coordinates for the contour
    y : list
        corresponding list of y coordinates for the contour
    """

    # Get the points
    a = get_random_points(n=num_points, scale=1)

    # Generate the curve lines between them
    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)

    return x, y

### Below functions taken from stackoverflow answer as explained in file docstring

def bezier(points, num=200):
    bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)

def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)
