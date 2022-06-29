#!/usr/bin/env python3.7

"""
The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""

# builtins
import random
from typing import Tuple, Union, List
from copy import deepcopy

# packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.affinity import scale, rotate
from shapely.ops import nearest_points
import pyclipper
import pymunk

# ascent
from src.utils import Exceptionable, DownSampleMode, Config, WriteMode, SetupMode


class Trace(Exceptionable):
    """
    Core object for manipulating points/traces of nerve sections. Trace is the fundamental building block for nerve
    geometries (fascicles, nerve, endoneurium, perineurium)
    """

    def __init__(self, points, exception_config):
        """
        :param points: nx3 list expected to be a loop. If a non-loop is given, then the functionality is not defined.
        :param exception_config: data passed from a higher object for exceptions.json, hence why it inherits exceptionable.
        """

        # These are private instance variables that are returned by getter
        self.__contour = None
        self.__polygon = None
        self.__centroid = None
        self.__int_points = None
        self.__min_circle = None

        # set up superclass
        Exceptionable.__init__(self, SetupMode.OLD, exception_config)

        # add 0 as z value if only x and y given
        if np.shape(points)[1] == 2:
            points = np.append(points, np.zeros([len(points), 1]), 1)

        self.points = None  # must declare instance variable in __init__ at some point!
        self.append(points)

    # %% public, MUTATING methods
    def append(self, points):
        """
        :param points: nx3 ndarray, where each row is a point [x, y, z]
        """
        # make sure points is multidimensional ndarray (only applicable if a single point is passed in)
        points = np.atleast_2d(points).astype(float)

        # ensure 3 columns
        if np.shape(points)[1] != 3:
            self.throw(5)

        # if points has not been initialized (case when called by __init__)
        if self.points is None:
            self.points = points
        else:
            self.points = np.append(self.points, points, axis=0)  # axis = 0 to append rows

        # required for mutating method
        self.__update()

    def offset(self, fit: dict = None, distance: float = None):
        """
        NOT AN AFFINE TRANSFORMATION
        :param fit: used to scale up by a factor if you are offsetting the boundary by a linear fit
        :param distance: used to scale by a discrete distance
        """

        # create clipper offset object
        pco = pyclipper.PyclipperOffset()

        # put self.points into a 2-D tuple
        tuple_points = tuple([tuple(point[:2]) for point in self.points])

        # add points to clipper
        pco.AddPath(tuple_points, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)

        if fit is not None:
            # find offset distance from factor and mean radius
            distance: float = fit.get("a") * 2 * np.sqrt(self.area() / np.pi) + fit.get("b")
        elif distance is None:
            self.throw(29)

        # set new points of offset
        self.points = None
        newpoints = pco.Execute(distance)
        self.append([point + [0] for point in newpoints[np.argmax([len(l) for l in newpoints])]])

        # cleanup
        self.__update()
        pco.Clear()

    def smooth(self, distance, area_compensation=True):
        """
        Smooths a contour using a dilation followed by erosion
        :param distance: amount to use for dilation and erosion, in whatever units the trace is using
        """
        if distance < 0: self.throw(111)
        if distance == 0: return
        pre_area = self.area()
        self.offset(fit=None, distance=distance)
        self.offset(fit=None, distance=-distance)
        if area_compensation == True:
            # scale back to area of original trace
            self.scale((pre_area / self.area())**.5)
            if abs(pre_area - self.area()) > 1:
                self.throw(128)
        else:
            self.scale(1)
        self.points = np.flip(self.points, axis=0)  # set points to opencv orientation

    def scale(self, factor: float = 1, center: Union[List[float], str] = 'centroid'):
        """
        :param factor: scaling factor to scale up by - multiple all points by a factor.
        [X 0 0; 0 Y 0; 0 0 Z]
        :param center: string "centroid", string "center" or a point [x,y]
        """

        if isinstance(center, list):
            center = tuple(center)
        else:
            if center not in ['centroid', 'center']:
                self.throw(17)

        scaled_polygon: Polygon = scale(self.polygon(), *([factor] * 3), origin=center)

        self.points = None
        self.append([list(coord[:2]) + [0] for coord in scaled_polygon.boundary.coords])
        self.__update()

    def rotate(self, angle: float, center: Union[List[float], str] = 'centroid'):
        """
        :param angle: rotates trace by radians CCW
        :param center: string "centroid", string "center" or a point [x,y]
        """

        if isinstance(center, list):
            center = tuple(center)
        else:
            if center not in ['centroid', 'center']:
                self.throw(17)

        rotated_polygon: Polygon = rotate(self.polygon(), angle, origin=center, use_radians=True)

        self.points = None
        self.append([list(coord[:2]) + [0] for coord in rotated_polygon.boundary.coords])
        self.__update()

    def shift(self, vector):
        """
        :param vector: 1-dim vector with 3 elements... shape is (3,)
        Moves trace [x,y] in any direction (x,y,z) could be np.ndarray/list/tuple
        """

        # must be 3 item vector
        if np.shape(vector) != (3,):
            self.throw(3)

        # apply shift to each point
        vector = [float(item) for item in vector]
        self.points += vector

        # required for mutating method
        self.__update()

    def down_sample(self, mode: DownSampleMode, step: int):
        """
        Simple down sample method to remove points at even intervals.
        Will start indices on "step-th" element (i.e. if step is 4, first selected element at index 3)
        :param mode: decide whether to KEEP only the points on steps, or REMOVE only those points
        :param step: spacing between each selected point (both keep and remove)
        """

        ii = list(range(step - 1, self.count(), step))

        if mode == DownSampleMode.KEEP:
            pass  # nothing here; just showing the case for readability
        else:  # mode == DownSampleMode.REMOVE:
            ii = [i for i in list(range(0, self.count())) if i not in ii]

        self.points = self.points[ii, :]

        # required for mutating method
        self.__update()

    # %% public, NON-MUTATING methods
    def count(self) -> int:
        """
        :return: number of rows in self.points (i.e. number of points)
        """
        return np.shape(self.points)[0]

    # %% dependent on shapely.geometry.Polygon (ALL 2D GEOMETRY)
    def polygon(self) -> Polygon:
        """
        :return: shape of polygon as a shapely.geometry.Polygon (ALL 2D geometry)
        """
        if self.__polygon is None:
            if len(set(self.points[:, 2])) != 1:
                self.throw(6)

            self.__polygon = Polygon([tuple(point) for point in self.points[:, :2]])

        return self.__polygon

    def bounds(self):
        """
        :return: bounds of the trace object
        """
        return self.polygon().bounds

    def random_points(self, count: int, buffer: float = 0, my_xy_seed: int = 123) -> List[Tuple[float]]:
        """
        :param my_xy_seed:
        :param buffer:
        :param count: number of points to find
        :return: list of tuples (x,y) that are within the trace (polygon)
        """
        trace_to_compare = self.deepcopy()
        trace_to_compare.offset(None, 1)  # seems to improve reliability to expand 1 um, and then shrink back first
        trace_to_compare.offset(None, -1)
        trace_to_compare.offset(None, -buffer)

        min_x, min_y, max_x, max_y = trace_to_compare.polygon().bounds

        points: List[Tuple[float]] = []
        random.seed(my_xy_seed)

        while len(points) < count:

            coordinate = tuple((random.random() * (ceiling - floor)) + floor
                               for floor, ceiling in ((min_x, max_x), (min_y, max_y)))

            if Point(coordinate).within(trace_to_compare.polygon()):
                # print('coord:{}, {}'.format(coordinate[0], coordinate[1]))
                points.append(coordinate)

        return points

    def within(self, outer: 'Trace') -> bool:
        """
        :param outer: other Trace to check
        :return: True if within other Trace, else False
        """
        return self.polygon().within(outer.polygon())

    def intersects(self, other: 'Trace') -> bool:
        """
        :param other: other Trace to check
        :return: True if intersecting, else False
        """
        return self.polygon().boundary.intersects(other.polygon().boundary)

    def centroid(self) -> Tuple[float, float]:
        """
        :return: ellipse centroid as tuple: center --> (x, y)
        """
        if self.__centroid is None:
            self.__centroid = list(self.polygon().centroid.coords)[0]

        return self.__centroid

    def angle_to(self, other: 'Trace'):
        """
        :param other: type Trace
        :return: returns the CCW angle to the other trace based on self and other's centroids
        """
        return Trace.angle(self.centroid(), other.centroid())

    @staticmethod
    def angle(first, second):
        return np.arctan2(second[1] - first[1], second[0] - first[0])

    def area(self) -> float:
        """
        :return: area of Trace
        """
        return self.polygon().area

    def min_distance(self, other: 'Trace', return_points: bool = False) -> Union[float, tuple]:
        """
        :param return_points: boolean for whether the closest points will be returned
        :param other: Trace to find distance to
        :return: float minimum distance and the points if indicated
        """

        distance = self.polygon().boundary.distance(other.polygon().boundary)

        if not return_points:
            return distance
        else:
            return distance, nearest_points(self.polygon(), other.polygon())

    def max_distance(self, other: 'Trace') -> float:
        """
        :param other: Trace to find distance to
        :return: float maximum distance
        """
        return self.polygon().boundary.hausdorff_distance(other.polygon().boundary)

    def centroid_distance(self, other: 'Trace', return_points: bool = False) -> Union[float, tuple]:
        """
        :param return_points: boolean for whether the closest points will be returned
        :param other: Trace to find distance to
        :return: float maximum distance and the points if indicated
        """
        self_c = Point([self.centroid()])

        distance = self_c.distance(other.polygon().boundary)

        if not return_points:
            return distance
        else:
            return distance, nearest_points(self_c, other.polygon().boundary)

    # %% contour-dependent (cv2)
    def contour(self) -> np.ndarray:
        """
        Builds a "fake" contour so that cv2 can analyze it (independent of the image)
        Use for to_circle and to_ellipse
        """
        if self.__contour is None:
            # check points all have same z-value (MAY BE CHANGED?)
            if len(set(self.__int_points[:, 2])) != 1:
                self.throw(6)

            self.__contour = np.zeros([self.count(), 1, 2])
            # check that all points in same plane
            for i, point in enumerate(self.__int_points):
                self.__contour[i, 0] = point[:-1]  # do not include z

            self.__contour = self.__int32(self.__contour)

        return self.__contour

    def ellipse(self):
        """
        NOTE: this uses 2-D contour (ignores z-coordinate)
        :return: ellipse specs as 2-D tuple: (center, axes) --> ((x, y), (a, b), angle)
        """
        return cv2.fitEllipse(self.contour())

    def mean_radius(self) -> float:
        """
        :return: the mean radius of the best-fit ellipse
        """
        ((_, _), (a, b), _) = cv2.fitEllipse(self.contour())
        return float(np.mean([item / 2 for item in (a, b)], axis=0))

    def to_ellipse(self):
        """
        :return: returns ellipse object methods for best-fit ellipse
        """
        # ((centroid), (axes), angle) ... note angle is in degrees
        ((u, v), (a, b), angle) = self.ellipse()

        # return the associated ellipse object, after converting angle to degrees
        return self.__ellipse_object(u, v, a, b, angle * 2 * np.pi / 360)

    def to_circle(self, buffer: float = 0.0):
        """
        :return: returns ellipse object methods for best-fit circle (averages axes of best fit ellipse and
        sets as circle radius)
        """
        # ((centroid), (axes), angle) ... note angle is in degrees
        ((u, v), (a, b), angle) = self.ellipse()

        # find average radius of circle
        # casting to float is just so PyCharm stops yelling at me (I think it should already be a float64?)
        r = float(np.sqrt(self.area() / np.pi)) - buffer

        # return the associated ellipse object, after converting angle to degrees
        # also, PyCharm thinks that np.mean returns a ndarray, but it definitely isn't in this case
        return self.__ellipse_object(u, v, 2 * r, 2 * r, angle * 2 * np.pi / 360)

    def __ellipse_object(self, u: float, v: float, a: float, b: float, angle: float) -> 'Trace':
        """
        :param u: x value of center
        :param v: y value of center
        :param a: first (minor) axis, twice the "radius"
        :param b: second (major) axis, twice the "radius"
        :param angle: clockwise angle to rotate in radians
        :return: a new Trace object with the points of the best-fit ellipse (point count is preserved)

        This one is for all you Matlab fanpeople
        """
        # get t values for parameterized ellipse and preserve number of points
        t = np.linspace(0, 2 * np.pi, self.count())

        # divide a and b by 2 because they are AXIS lengths
        (a, b) = (item / 2 for item in (a, b))

        # find x and y values along ellipse (not shifted to centroid yet)
        (x, y) = (a * np.cos(t), b * np.sin(t))

        # create rotation matrix
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])

        # apply rotation and shift to centroid
        points = (rot_mat @ np.array([x, y])).T + [u, v]

        # add column of z-values (should all be the same)
        points = np.append(points, np.c_[self.points[:, 2]], axis=1)

        return Trace(points, self.configs[Config.EXCEPTIONS.value])

    # %% output
    def plot(self, plot_format: str = 'k-', color: Tuple[float, float, float, float] = None, ax: plt.Axes = None, linewidth=1):
        """
        :param ax:
        :param color:
        :param plot_format: the plt.plot format spec (see matplotlib docs)
        """

        if ax is None:
            ax = plt.gca()

        # append first point to plot as loop
        points = np.vstack([self.points, self.points[0]])

        if color is not None:
            ax.fill(points[:, 0], points[:, 1], color=color)

        ax.plot(points[:, 0], points[:, 1], plot_format, linewidth=linewidth)

    def plot_centroid(self, plot_format: str = 'k*'):
        """
        :param plot_format: the plt.plot format spec (see matplotlib docs)
        """
        plt.plot(*self.centroid(), plot_format)

    def write(self, mode: WriteMode, path: str):
        """
        Write Trace data (points, elements, etc.) to file
        :param mode: choice of write implementations - sectionwise file for COMSOL
        :param path: string path with file name of file WITHOUT extension (it is derived from mode)
        :return: string full path including extension that was written to
        """
        # add extension
        # path += mode.value
        path += WriteMode.file_endings.value[mode.value]

        try:
            # open in write mode; "+" indicates to create file if not found
            # print('writing to: {}'.format(path))
            with open(path, 'w+') as f:
                count = self.count()

                # choose implementation from mode
                if mode == WriteMode.SECTIONWISE:
                    # write coordinates
                    f.write('%% Coordinates\n')
                    for i in range(count):
                        f.write('{}\t{}\t{}\n'.format(self.points[i, 0],
                                                      self.points[i, 1],
                                                      self.points[i, 2]))

                    # write elements (corresponding to their coordinates)
                    f.write('%% Elements\n')
                    for i in range(count):
                        # if not last point, attach to next point
                        if i < count - 1:
                            f.write('{}\t{}\n'.format(i + 1, i + 2))
                        else:  # attach to first point (closed loop)
                            f.write('{}\t{}\n'.format(i + 1, 1))

                elif mode == WriteMode.SECTIONWISE2D:
                    # write coordinates
                    f.write('%% Coordinates\n')
                    for i in range(count):
                        f.write('{}\t{}\n'.format(self.points[i, 0],
                                                  self.points[i, 1]))

                else:
                    self.throw(4)

        except EnvironmentError:
            # only one of these can run, so comment at will
            self.throw(7)
            # raise

        return path

    def deepcopy(self) -> 'Trace':
        """
        :return: creates a new place in memory for the Trace. See: https://docs.python.org/2/library/copy.html
        """
        return deepcopy(self)

    def pymunk_poly(self) -> Tuple[pymunk.Body, pymunk.Poly]:
        """
        :return: a body and polygon shape, rigid polygon
        """
        copy = self.deepcopy()
        copy.shift(-np.array(list(copy.centroid()) + [0]))
        # copy.down_sample(DownSampleMode.KEEP, 1)

        mass = 1
        radius = 1
        vertices = [tuple(point[:2]) for point in copy.points]
        inertia = pymunk.moment_for_poly(mass, vertices)
        body = pymunk.Body(mass, inertia)
        body.position = self.centroid()  # position is tracked from trace centroid
        shape = pymunk.Poly(body, vertices, radius=radius)
        shape.density = 0.01  # all fascicles have same density so this value does not matter
        shape.friction = 0.5
        shape.elasticity = 0.0  # they absorb all energy, i.e. they do not bounce
        return body, shape

    def pymunk_segments(self, space: pymunk.Space) -> List[pymunk.Segment]:
        """
        :param space:
        :return: returns a list of static line segments that cannot be moved
        """
        copy = self.deepcopy()
        # copy.down_sample(DownSampleMode.KEEP, 1)

        points = np.vstack((copy.points, copy.points[0]))

        segments: List[pymunk.Segment] = []

        for first, second in zip(points[:-1], points[1:]):
            if np.array_equiv(first[:2], second[:2]):
                pass
            segments.append(pymunk.Segment(space.static_body,
                                           first[:2].tolist(),
                                           second[:2].tolist(),
                                           radius=1.0))
        return segments

        # %% METHODS ADAPTED FROM: https://www.nayuki.io/res/smallest-enclosing-circle/smallestenclosingcircle-test.py

        # Data conventions: A point is a pair of floats (x, y).
        # A circle is a triple of floats (center x, center y, radius).

        # Returns the smallest circle that encloses all the given points. Runs in expected O(n) time, randomized.
        # Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
        # Output: A triple of floats representing a circle.
        # Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.
        #
        # Initially: No boundary points known

    def make_circle(self):
        # Convert to float and randomize order
        shuffled = [(float(x), float(y)) for (x, y) in self.points[:, 0:2]]
        random.shuffle(shuffled)

        # Progressively add points to circle or recompute circle
        c = None
        for (i, p) in enumerate(shuffled):
            if c is None or not self.is_in_circle(c, p):
                c = self._make_circle_one_point(shuffled[: i + 1], p)
        return c

    # One boundary point known
    def _make_circle_one_point(self, points, p):
        c = (p[0], p[1], 0.0)
        for (i, q) in enumerate(points):
            if not self.is_in_circle(c, q):
                if c[2] == 0.0:
                    c = self._make_diameter(p, q)
                else:
                    c = self._make_circle_two_points(points[: i + 1], p, q)
        return c

    # Two boundary points known
    def _make_circle_two_points(self, points, p, q):
        circ = self._make_diameter(p, q)
        left = None
        right = None
        px, py = p
        qx, qy = q

        # For each point not in the two-point circle
        for r in points:
            if self.is_in_circle(circ, r):
                continue

            # Form a circumcircle and classify it on left or right side
            cross = self._cross_product(px, py, qx, qy, r[0], r[1])
            c = self._make_circumcircle(p, q, r)
            if c is None:
                continue
            elif cross > 0.0 and (
                    left is None or self._cross_product(px, py, qx, qy, c[0], c[1]) >
                    self._cross_product(px, py, qx, qy, left[0], left[1])):
                left = c
            elif cross < 0.0 and (
                    right is None or self._cross_product(px, py, qx, qy, c[0], c[1]) <
                    self._cross_product(px, py, qx, qy, right[0], right[1])):
                right = c

        # Select which circle to return
        if left is None and right is None:
            return circ
        elif left is None:
            return right
        elif right is None:
            return left
        else:
            return left if (left[2] <= right[2]) else right

    def _make_diameter(self, a, b):
        cx = (a[0] + b[0]) / 2.0
        cy = (a[1] + b[1]) / 2.0
        r0 = np.math.hypot(cx - a[0], cy - a[1])
        r1 = np.math.hypot(cx - b[0], cy - b[1])
        return cx, cy, max(r0, r1)

    def _make_circumcircle(self, a, b, c):
        # Mathematical algorithm from Wikipedia: Circumscribed circle
        ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2.0
        oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2.0
        ax = a[0] - ox
        ay = a[1] - oy
        bx = b[0] - ox
        by = b[1] - oy
        cx = c[0] - ox
        cy = c[1] - oy
        d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
        if d == 0.0:
            return None
        x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
        y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
        ra = np.math.hypot(x - a[0], y - a[1])
        rb = np.math.hypot(x - b[0], y - b[1])
        rc = np.math.hypot(x - c[0], y - c[1])
        return x, y, max(ra, rb, rc)

    def is_in_circle(self, c, p):
        _MULTIPLICATIVE_EPSILON = 1 + 1e-14
        return c is not None and np.math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON

    # Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
    def _cross_product(self, x0, y0, x1, y1, x2, y2):
        return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

    def smallest_enclosing_circle_naive(self) -> Tuple[float, float, float]:
        # Returns the smallest enclosing circle in O(n^4) time using the naive algorithm.

        # Degenerate cases
        if len(self.points) == 0:
            return None
        elif len(self.points) == 1:
            return self.points[0][0], self.points[0][1], 0

        # Try all unique pairs
        result = None
        for i in range(len(self.points)):
            p = self.points[i]
            for j in range(i + 1, len(self.points)):
                q = self.points[j]
                c = self._make_diameter(p, q)
                if (result is None or c[2] < result[2]) and \
                        all(self.is_in_circle(c, r) for r in self.points):
                    result = c
        if result is not None:
            return result  # This optimization is not mathematically proven

        # Try all unique triples
        for i in range(len(self.points)):
            p = self.points[i]
            for j in range(i + 1, len(self.points)):
                q = self.points[j]
                for k in range(j + 1, len(self.points)):
                    r = self.points[k]
                    c = self._make_circumcircle(p, q, r)
                    if c is not None and (result is None or c[2] < result[2]) and \
                            all(self.is_in_circle(c, s) for s in self.points):
                        result = c

        if result is None:
            raise AssertionError()

        return result

    # %% private utility methods
    def __update(self):
        self.__int_points = self.__int32(self.points)
        self.__contour = None
        self.__polygon = None
        self.__centroid = None

    @staticmethod
    def __int32(points: np.ndarray):
        return np.array(np.round(points), dtype=np.int32)
