#!/usr/bin/env python3.7

"""
The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""

# builtins
import itertools
from typing import List, Tuple, Union
from copy import deepcopy
import os

# packages
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ascent
from .trace import Trace
from .nerve import Nerve
from src.utils import Exceptionable, SetupMode, WriteMode


class Fascicle(Exceptionable):

    def __init__(self, exception_config, outer: Trace, inners: List[Trace] = [], outer_scale: dict = None):
        """
        Fascicle can be created with either:
         option 1: an outer and any number of inners
         option 2: an inner, which is passed in as an outer argument, and scaled out to make a virtual outer
         option 3: ... tbd

        :param outer_scale: how much the inner will be scaled to make a virtual outer
        :param exception_config: existing data already loaded form JSON (hence SetupMode.OLD)
        :param inners: list of inner Traces (i.e. endoneuriums)
        :param outer: single outer Trace (i.e. perineurium)
        """

        # set up superclass
        Exceptionable.__init__(self, SetupMode.OLD, exception_config)

        # intialize scale attribute
        self.outer_scale = None

        # initialize constituent traces
        self.inners: List[Trace] = inners
        self.outer: Trace = outer

        if len(inners) > 0: self.validate()

    def validate(self):
        """
        Performs checks on traces
        """
        # ensure all inner Traces are actually inside outer Trace
        if any([not inner.within(self.outer) for inner in self.inners]):
            self.throw(8)
        # ensure no Traces intersect (and only check each pair of Traces once)
        pairs: List[Tuple[Trace]] = list(itertools.combinations(self.all_traces(), 2))
        if any([pair[0].intersects(pair[1]) for pair in pairs]):
            self.plot()
            plt.axes().set_aspect('equal')
            plt.show()
            self.throw(9)

    def intersects(self, other: Union['Fascicle', Nerve]):
        """
        :param other: the other Fascicle or Nerve to check
        :return: True if a Trace intersection is found
        """
        if isinstance(other, Fascicle):
            return self.outer.intersects(other.outer)
        else:  # other must be a Nerve
            return self.outer.intersects(other)

    def min_distance(self, other: Union['Fascicle', Nerve], return_points: bool = False) -> Union[float, tuple]:
        """
        :param return_points:
        :param other: other Fascicle or Nerve to check
        :return: minimum straight-line distance between self and other
        """
        if isinstance(other, Fascicle):
            return self.outer.min_distance(other.outer, return_points=return_points)
        else:  # other must be a Nerve
            return self.outer.min_distance(other, return_points=return_points)

    def centroid_distance(self, other: Union['Fascicle', Nerve], return_points: bool = False):
        """
        :param return_points:
        :param other: other Fascicle or Nerve to check
        :return: minimum straight-line distance between self and other
        """
        if isinstance(other, Fascicle):
            return self.outer.centroid_distance(other.outer, return_points=return_points)
        else:  # other must be a Nerve
            return self.outer.centroid_distance(other, return_points=return_points)

    def within_nerve(self, nerve: Nerve) -> bool:
        """
        :param nerve:
        :return: returns boolean, true if fascicle is within the nerve, false if fascicle is not in the nerve
        """
        return self.outer.within(nerve)

    def shift(self, vector):
        """
        :param vector: shift to apply to all Traces in the fascicle
        :return:
        """
        for trace in self.all_traces():
            trace.shift(vector)

    def all_traces(self) -> List[Trace]:
        """
        :return: list of all traces
        """
        return list(self.inners) + [self.outer]

    def centroid(self) -> Tuple[float, float]:
        """
        :return: centroid of outer trace (ellipse method)
        """
        return self.outer.centroid()

    def area(self):
        """
        :return: area of outer trace
        """
        return self.outer.area()

    def ellipse(self):
        """
        :return: ellipse fit of outer trace (see Trace.ellipse for return format/type)
        """
        return self.outer.ellipse()

    def mean_radius(self):
        """
        :return:
        """
        return self.outer.mean_radius()

    def angle_to(self, other: Union['Fascicle', Nerve]):
        """
        :param other: type Trace
        :return: returns the CCW angle to the other trace based on self and other's centroids
        """
        if isinstance(other, Fascicle):
            return self.outer.angle_to(other.outer)
        else:  # must be Nerve
            return self.outer.angle_to(other)

    def plot(self, plot_format: str = 'b-', color: Union[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]] = None,
             ax: plt.Axes = None, outer_flag=True, inner_index_start: int = None):
        """
        :param outer_flag:
        :param inner_index_start:
        :param ax:
        :param color:
        :param plot_format: outers automatically black, plot_format only affects inners
        """
        if ax is None:
            ax = plt.gca()

        if outer_flag:
            self.outer.plot(ax=ax)
            
        if color is not None:
            if not len(self.inners) == len(color):
                self.throw(145)
        else:
            color = [None] *  len(self.inners)
            
        for i, (inner, c) in enumerate(zip(self.inners, color)):
            inner.plot(plot_format, color=c, ax=ax)
            if inner_index_start is not None:
                ax.text(*inner.centroid(), s=str(i + inner_index_start), ha='center', va='center')

    def deepcopy(self):
        """
        :return: creates a new place in memory for the Trace. See: https://docs.python.org/2/library/copy.html
        """
        return deepcopy(self)

    def scale(self, factor: float, center: List[float] = None):
        """
        :param factor: scale factor
        :param center: [x,y]
        """
        if center is None:
            center = list(self.centroid())

        for trace in self.all_traces():
            trace.scale(factor, center)

    def rotate(self, angle: float, center: List[float] = None):
        """
        :param angle: angle in radians
        :param center: [x,y]
        """
        if center is None:
            center = list(self.centroid())

        for trace in self.all_traces():
            trace.rotate(angle, center)

    def smallest_trace(self) -> Trace:
        """
        :return: the trace with the smallest area
        """
        areas = np.array([trace.area() for trace in self.all_traces()])
        return np.array(self.all_traces())[np.where(areas == np.min(areas))][0]

    def perineurium_setup(self, fit: dict):
        """
        Takes inners which were passed in as outers, and generates the perineurium
        :param factor: a dictionary with the values describing a linear relationship 
        between fascicle size and perineurium thickness
        """
        self.outer_scale = fit
        # check that outer scale is provided
        if self.outer_scale is None:
            self.throw(14)

        # set single inner trace
        self.inners = [self.outer.deepcopy()]

        # scale up outer trace
        self.outer.offset(fit=fit)

        # check for any bad traces
        self.validate()

    @staticmethod
    def to_list(inner_img_path: str, outer_img_path: str, exception_config,
                plot: bool = False, z: float = 0) -> List['Fascicle']:
        """
        Generates list of fascicle objects from an inner and an outer image
        Example usage:
            fascicles = Fascicle.to_list(my_inner_image_path,
                                                  my_outer_image_path, ... )

        :param z:
        :param outer_img_path:
        :param inner_img_path:
        :param exception_config:
        :param plot: boolean
        :return: list of Fascicles derived from the image(s)
        if outer_img_path is None, returns a list of fascicles with inners only (inners stored as an outer)
        """

        def build_traces(path: str) -> List[Trace]:
            # default findContours params
            params = [cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE]
            # default findContours params

            img = np.flipud(cv2.imread(path, -1))

            if len(img.shape) > 2 and img.shape[2] > 1:
                img = img[:, :, 0]

            # find points of traces (hierarchy is IGNORED! ... see that "_")
            contours, _ = cv2.findContours(img, *params)

            # build list of traces
            return [Trace([item + [z] for item in contour[:, 0, :]], exception_config) for contour in contours]
        if outer_img_path is None:
            # inners only case, set each inner as an outer
            outers = np.array(build_traces(inner_img_path))
        else:
            # build traces list for inner and outer image paths
            inners, outers = (np.array(build_traces(path)) for path in (inner_img_path, outer_img_path))

            # create empty list to hold the outer traces that inners correspond to
            inner_correspondence: List[int] = []

            # iterate through all inner traces and assign outer index
            for inner in inners:
                mask = [inner.within(outer) for outer in outers]
                inner_correspondence.append(np.where(mask)[0][0])

        # create empty list to hold fascicles
        fascicles: List[Fascicle] = []

        if plot:
            plt.axes().set_aspect('equal', 'datalim')

        # iterate through each outer and build fascicles
        for index, outer in enumerate(outers):
            if outer_img_path is not None:
                # get all the inner traces that correspond to this outer
                inners_corresponding = inners[np.where(np.array(inner_correspondence) == index)]

                # add fascicle!
                fascicles.append(Fascicle(exception_config, outer, inners_corresponding))
            else:
                # inners only case
                fascicles.append(Fascicle(exception_config, outer))
            if plot:
                fascicles[index].plot()

        if plot:
            plt.show()

        return fascicles

    def write(self, mode: WriteMode, path: str):
        """
        :param mode: Sectionwise... for now
        :param path: root path of fascicle trace destination
        """

        start = os.getcwd()

        if not os.path.exists(path):
            self.throw(25)
        else:
            # go to directory to write to
            os.chdir(path)

            # keep track of starting place
            sub_start = os.getcwd()

            # write outer and inners
            for items, folder in [([self.outer], 'outer'), (self.inners, 'inners')]:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                os.chdir(folder)

                # write all items (give filename as i (index) without the extension
                for i, item in enumerate(items):
                    item.write(mode, os.path.join(os.getcwd(), str(i)))

                # change directory back to starting place
                os.chdir(sub_start)

        os.chdir(start)

    def morphology_data(self):
        inners = [{"area": inner.area(),
                   "x": inner.ellipse()[0][0],
                   "y": inner.ellipse()[0][1],
                   "a": inner.ellipse()[1][0],
                   "b": inner.ellipse()[1][1],
                   "angle": inner.ellipse()[2]}
                  for inner in self.inners]
        outer = {"area": self.outer.area(),
                 "x": self.outer.ellipse()[0][0],
                 "y": self.outer.ellipse()[0][1],
                 "a": self.outer.ellipse()[1][0],
                 "b": self.outer.ellipse()[1][1],
                 "angle": self.outer.ellipse()[2]}

        return {"outer": outer, "inners": inners}
