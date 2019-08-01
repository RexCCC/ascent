#!/usr/bin/env python3.7

"""
File:       slide.py
Author:     Jake Cariello
Created:    July 24, 2019

Description:

    OVERVIEW

    INITIALIZER

    PROPERTIES

    METHODS

"""

# TODO: add more documentation for reposition fascicles

import itertools
from typing import List, Tuple, Union
from shapely.geometry import LineString, Point
from shapely.affinity import scale
import numpy as np
import random
import matplotlib.pyplot as plt

# really weird syntax is required to directly import the class without going through the pesky init
from .fascicle import Fascicle
from .nerve import Nerve
from .trace import Trace
from src.utils import *


class Slide(Exceptionable, Configurable):

    def __init__(self, fascicles: List[Fascicle], nerve: Nerve, master_config: dict, exception_config: list,
                 will_reposition: bool = False):

        # init superclasses
        Configurable.__init__(self, SetupMode.OLD, ConfigKey.MASTER, master_config)
        Exceptionable.__init__(self, SetupMode.OLD, exception_config)

        self.nerve: Nerve = nerve
        self.fascicles: List[Fascicle] = fascicles

        if not will_reposition:
            # do validation (default is specific!)
            self.validation()

    def validation(self, specific: bool = True, die: bool = True, tolerance: float = None):

        if specific:
            if self.fascicle_fascicle_intersection():
                self.throw(10)

            if self.fascicle_nerve_intersection():
                self.throw(11)

            if self.fascicles_outside_nerve():
                self.throw(12)

        else:
            if any([self.fascicle_fascicle_intersection(), self.fascicle_nerve_intersection(),
                    self.fascicles_outside_nerve(), self.fascicles_too_close(tolerance)]):
                if die:
                    self.throw(13)
                else:
                    return False
            else:
                return True

    def fascicles_too_close(self, tolerance: float = None):
        """
        :param tolerance:
        :return:
        """

        if tolerance is None:
            return False
        else:
            pairs = itertools.combinations(self.fascicles, 2)
            return any([first.min_distance(second) < tolerance for first, second in pairs]) or \
                any([fascicle.min_distance(self.nerve) < tolerance for fascicle in self.fascicles])

    def fascicle_fascicle_intersection(self) -> bool:
        """
        :return: True if any fascicle intersects another fascicle, otherwise False
        """
        pairs = itertools.combinations(self.fascicles, 2)
        return any([first.intersects(second) for first, second in pairs])

    def fascicle_nerve_intersection(self) -> bool:
        """
        :return: True if any fascicle intersects the nerve, otherwise False
        """
        return any([fascicle.intersects(self.nerve) for fascicle in self.fascicles])

    def fascicles_outside_nerve(self) -> bool:
        """
        :return: True if any fascicle lies outside the nerve, otherwise False
        """
        return any([not fascicle.within_nerve(self.nerve) for fascicle in self.fascicles])

    def to_circle(self):
        """
        :return:
        """
        self.nerve = self.nerve.to_circle()

    def to_ellipse(self):
        """
        :return:
        """
        self.nerve = self.nerve.to_ellipse()

    def move_center(self, point: np.ndarray):
        """
        :param point: the point of the new slide center
        """
        # get shift from nerve centroid and point argument
        shift = list(point - np.array(self.nerve.centroid())) + [0]

        # apply shift to nerve trace and all fascicles
        self.nerve.shift(shift)
        for fascicle in self.fascicles:
            fascicle.shift(shift)

    def reshaped_nerve(self, mode: ReshapeNerveMode) -> Nerve:
        """
        :param mode:
        :return:
        """
        if mode == ReshapeNerveMode.CIRCLE:
            return self.nerve.deepcopy().to_circle()
        elif mode == ReshapeNerveMode.ELLIPSE:
            return self.nerve.deepcopy().to_ellipse()
        else:
            self.throw(16)

    def reposition_fascicles(self, new_nerve: Nerve, minimum_distance: float = 10, seed: int = None):
        """
        :param new_nerve:
        :param minimum_distance:
        :param seed:
        :return:
        """
        self.plot(final=False, fix_aspect_ratio=True)

        # seed the random number generator
        if seed is not None:
            random.seed(seed)

        def random_permutation(iterable, r=None):
            "Random selection from itertools.permutations(iterable, r)"
            pool = tuple(iterable)
            r = len(pool) if r is None else r
            return tuple(random.sample(pool, r))

        def jitter(first: Fascicle, second: Union[Fascicle, Nerve]):

            fascicles_to_jitter = [first]
            if isinstance(second, Fascicle):
                fascicles_to_jitter.append(second)
                angle = first.angle_to(second)
            else:
                _, points = first.min_distance(second, return_points=True)
                angle = Trace.angle(*[point.coords[0] for point in points])

            factor = -1
            angle_magnitude = 0 #((random.random() * 2) - 1) * (2 * np.pi) / 10
            for f in fascicles_to_jitter:
                step_scale = 1
                if isinstance(second, Fascicle) and [f.outer.within(h.outer) for h in (first, second) if h is not f][0]:
                    step_scale *= -20

                step_magnitude = random.random() * minimum_distance

                step = list(np.array([np.cos(angle), np.sin(angle)]) * step_magnitude)

                f.shift([step_scale * factor * item for item in step] + [0])
                f.rotate(factor * angle_magnitude)

                if not f.within_nerve(new_nerve):
                    # for _ in range(2):
                    f.shift([step_scale * -factor * item for item in step] + [0])

                factor *= -1




            # # to decide whether or not to move second argument
            # move_second: bool
            #
            # step_factor = -1
            # second_step_factor = 1
            #
            # # based on second argument type, get angle
            # if isinstance(second, Fascicle):
            #     move_second = True
            #     angle = first.angle_to(second)  # add random?
            #
            #     if first.outer.within(second.outer):
            #         step_factor = -20
            #         second_step_factor = 0
            #
            # else:  # second must be a Nerve
            #     move_second = False
            #     _, points = first.min_distance(second, return_points=True)
            #
            #     angle = Trace.angle(*[point.coords[0] for point in points])
            #
            #     if not first.within_nerve(second):
            #         step_factor = 20
            #
            # # create step size
            # step_magnitude = random.random() * minimum_distance
            # step = list(np.array([np.cos(angle), np.sin(angle)]) * step_magnitude)
            #
            # angle_magnitude = ((random.random() * 2) - 1) * (2 * np.pi) / 100
            #
            # first_step = [step_factor * item for item in step] + [0]

            # shift input arguments
            # first.shift(first_step)  # negative shift for first item
            # first.rotate(angle_magnitude)
            # if move_second:
            #     # second_in_before = second.within_nerve(new_nerve)
            #     second_step = [second_step_factor * item for item in step] + [0]
            #     second.shift(second_step)  # positive shift for second item
            #     second.rotate(-angle_magnitude)


        # Initial shift - proportional to amount of change in the nerve boundary and distance of
        # fascicle centroid from nerve centroid

        for i, fascicle in enumerate(self.fascicles):
            # print('fascicle {}'.format(i))

            fascicle_centroid = fascicle.centroid()
            new_nerve_centroid = new_nerve.centroid()
            r_fascicle_initial = LineString([new_nerve_centroid, fascicle_centroid])

            r_mean = new_nerve.mean_radius()
            r_fasc = r_fascicle_initial.length
            a = 3  # FIXME:
            exterior_scale_factor = a * (r_mean / r_fasc)
            exterior_line: LineString = scale(r_fascicle_initial,
                                              *([exterior_scale_factor] * 3),
                                              origin=new_nerve_centroid)

            # plt.plot(*new_nerve_centroid, 'go')
            # plt.plot(*fascicle_centroid, 'r+')
            # new_nerve.plot()
            # plt.plot(*np.array(exterior_line.coords).T)
            # plt.show()

            new_intersection = exterior_line.intersection(new_nerve.polygon().boundary)
            old_intersection = exterior_line.intersection(self.nerve.polygon().boundary)
            # nerve_change_vector = LineString([new_intersection.coords[0], old_intersection.coords[0]])

            # plt.plot(*np.array(nerve_change_vector.coords).T)
            # self.nerve.plot()
            # new_nerve.plot()

            # get radial vector to new nerve trace
            r_new_nerve = LineString([new_nerve_centroid, new_intersection.coords[0]])

            # get radial vector to FIRST coordinate intersection of old nerve trace
            if isinstance(old_intersection, Point):  # simple Point geometry
                r_old_nerve = LineString([new_nerve_centroid, old_intersection.coords[0]])
            else:  # more complex geometry (MULTIPOINT)
                r_old_nerve = LineString([new_nerve_centroid, list(old_intersection)[0].coords[0]])

            fascicle_scale_factor = (r_new_nerve.length/r_old_nerve.length) * 0.75

            # TODO: nonlinear scaling of fascicle_scale_factor
            # if fascicle_scale_factor > 1:
            #     fascicle_scale_factor **= exponent

            r_fascicle_final = scale(r_fascicle_initial,
                                     *([fascicle_scale_factor] * 3),
                                     origin=new_nerve_centroid)

            shift = list(np.array(r_fascicle_final.coords[1]) - np.array(r_fascicle_initial.coords[1])) + [0]
            fascicle.shift(shift)
            # fascicle.plot('r-')

            # attempt to move in direction of closest boundary
            _, min_dist_intersection_initial = fascicle.centroid_distance(self.nerve, return_points=True)
            _, min_dist_intersection_final = fascicle.centroid_distance(new_nerve, return_points=True)
            min_distance_length = LineString([min_dist_intersection_final[1].coords[0],
                                              min_dist_intersection_initial[1].coords[0]]).length
            min_distance_vector = np.array(min_dist_intersection_final[1].coords[0]) - \
                                  np.array(min_dist_intersection_initial[1].coords[0])
            min_distance_vector *= 1

            # fascicle.shift(list(-min_distance_vector) + [0])

        # NOW, set the slide's actual nerve to be the new nerve
        self.nerve = new_nerve

        # Jitter
        iteration = 0
        print('START random jitter')
        while not self.validation(specific=False, die=False, tolerance=None):
            iteration += 1
            plt.figure()
            self.plot(final=True, fix_aspect_ratio=True, inner_format='r-')
            plt.title('iteration: {}'.format(iteration - 1))
            plt.show()
            # raise Exception('')  # FIXME: TEMPORARY STOP TO VIEW GRAPH
            print('\titeration: {}'.format(iteration))
            for fascicle in random_permutation(self.fascicles):
                while fascicle.min_distance(self.nerve) < minimum_distance:
                    jitter(fascicle, self.nerve)

                for other_fascicle in random_permutation(filter(lambda item: item is not fascicle, self.fascicles)):
                    while fascicle.min_distance(other_fascicle) < minimum_distance or \
                            fascicle.outer.within(other_fascicle.outer):
                        jitter(fascicle, other_fascicle)

        print('END random jitter')

        # validate again just for kicks
        self.validation()

        self.plot('CHANGE', inner_format='r-')

    def plot(self, title: str = None, final: bool = True, inner_format: str = 'b-', fix_aspect_ratio: bool = False):
        """
        Quick util for plotting the nerve and fascicles
        :param title: optional string title for plot
        :param final: optional, if False, will not show or add title (if comparisons are being overlayed)
        :param inner_format: optional format for inner traces of fascicles
        :param fix_aspect_ratio: optional, if True, will set equal aspect ratio
        """
        # if not the last graph plotted
        if fix_aspect_ratio:
            plt.axes().set_aspect('equal', 'datalim')

        # loop through constituents and plot each
        self.nerve.plot(plot_format='g-')
        for fascicle in self.fascicles:
            fascicle.plot(inner_format)

        # if final plot, add title and show
        if final:
            if title is not None:
                plt.title(title)

            plt.show()

    def scale(self, factor: float):
        """
        :param factor:
        :return:
        """
        center = list(self.nerve.centroid())

        self.nerve.scale(factor, center)
        for fascicle in self.fascicles:
            fascicle.scale(factor, center)

