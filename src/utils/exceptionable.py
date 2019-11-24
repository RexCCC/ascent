#!/usr/bin/env python3.7

"""
Description:

    OVERVIEW
    Centralized way to organize and "throw" exceptions.

    INITIALIZER
    See docstring of __init__.

    PROPERTIES
    none, but creates key/value in configs (inherited form Configurable) that pertains to exception data

    METHODS
    throw

"""

from .configurable import Configurable
from .enums import SetupMode, ConfigKey
import os


class Exceptionable(Configurable):

    def __init__(self, mode: SetupMode, config=None):
        """
        :param mode: SetupMode, determines if Configurable loads new JSON or uses old data
        :param config: if SetupMode.OLD, this is the data. if SetupMode.NEW, this is str path to JSON
        """

        if mode == SetupMode.OLD:
            Configurable.__init__(self, mode, ConfigKey.EXCEPTIONS, config)
        else:  # mode == SetupMode.NEW
            Configurable.__init__(self, mode, ConfigKey.EXCEPTIONS, os.path.join('config', 'system', 'exceptions.json'))

    def throw(self, code):
        """
        Use this to throw an exception

        example:
            if FATAL_CONDITION:
                self.throw(CODE)

        :param code: index of exception in json file (i.e. exceptions.json)
        :return: full message (with code and text)
        """

        # force to exception 0 if incorrect bounds
        if code not in range(1, len(self.configs[ConfigKey.EXCEPTIONS.value])):
            code = 0

        exception = self.configs[ConfigKey.EXCEPTIONS.value][code]
        # note that the json purposefully has the redundant entry "code"
        # this is done for ease of use and organizational purposes
        raise Exception('\n\tcode:\t{}\n'
                        '\ttext:\t{}\n'
                        '\tsource:\t{}'.format(exception.get('code'),
                                               exception.get('text'),
                                               exception.get('source')))
