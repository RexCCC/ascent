#!/usr/bin/env python3.7

"""
The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""

import os
import sys

sys.path.append(os.path.sep.join([os.getcwd(), '']))
import numpy as np

import matplotlib.pyplot as plt
from src.core.query import Query

# set default fig size
plt.rcParams['figure.figsize'] = list(np.array([16.8, 10.14*2]) / 2)

q = Query({
    'partial_matches': False,
    'include_downstream': True,
    'indices': {
        'sample': [0],
        'model': [0],
        'sim': [8]
    }
}).run()

q.ap_time_and_location(
    delta_V=60,
    plot=True,
    absolute_voltage=False,
    # n_sim_label_override='7.3 µm MRG Fiber',
    # model_labels=[
    #     '5000 µm model radius',
    #     '7500 µm model radius',
    #     '10000 µm model radius',
    # ],
    # n_sim_filter=[0, 1, 2],
    # save=True,
    # subplots = True,
    nodes_only = True)
