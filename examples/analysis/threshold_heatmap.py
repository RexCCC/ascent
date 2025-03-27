"""Generate a heatmap of activation thresholds.

The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent.

Note: if more than one heatmap is desired, you must use a Seaborn FacetGrid.
RUN THIS FROM REPOSITORY ROOT
"""

import os
import sys

sys.path.append(os.path.sep.join([os.getcwd(), '']))

import matplotlib.pyplot as plt

from src.core.plotter import heatmaps
from src.core.query import Query

# Initialize and run Querys
q = Query(
    {
        'partial_matches': True,
        'include_downstream': True,
        'indices': {'sample': [1], 'model': [0], 'sim': [1]},
    }
).run()

# Build heatmap
data=q.threshold_data(ignore_missing=True)
# write filter logic particular examing how many n_sim and create a for loop to go through each of them
heatmaps(data=data)
plt.title('Activation threshold heatmap')

save_directory = os.path.join('out', 'analysis')
os.makedirs(save_directory, exist_ok=True)
plt.savefig(os.path.join(save_directory, 'threshold_heatmap.png'), dpi=400, bbox_inches='tight')
