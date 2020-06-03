import os
import numpy as np
import matplotlib.pyplot as plt

sample = 80
model = 0
sim = 10

base_n_sim = os.path.join('samples', str(sample), 'models', str(model), 'sims', str(sim), 'n_sims')

inner = 46
fiber = 0
n_sims = [0, 4, 8, 12, 16, 20]
pve1 = os.path.join(base_n_sim, str(n_sims[0]), 'data', 'inputs', 'inner{}_fiber{}.dat'.format(inner, fiber))
dpve1 = np.loadtxt(pve1)
plt.plot(dpve1[1:], 'r-', label='p1')

fiberset = 0
fiber = inner
base_fiberset = os.path.join('samples', str(sample), 'models', str(model), 'sims', str(sim), 'potentials', str(fiberset))
fve1 = os.path.join(base_fiberset, '{}.dat'.format(fiber))
dfve1 = np.loadtxt(fve1)
plt.plot(dfve1[1:], 'g--', label='f1')
plt.legend()
plt.show()

print('done')
