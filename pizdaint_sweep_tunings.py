import sys
import os
import json
import lzma
import base64
import numpy as np
import uuid

import iopublic

experiment = sys.argv[1]

outdir = '/scratch/snx3000/llandsme/simulations'

if experiment == 'decay1':
    i = int(sys.argv[2])
    tuned_networks = list(sorted(os.listdir(f'tuned_networks')))
    selected_tuning = tuned_networks[i]
    sim_args = dict(
        selected=selected_tuning,
        tfinal=10000,
        dt=0.025,
        gpu_id=0,
        spikes={
            5000: 0.01
        }
    )
elif experiment == 'decay2':
    selected = '2021-12-08-shadow_averages_0.001_0.5_3447248c-68a1-4860-b512-39fa22a5fa86'
    i = int(sys.argv[2])
    sim_args = dict(
        selected=selected,
        tfinal=10000,
        dt=0.025,
        gpu_id=0,
        spikes={
            5000: 0.01,
            5000 + 1 * i: 0.01
        }
    )
elif experiment == 'local1':
    selected = '2021-12-08-shadow_averages_0.001_0.5_3447248c-68a1-4860-b512-39fa22a5fa86'
    r = 300
    neurons = iopublic.get_network_for_tuning(selected).neurons
    a, b = np.random.randint(len(neurons), size=2)
    x0, y0, z0 = round(neurons[a].x, 1), round(neurons[a].y, 1), round(neurons[a].z, 1)
    x1, y1, z1 = round(neurons[b].x, 1), round(neurons[b].y, 1), round(neurons[b].z, 1)
    sim_args = dict(
        selected=selected,
        tfinal=10000,
        dt=0.025,
        gpu_id=0,
        spikes={
            5000: (x0, y0, z0, r, 0.01),
            5000: (x1, y1, z1, r, 0.01)
        }
    )
elif experiment == 'local2':
    selected = '2021-12-08-shadow_averages_0.001_0.5_3447248c-68a1-4860-b512-39fa22a5fa86'
    r = 100
    neurons = iopublic.get_network_for_tuning(selected).neurons
    a, b = np.random.randint(len(neurons), size=2)
    x0, y0, z0 = round(neurons[a].x, 1), round(neurons[a].y, 1), round(neurons[a].z, 1)
    x1, y1, z1 = round(neurons[b].x, 1), round(neurons[b].y, 1), round(neurons[b].z, 1)
    sim_args = dict(
        selected=selected,
        tfinal=10000,
        dt=0.025,
        gpu_id=0,
        spikes={
            5000: (x0, y0, z0, r, 0.01),
            5000: (x1, y1, z1, r, 0.01)
        }
    )
else:
    assert False

key = base64.urlsafe_b64encode(json.dumps(sim_args).encode('utf8')).decode('latin1')

if len(key) >= 250:
    filename = f'{outdir}/rand-{str(uuid.uuid4())}.npz'
else:
    filename = f'{outdir}/{key}.npz'

if os.path.exists(filename):
    print('skipping')
    exit()

t, vs = iopublic.simulate_tuned_network(**sim_args)

np.savez_compressed(filename, key=key, t=t, vs=vs)
