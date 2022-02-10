import gzip
import json
import numpy as np
import glob
import vispy_tube
from multiprocessing import Pool

NPROC = 12

def mesh_neuron(neuron):
    vertices = []
    faces = []
    for nt in neuron['traces']:
        trace = []
        for i, seg in enumerate(nt['trace'][::2]):
            x, y, z = seg['x'], seg['y'], seg['z']
            trace.append((x, y, z))
        trace = np.array(trace)
        if len(trace) >= 3:
            vv, ff = vispy_tube.mesh_tube(trace)
            faces.append(ff + sum(map(len, vertices)))
            vertices.append(vv)
    if vertices:
        vertices = np.vstack(vertices)
        faces = np.vstack(faces)
        return vertices, faces
    else:
        return (), ()

network = None
def initializer(fn_network):
    global network
    with gzip.open(fn_network) as f:
        network = json.load(f)

def worker(worker_id):
    res = []
    for i, neuron in enumerate(network['neurons']):
        if i % NPROC == worker_id:
            dend = vv, ff = mesh_neuron(neuron)
            res.append((i, dend))
            with open(f'mesh/{i}.obj', 'w') as stream:
                for x, y, z in vv:
                    print(f'v {x:.2f} {y:.2f} {z:.2f}', file=stream)
                for f in ff:
                    print('f', *(f+1), file=stream)

    return res

if __name__ == '__main__':
    fn_network = '/home/llandsmeer/Repos/llandsmeer/iopublic/networks/7eff83d2-25a6-460d-ac5f-908305cc7a57.json.gz'

    out = []
    with Pool(NPROC, initializer, (fn_network,)) as pool:
        for part in pool.map(worker, range(NPROC)):
            out.extend(part)
    out.sort()
    for i, k in out:
        print(i, end= ' ')
