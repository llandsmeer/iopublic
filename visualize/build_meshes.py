
print('')

import gzip
import json
import numpy as np
import glob
import vispy_tube
from multiprocessing import Pool
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable
from vispy.geometry import create_sphere
import collections

NPROC = 12

def mesh_neuron(neuron):
    vertices = []
    faces = []
    for nt in neuron['traces']:
        trace = []
        trace.append((neuron['x'], neuron['y'], neuron['z']))
        for i, seg in enumerate(nt['trace'][::2]):
            x, y, z = seg['x'], seg['y'], seg['z']
            trace.append((x, y, z))
        trace = np.array(trace)
        if len(trace) >= 3:
            vv, ff = vispy_tube.mesh_tube(trace, 3)
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
    soma_data = create_sphere()
    vv_soma, ff_soma = soma_data.get_vertices(), soma_data.get_faces()
    per_cluster = collections.defaultdict(list)
    for i, neuron in enumerate(network['neurons']):
        c = neuron['cluster']
        if c % NPROC != worker_id:
            continue
        dend = vv_dend, ff_dend = mesh_neuron(neuron)
        res.append((i, dend))
        somapos = np.array([neuron['x'], neuron['y'], neuron['z']])
        vv = np.vstack([vv_dend, vv_soma * 5 + somapos])
        ff = np.vstack([ff_dend, ff_soma + len(vv_dend)])
        with open(f'mesh/c{c:03d}_n{i:04d}.obj', 'w') as stream:
            for x, y, z in vv:
                print(f'v {x:.2f} {y:.2f} {z:.2f}', file=stream)
            for f in ff:
                print('f', *(f+1), file=stream)
        per_cluster[c].append((vv, ff))
    for cluster, meshes in per_cluster.items():
        with open(f'mesh/c{cluster:03d}_all.obj', 'w') as stream:
            offset = 0
            for vv, ff in meshes:
                for x, y, z in vv:
                    print(f'v {x:.2f} {y:.2f} {z:.2f}', file=stream)
                for f in ff:
                    print('f', *(f+1+offset), file=stream)
                offset += len(vv)
    offset = 0
    with open(f'mesh/worker{worker_id}_all.obj', 'w') as stream:
        for cluster, meshes in per_cluster.items():
            for vv, ff in meshes:
                for x, y, z in vv:
                    print(f'v {x:.2f} {y:.2f} {z:.2f}', file=stream)
                for f in ff:
                    print('f', *(f+1+offset), file=stream)
                offset += len(vv)
    return res

if __name__ == '__main__':
    network_id = 'ada2023a-4377-409b-a5ce-02b6768ffe41'
    fn_network = f'/home/llandsmeer/repos/llandsmeer/iopublic/networks/{network_id}.json.gz'
    out = []
    with Pool(NPROC, initializer, (fn_network,)) as pool:
        for part in pool.map(worker, range(NPROC)):
            out.extend(part)
    out.sort()
    for i, k in out:
        print(i, end= ' ')
