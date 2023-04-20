import socket
if socket.gethostname() == 'henkdenktenk':
    import sys
    import importlib
    ARBOR_LOCATION = '/home/lennart/Repos/arbor-sim/arbor/build/python/arbor/__init__.py'
    spec = importlib.util.spec_from_file_location('arbor', ARBOR_LOCATION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import iopublic
    iopublic.ARBOR_BUILD_CATALOGUE = '/home/lennart/Repos/arbor-sim/arbor/build/arbor-build-catalogue'

import sys
import os
import random
import numpy as np
import scipy.signal
import json

import h5py
import lfpykit
import arbor
import multiprocessing

import fasteners

import iopublic

tuned_networks = list(sorted(os.listdir(f'tuned_networks')))

if len(sys.argv) == 1:
    for i, n in enumerate(tuned_networks):
        print(f'{i:03d} {n}')
    exit(1)


i = int(sys.argv[1])
selected = tuned_networks[i]

# selected = '2021-12-08-shadow_averages_0.01_0.8_d1666304-c6fc-4346-a55d-a99b3aad55be'

if len(sys.argv) >= 3:
    stim = sys.argv[2]
else:
    stim = 'none'

radial_vext_probes = 'probe' in stim
probe_radius = 10

if '--gpu1' in sys.argv:
    gpu_id = 1
else:
    gpu_id = 0

print(stim, selected, 'gpu=', gpu_id)

import socket
if socket.gethostname() == 'henkdenktenk':
    database_file = '/mnt/Data/llandsmeer/database_v2.h5'
else:
    database_file = '/scratch/snx3000/llandsme/database.h5'
lock_file = f'{database_file}.lock'
tfinal = 25000
dt = 0.005

def lock():
    return fasteners.InterProcessLock(lock_file)

with lock():
    with h5py.File(database_file, 'a') as f:
        h5key = f'{selected}/{stim}'
        print('checking', h5key)
        if h5key not in f.keys():
            print('making key', h5key)
            f.create_group(h5key)
        else:
            print('h5 key already exists, exiting now')
            exit(1)
os.sync()

# from https://github.com/LFPy/LFPykit/blob/master/examples/Example_Arbor_swc.ipynb
class ArborCellGeometry(lfpykit.CellGeometry):
    def __init__(self, p, cables):
        x, y, z, r = [], [], [], []
        CV_ind = np.array([], dtype=int)  # tracks which CV owns segment
        for i, m in enumerate(cables):
            segs = p.segments([m])
            for j, seg in enumerate(segs):
                x.append([seg.prox.x, seg.dist.x])
                y.append([seg.prox.y, seg.dist.y])
                z.append([seg.prox.z, seg.dist.z])
                r.append([seg.prox.radius, seg.dist.radius])
                CV_ind = np.r_[CV_ind, i]
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        z = np.array(z, dtype=float)
        d = 2*np.array(r, dtype=float)
        super().__init__(x=x, y=y, z=z, d=d)
        self._CV_ind = CV_ind


class ArborLineSourcePotential(lfpykit.LineSourcePotential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._get_transformation_matrix = super().get_transformation_matrix

    def get_transformation_matrix(self):
        M_tmp = self._get_transformation_matrix()
        n_CVs = np.unique(self.cell._CV_ind).size
        M = np.zeros((self.x.size, n_CVs))
        for i in range(n_CVs):
            inds = self.cell._CV_ind == i
            M[:, i] = M_tmp[:, inds] @ (self.cell.area[inds] / self.cell.area[inds].sum())
        return M

# 2 minutes
neurons = iopublic.get_network_for_tuning(selected).neurons

with lock():
    with h5py.File(database_file, 'a') as f:
        already_simulated = 'Vext' in f[h5key]
os.sync()

if already_simulated:
    with lock():
        with h5py.File(database_file, 'a') as f:
            time = np.array(f[h5key]['time'])
            vsall = np.array(f[h5key]['vsall'])
            Vext = np.array(f[h5key]['Vext'])
    os.sync()
else:
    tstart =  5000
    num_sources = 100

    def spiketrain(f_hz, a=0, b=tfinal):
        x = 0
        while True:
            interval = np.random.poisson(lam=1000/f_hz)
            x = x + interval
            if x > b:
                return
            if x >= a:
                yield x

    spikes = []
    for i in range(num_sources):
        tgt = random.choice(neurons)
        if 'gaba' in stim:
            at = tuple([at for at in spiketrain(100, tstart-500, tfinal) if not (0 < at % 1000 < 100)])
            spikes.append((at, [tgt.x, tgt.y, tgt.z, 75, 0.0000005, 'gaba']))
        if 'ghalf' in stim: # gaba
            at = tuple([at for at in spiketrain(100, tstart-500, tfinal) if not (0 < at % 4000 < 2000)])
            spikes.append((at, [tgt.x, tgt.y, tgt.z, 75, 0.0000005, 'gaba']))
        if 'ampa' in stim:
            at = tuple([at for at in spiketrain(10, tstart, tfinal) if (0 < at % 1000 < 100)])
            spikes.append((at, [tgt.x, tgt.y, tgt.z, 250, 0.005, 'ampa']))
        if 'spike1target' in stim:
            at = tuple(range(tstart, tfinal, 1000))
            at = tuple([t + np.random.poisson(20) for t in at])
            spikes.append((at, [tgt.x, tgt.y, tgt.z, 250, 0.005, 'ampa']))
        if 'figure2c' in stim:
            at = tuple(range(tstart, tfinal, 1000))
            at = tuple([t + np.random.beta(1.8, 20)*400 for t in at])
            spikes.append((at, [tgt.x, tgt.y, tgt.z, 250, 0.01, 'ampa']))
        if 'spike2full' in stim:
            at = tuple(range(tstart, tfinal, 1000))
            at = tuple([t + np.random.poisson(20) for t in at])
            spikes.append((at, [0.0005*np.random.random(), 'ampa']))
    if 'spike1s' in stim:
        at = tuple(range(tstart, tfinal, 1000))
        spikes.append((at, [0.005, 'ampa']))
    # save spikes
    with lock():
        with h5py.File(database_file, 'a') as f:
            dat = f[h5key].create_dataset('at', (len(spikes,)), dtype=h5py.vlen_dtype(np.dtype(float)))
            dw = f[h5key].create_dataset('w', (len(spikes,)), dtype=h5py.string_dtype(encoding='utf-8'))
            for i, (at, w) in enumerate(spikes):
                dat[i] = at
                dw[i] = json.dumps(w)
    os.sync()

    print('done')
    recipe = iopublic.build_recipe(
        selected,
        spikes=spikes
    )
    print('done')
    context = arbor.context(threads=8, gpu_id=gpu_id)
    domains = arbor.partition_load_balance(recipe, context)
    sim = arbor.simulation(recipe, domains, context)
    print('done')
    tmem_current_handles = [sim.sample((gid, 2), arbor.regular_schedule(tstart, 5, tfinal), arbor.sampling_policy.exact) for gid in range(recipe.num_cells())]
    stim_current_handles = [sim.sample((gid, 3), arbor.regular_schedule(tstart, 5, tfinal), arbor.sampling_policy.exact) for gid in range(recipe.num_cells())]
    handles = [sim.sample((gid, 0), arbor.regular_schedule(1)) for gid in range(recipe.num_cells())]

    # 12 minutes
    sim.run(tfinal=tstart, dt=dt)

    I_meta = [sim.samples(handle)[0][1] for handle in tmem_current_handles]

    # 2 minutes

    N_SAMPLES = 64 # along 1 dimension. values > 10 = slow
    soma = np.array([(a.x, a.y, a.z) for a in recipe.neurons]).T

    sx = np.linspace(soma[0].min(), soma[0].max(), N_SAMPLES)
    sy = np.linspace(soma[1].min(), soma[1].max(), N_SAMPLES)
    z0 = soma[2].mean()
    e = np.eye(3)

    Xplane, Yplane = np.meshgrid(sx, sy)
    X, Y, Z = (e[0]*Xplane.reshape(-1, 1) + e[1]*Yplane.reshape(-1, 1) + e[2]*z0).T
    if radial_vext_probes:
        X = np.array([ -613, -498,  -153,  -38,  -500, -500,   -500, -500])
        Y = np.array([-1173, -1173, -933, -813, -1000, -1000, -1000, -1000])
        Z = np.array([1200,  1200,  1200, 1200,  1200,  1000,  8000,  6000])
        X = np.concatenate([X+probe_radius*np.cos(i/8*2*np.pi) for i in range(8)])
        Y = np.concatenate([Y+probe_radius*np.sin(i/8*2*np.pi) for i in range(8)])
        Z = np.concatenate([Z for i in range(8)])
    lsps = []
    Ms = []
    geometries = []
    for gid in range(recipe.num_cells()):
        segtree = recipe.cell_morphology(gid)
        p = arbor.place_pwlin(arbor.morphology(segtree))
        cell_geometry = ArborCellGeometry(p, I_meta[gid])
        lsp = ArborLineSourcePotential(cell=cell_geometry, x=X, y=Y, z=Z)
        M = lsp.get_transformation_matrix()
        geometries.append(cell_geometry)
        lsps.append(lsp)
        Ms.append(M)
        if gid % 5 == 0:
            print(gid, end=' ', flush=True)
    print()

    with lock():
        with h5py.File(database_file, 'a') as f:
            f[h5key].create_dataset('Xplane', data=Xplane, compression='gzip')
            f[h5key].create_dataset('Yplane', data=Yplane, compression='gzip')
            f[h5key].create_dataset('X', data=X, compression='gzip')
            f[h5key].create_dataset('Y', data=Y, compression='gzip')
            f[h5key].create_dataset('Z', data=Z, compression='gzip')
    os.sync()

    # 1 minutes per 1000 ms
    # this loops prevents our GPU from running out of memory
    # in steps of 1 second, we simulate the network
    # each time, we clear the GPU memory, load back & reduce

    concat_vsall = []
    concat_Vext = []
    concat_time = []
    for tcurrent in range(tstart, tfinal, 1000):
        print(tcurrent, tcurrent+1000)
        # clear GPU memory
        sim.clear_samplers()
        # run simulation for 1 second
        sim.run(tfinal=tcurrent+1000, dt=dt)
        # get currents
        tmem_current_traces = [sim.samples(handle)[0] for handle in tmem_current_handles]
        stim_current_traces = [sim.samples(handle)[0] for handle in stim_current_handles]
        traces = [sim.samples(handle)[0][0].T for handle in handles]
        # get voltages
        vsall = np.array([vs for t, vs in traces])
        # get timestamps
        time = tmem_current_traces[0][0][:,0]
        # calculate total currents
        I_m  = [tmem[0][:,1:].T + stim[0][:,1:].T for tmem, stim in zip(tmem_current_traces, stim_current_traces)]
        # reduce to external potential
        V_ext = 0
        for gid in range(recipe.num_cells()):
            V_ext = V_ext + np.nan_to_num(Ms[gid] @ I_m[gid])
        V_ext = V_ext.reshape((N_SAMPLES, N_SAMPLES, -1))
        # save reduced values
        print(vsall.shape, np.isnan(vsall.ptp(1)).mean())
        concat_vsall.append(vsall)
        concat_Vext.append(V_ext)
        concat_time.append(time)
    time = np.concatenate(concat_time)
    vsall = np.hstack(concat_vsall)
    Vext = np.concatenate(concat_Vext, axis=2)

    with lock():
        with h5py.File(database_file, 'a') as f:
            f[h5key].create_dataset('time', data=time, compression='gzip')
            f[h5key].create_dataset('vsall', data=vsall, compression='gzip')
            f[h5key].create_dataset('Vext', data=Vext, compression='gzip')
    os.sync()
    #END IF STATEMENT already_simulated

def go(i, j, peaks_i, peaks_j):
    nosc_min = 20
    peaks_, ij = np.array(sorted([(p, 0) for p in peaks_i] + [(p, 1) for p in peaks_j])).T
    interleaved = (ij[:-1] != ij[1:])
    sync = []
    for region in np.split(np.arange(len(peaks_)), np.where(~interleaved)[0] + 1):
        if len(region) <= 2: continue
        if ij[region[0]] == 1: region = region[1:]
        if ij[region[-1]] == 1: region = region[:-1]
        if len(region) <= 2*nosc_min: continue # minsize
        pi = peaks_[region][0::2]
        pj = peaks_[region][1::2]
        d = (((pj - pi[:-1]) / np.diff(pi) * np.pi * 2) + np.pi) % (2*np.pi) - np.pi
        d = np.unwrap(d)
        fit = scipy.stats.linregress(np.arange(len(d)), d)
        zscore = abs(fit.slope / fit.stderr) if fit.stderr > 0 else 0
        if zscore < 2:
            sync.append((i, j, pi[0], pj[-1], zscore))
    return sync

tmp_ptp = vsall[:,-5000:].ptp(1)
peaks = [scipy.signal.find_peaks(vsall[i], prominence=0.1)[0] for i in range(len(vsall))]

todo = []
n = len(vsall)
#n = 400
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if np.isnan(tmp_ptp[i]) or tmp_ptp[i] < 1 or tmp_ptp[j] < 1:
            continue
        todo.append((i, j, peaks[i], peaks[j]))
print('tasks done')
with multiprocessing.Pool(64) as pool:
    res = pool.starmap(go, todo)
print('pool done')

sync = []
for l in res:
    sync.extend(l)

with lock():
    with h5py.File(database_file, 'a') as f:
        f[h5key].create_dataset('sync', data=sync, compression='gzip')
os.sync()
