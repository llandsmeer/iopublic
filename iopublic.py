# if you want a specific arbor load location after a custom cmake build
# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
# import sys
# import importlib
# ARBOR_LOCATION = '/usr/local/lib/python3.8/dist-packages/arbor/__init__.py'
# spec = importlib.util.spec_from_file_location('arbor', ARBOR_LOCATION)
# module = importlib.util.module_from_spec(spec)
# sys.modules[spec.name] = module
# spec.loader.exec_module(module)

import os
import math
import json
import collections
import numpy as np
import arbor

ARBOR_BUILD_CATALOGUE = 'arbor-build-catalogue'

def compile_smol_model():
    '''Builds the Inferior Olive mechanisms in the smol_model directory

    If we don't need to rebuild because the mod file have not changed we
    do not recompile. An arbor catalogue for the smol model is returned.
    '''
    if hasattr(arbor, 'smol_catalogue'):
        # if you are using a local copy
        return arbor.smol_catalogue()
    import glob
    import subprocess
    expected_fn = './smol-catalogue.so'
    if os.path.exists(expected_fn):
        needs_recompile = False
        for src in glob.glob('smol_model/*.mod'):
            if os.path.getmtime(src) > os.path.getmtime(expected_fn):
                print(src, 'is newer than compiled library')
                needs_recompile = True
        if not needs_recompile:
            return arbor.load_catalogue(expected_fn)
    res = subprocess.getoutput(f'{ARBOR_BUILD_CATALOGUE} -g cuda smol smol_model')
    path = res.split()[-1]
    print(res)
    assert path[0] == '/' and path.endswith('.so')
    return arbor.load_catalogue(path)

def load_spacefilling_network(filename):
    '''Loads the network morphology json/gz given by the filename

    The file is just a regular json file but we wrap it in a SimpleNamespace
    we can use dot notation (obj.attr) to access attributes instead of obj['attr'].
    '''
    import json
    import gzip
    from types import SimpleNamespace
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as f:
            network = json.load(f, object_hook=lambda d:SimpleNamespace(**d))
    else:
        with open(filename) as f:
            network = json.load(f, object_hook=lambda d:SimpleNamespace(**d))
    return network

class TunedIOModel(arbor.recipe):
    def __init__(self, network_json, tuning, spikes=(), noise=(('sigma', 0),)):
        '''Arbor recipe for a fully tuned and self-contained IO model

        network_json: result of load_spacefilling_network(<filename>)
        tuning: result of json.load'ing a tuning file (containing gmax scalings)
        spikes: dict or list of tuples containing (timestep_ms,weight_uS) pairs
        noise: dict or list of tuples that define the noise component (needs ou_noise mech)
        '''
        arbor.recipe.__init__(self)
        self.neurons = network_json.neurons
        self.soma = []
        for neuron in self.neurons:
            self.soma.append((neuron.x, neuron.y, neuron.z))
        self.soma = np.array(self.soma)
        # idmap: scaffold id to gid map
        self.idmap = {neuron.old_id:new_id for new_id, neuron in enumerate(self.neurons)}
        self.props = arbor.neuron_cable_properties()
        self.props.catalogue.extend(compile_smol_model(), '')
        self.tuning = tuning
        self.ggap = tuning['ggap']
        self.spikes = dict(spikes)
        self.noise = dict(noise)

    def cell_kind(self, gid): return arbor.cell_kind.cable
    def connections_on(self, gid): return []
    def num_targets(self, gid): return 0
    def num_sources(self, gid): return 0
    def event_generators(self, gid): return []
    def probes(self, gid): return [arbor.cable_probe_membrane_voltage('"root"')]
    def global_properties(self, kind): return self.props

    def num_cells(self):
        return len(self.neurons)

    def cell_description(self, gid):
        neuron = self.neurons[gid]
        mod = self.tuning['mods'][gid]
        gjs = self.gap_junctions_on(gid, just_ids=True)
        cell = make_space_filling_neuron(neuron, mod=dict(mod), gjs=gjs, noise=self.noise)
        return cell

    def num_gap_junction_sites(self, gid):
        return len(self.neurons[gid].traces)

    def gap_junctions_on(self, gid, just_ids=False):
        conns = []
        for trace in self.neurons[gid].traces:
            local = f'gj{trace.local_from}'
            peer  = self.idmap[trace.global_to], f'gj{trace.local_to}'
            assert self.idmap[trace.global_from] == gid
            if just_ids:
                conn = local, peer
            else:
                conn = arbor.gap_junction_connection(local=local, peer=peer, weight=self.ggap)
            conns.append(conn)
        return conns

    def event_generators(self, gid):
        if not self.spikes: return []
        events = []
        for at, weight in self.spikes.items():
            if hasattr(weight, 'len') and len(weight) == 5:
                x, y, z, r, weight = weight
                nx, ny, nz = self.pos[gid]
                w0 = np.exp(-((x-nx)**2 + (y-ny)**2 + (z-nz)**2)/r**2)
                ev = arbor.event_generator('syn', w0 * weight, arbor.explicit_schedule([at]))
            else:
                ev = arbor.event_generator('syn', weight, arbor.explicit_schedule([at]))
            events.append(ev)
        return [
            arbor.event_generator('syn', weight, arbor.explicit_schedule([at]))
            for at, weight in self.spikes.items()]

def mkdecor(mod=()): # mod is read only
    '''
    mod keys can be things like

    scal_cal = 1.0 # scale gmax
    scal_ks = 0.5
    override_cal = calpid/global=1 # different mechanism and/or global
    cal.stopAfter = 10 # mech param
    '''
    SOMA = 'soma_group'
    DEND = 'dendrite_group'
    AXON = 'axon_group'

    mod = dict(mod)
    decor = arbor.decor()

    def mech(group, name, value, alt_name=False):
        gmax = mod.pop(name, value)*mod.pop(f'scal_{alt_name or name}', 1)
        mechname = mod.pop(f'overide_{alt_name or name}', name)
        params = dict(gmax=gmax)
        prefix = f'{alt_name or name}.'
        extra_params = set(k for k in mod if k.startswith(prefix))
        for k in extra_params:
            param_name = k.replace(prefix, '')
            param_value = mod.pop(k)
            params[param_name] = param_value
        decor.paint(f'"{group}"', arbor.density(mechname, params))

    mech(SOMA, 'na_s', 0.040)
    mech(SOMA, 'kdr',  0.030)
    mech(SOMA, 'k',    0.015 * 1.2, 'ks')
    mech(SOMA, 'cal',  0.030 * 1.2)
    mech(DEND, 'cah',  0.010 * 1.7 / 2)
    mech(DEND, 'kca',  0.200 * 0.7 * 1.5)
    mech(DEND, 'h',    0.025 * 1.7)
    mech(DEND, 'cacc', 0.007)
    mech(AXON, 'na_a', 0.250)
    mech(AXON, 'k',    0.200)
    #decor.paint('"all"', arbor.mechanism('ca_conc'))
    decor.paint('"dendrite_group"', arbor.density('ca_conc'))
    decor.paint('"all"', arbor.density('leak', dict(gmax=mod.pop('scal_leak', 1)*mod.pop('leak', 1.3e-05)  )))
    decor.set_property(cm=0.01) # F/m2
    Vm = mod.pop('Vm', -65)
    Vdend = mod.pop('Vdend', Vm)
    decor.set_property(Vm=Vm) # mV
    decor.paint(f'"{DEND}"', Vm=Vdend)
    decor.paint(f'"{SOMA}"', rL=100) # Ohm cm
    decor.paint(f'"{DEND}"', rL=100) # Ohm cm
    decor.paint(f'"{AXON}"', rL=100) # Ohm cm

    if mod:
        raise Exception(f'leftover config {mod}')

    return decor

def make_space_filling_neuron(neuron, mod=(), gjs=(), noise=()):
    '''Build a single IO cell given a neuron morphology and mechanism params
    '''
    mod = dict(mod)
    tree = arbor.segment_tree()

    def cable(*, length, radius, parent, tag):
        if isinstance(radius, (float, int)):
            radius = [radius, radius]
        return tree.append(
            parent,
            arbor.mpoint(x=0, y=0, z=0, radius=radius[0]),
            arbor.mpoint(x=length, y=0, z=0, radius=radius[1]),
            tag=tag)

    s = cable(length=12, radius=6, parent=arbor.mnpos, tag=1)
    a = cable(length=20, radius=[2.5, 1.5], parent=s, tag=2)

    segments = list(sorted(neuron.tree, key=lambda seg:seg.seg_id))
    seg_to_cable = {0: s}

    labels = arbor.label_dict()
    for i, seg in enumerate(segments):
        if i == 0:
            continue # soma
        prev_seg = segments[seg.parent]
        prev_cable_id = seg_to_cable[seg.parent]
        length = math.sqrt((prev_seg.x-seg.x)**2 + (prev_seg.y-seg.y)**2 + (prev_seg.z-seg.z)**2)
        cable_id = cable(length=length, radius=1, parent=prev_cable_id, tag=3)
        seg_to_cable[i] = cable_id
        for gj in seg.gj:
            labels[f'gj{gj}'] = f'(on-components 1.0 (segment {cable_id}))'

    labels['soma_group'] = '(tag 1)'
    labels['axon_group'] = '(tag 2)'
    labels['dendrite_group'] = '(tag 3)'
    labels['dendrite_distal'] = '(tag 4)'
    labels['all'] = '(all)'
    labels['synapse_site'] = '(location 0 0.5)'
    labels['root'] = '(root)'

    gj_mech = mod.pop('gj', 'cx36')

    decor = mkdecor(mod)

    if noise != {'sigma': 0}:
        args = ','.join(f'{k}={v}' for k, v in noise.items())
        decor.paint('"soma_group"', arbor.density(f'ou_noise/{args}'))

    # place a synapse at root - should probably be a dendrite
    decor.place('"root"', arbor.synapse('expsyn'), 'syn')

    for local, peer in gjs:
        decor.place(f'"{local}"', arbor.junction(gj_mech), local)

    policy = arbor.cv_policy_max_extent(10) | arbor.cv_policy_single('"soma_group"')
    decor.discretization(policy)
    return arbor.cable_cell(tree, labels, decor)

def get_network_for_tuning(selected):
    fn_tuned = f'tuned_networks/{selected}'
    tuning = json.load(open(fn_tuned))
    fn_network = f'{tuning["network"]}.gz'
    network = load_spacefilling_network(fn_network)
    return network

def simulate_tuned_network(selected, tfinal=10000, dt=0.025, gpu_id=0, spikes=()):
    #selected = '2021-12-08-shadow_averages_0.01_0.8_d1666304-c6fc-4346-a55d-a99b3aad55be'
    fn_tuned = f'tuned_networks/{selected}'
    tuning = json.load(open(fn_tuned))
    for mod in tuning['mods']:
        cal = mod.pop('scal_cal', None) # oops misnamed this one
        if cal is not None:
            mod['cal'] = cal
    fn_network = f'{tuning["network"]}.gz'
    network = load_spacefilling_network(fn_network)
    recipe = TunedIOModel(network, tuning, spikes=spikes)
    context = arbor.context(threads=8, gpu_id=gpu_id)
    domains = arbor.partition_load_balance(recipe, context)
    sim = arbor.simulation(recipe, domains, context)
    handles = [sim.sample((gid, 0), arbor.regular_schedule(1)) for gid in range(recipe.num_cells())]
    sim.run(tfinal=tfinal, dt=dt)
    traces = [sim.samples(handle)[0][0].T for handle in handles]
    vsall = np.array([vs for t, vs in traces])
    return traces[0][0], vsall
