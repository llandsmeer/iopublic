import gzip
import json
import numpy as np
import vispy
import vispy.scene
import glob
from vispy.scene import visuals
from vispy.util.transforms import perspective, translate, rotate
from vispy.visuals.transforms import MatrixTransform

class DisplayNetwork:
    def __init__(self, network):
        with gzip.open(network) as f:
            self.network = network = json.load(f)
        self.soma = []
        self.levels = []
        for neuron in network['neurons']:
            x, y, z = neuron['x'], neuron['y'], neuron['z']
            self.soma.append((x, y, -z))
        self.soma = np.array(self.soma)
        self.center = center = self.soma.mean(0)
        self.soma -= self.center
        for neuron in network['neurons']:
            for nt in neuron['traces']:
                for i, seg in enumerate(nt['trace'][::2]):
                    x, y, z = seg['x'], seg['y'], seg['z']
                    x, y, z = x - center[0], y - center[0], -z - center[0]
                    if i > 0:
                        while len(self.levels) < i:
                            self.levels.append([])
                        lines = self.levels[i - 1]
                        lines.append((px, py, pz))
                        lines.append((x, y, z))
                    px, py, pz = x, y, z
        for i in range(len(self.levels)):
            self.levels[i] = np.array(self.levels[i])

    def draw(self, view):
        soma = self.soma
        levels = self.levels
        self.vis_soma = visuals.Markers(spherical=True)
        colors = np.random.random((len(soma), 3)) * np.array([0.1, 0.4, 0.9]) + np.array([0.9, 0.6, 0.1])
        self.vis_soma.set_data(soma, size=8, edge_width=0, face_color=colors)
        print(self.vis_soma)
        view.add(self.vis_soma)
        for i, level in enumerate(levels):
            c = np.exp(-(i+1)/3)
            lines = visuals.Line(connect='segments', antialias=True, width=4*c, color=(0.5, 0.6, 0.7, 0.5 + 0.5*c))
            lines.set_data(level)
            #view.add(lines)

    def update(self, vals):
        assert len(vals) == len(self.soma)
        colors = [(c,c,c) for c in vals]
        self.vis_soma.set_data(self.soma, size=8, edge_width=0, face_color=colors)

class DisplayMesh:
    def __init__(self, filename, center=np.zeros(3)):
        vertices = []
        faces = []
        with open(filename) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                elif parts[0] == 'v':
                    x, y, z = parts[1:4]
                    vertices.append((float(x), float(y), -float(z)))
                elif parts[0] == 'f':
                    a, b, c = parts[1:4]
                    a = int(a.split('//')[0]) - 1
                    b = int(b.split('//')[0]) - 1
                    c = int(c.split('//')[0]) - 1
                    faces.append((a, b, c))
        vertices = np.array(vertices)
        vertices -= center
        faces = np.array(faces)
        color = np.array([0.6, 0.7, 0.8]) + np.random.random(3) * 0.2
        color = color / 2
        self.mesh = visuals.Mesh(vertices, faces, shading='smooth', color=color)

    def draw(self, view):
        view.add(self.mesh)

vispy.use('Glfw')
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

network_id = '3447248c-68a1-4860-b512-39fa22a5fa86'
fn_network = f'/home/llandsmeer/Repos/llandsmeer/iopublic/networks/{network_id}.json.gz'
nw = DisplayNetwork(fn_network)
nw.draw(view)

#

for mesh in glob.glob('../mesh/*.obj'):
    print(mesh)
    if 'PO_left' in mesh:
        print(mesh)
        continue
    m = DisplayMesh(mesh, center=nw.center)
    m.draw(view)

#

old = view.camera
view.camera = 'turntable'
axis = visuals.XYZAxis(parent=view.scene)

phi = 0
view.camera.distance = 10

y = np.array([n['cluster'] for n in nw.network['neurons']]) / 10
def update(ev):
    global phi
    t = ev.elapsed
    x = np.linspace(1, 1.5, len(nw.soma))
    nw.update(0.5 + 0.5*np.sin((x+y)*t))
    view.camera.elevation = 0
    view.camera.azimuth = -t * 19
    print(view.camera.distance)
#   view.camera.transform = MatrixTransform(
#           rotate(180,(0, 1, 0)) @
#           rotate(phi,  (0, 0, 1)) @
#           rotate(90,(1, 0, 0)))

timer = vispy.app.Timer()
timer.connect(update)
timer.start(0)

if __name__ == '__main__':
    vispy.app.run()
