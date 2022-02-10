import gzip
import json
import numpy as np
import vispy
import vispy.scene
import glob
from vispy.scene import visuals

class DisplayNetwork:
    def __init__(self, network):
        with gzip.open(network) as f:
            network = json.load(f)
        self.soma = []
        self.levels = []
        for neuron in network['neurons']:
            x, y, z = neuron['x'], neuron['y'], neuron['z']
            self.soma.append((x, y, z))
            for nt in neuron['traces']:
                for i, seg in enumerate(nt['trace'][::2]):
                    x, y, z = seg['x'], seg['y'], seg['z']
                    if i > 0:
                        while len(self.levels) < i:
                            self.levels.append([])
                        lines = self.levels[i - 1]
                        lines.append((px, py, pz))
                        lines.append((x, y, z))
                    px, py, pz = x, y, z
        for i in range(len(self.levels)):
            self.levels[i] = np.array(self.levels[i])
        self.soma = np.array(self.soma)

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
            view.add(lines)

    def update(self, vals):
        assert len(vals) == len(self.soma)
        colors = [(c,c,c) for c in vals]
        self.vis_soma.set_data(self.soma, size=8, edge_width=0, face_color=colors)

class DisplayMesh:
    def __init__(self, filename):
        vertices = []
        faces = []
        with open(filename) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                elif parts[0] == 'v':
                    x, y, z = parts[1:4]
                    vertices.append((float(x), float(y), float(z)))
                elif parts[0] == 'f':
                    a, b, c = parts[1:4]
                    a = int(a.split('//')[0]) - 1
                    b = int(b.split('//')[0]) - 1
                    c = int(c.split('//')[0]) - 1
                    faces.append((a, b, c))
        vertices = np.array(vertices)
        faces = np.array(faces)
        color = np.array([0.6, 0.6, 0.6]) + np.random.random(3) * 0.4
        self.mesh = visuals.Mesh(vertices, faces, shading='smooth', color=color)

    def draw(self, view):
        view.add(self.mesh)

vispy.use('Glfw')
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

network = '/home/llandsmeer/Repos/llandsmeer/iopublic/networks/7eff83d2-25a6-460d-ac5f-908305cc7a57.json.gz'
nw = DisplayNetwork(network)
nw.draw(view)

#

for mesh in glob.glob('../mesh/*.obj'):
    print(mesh)
    if 'MAO_left' in mesh:
        print(mesh)
        continue
    m = DisplayMesh(mesh)
    m.draw(view)

#

view.camera = 'turntable'
axis = visuals.XYZAxis(parent=view.scene)

def update(ev):
    t = ev.elapsed
    x = 8 * np.linspace(1, 1.5, len(nw.soma)) * t
    nw.update(0.5 + 0.5*np.sin(x))

timer = vispy.app.Timer()
timer.connect(update)
timer.start(0)

if __name__ == '__main__':
    vispy.app.run()
