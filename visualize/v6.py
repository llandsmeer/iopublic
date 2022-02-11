import gzip
import json
import glob
import time
import vispy_tube
from vispy import gloo
from vispy import app
import numpy as np
from vispy.util.transforms import perspective, translate, rotate

import threading
from queue import Queue

SCALE = 250.0

def moveto(c, n, s):
    if abs(c - n) < s:
        return n
    if c > n:
        return c - s
    return c + s


def mesh_neuron(neuron):
    vertices = []
    faces = []
    for nt in neuron['traces']:
        trace = []
        for i, seg in enumerate(nt['trace']):
            x, y, z = seg['x'], seg['y'], seg['z']
            trace.append((x, y, z))
            if len(trace) > 20:
                continue
        trace = np.array(trace)
        if len(trace) >= 3:
            vv, ff = vispy_tube.mesh_tube(trace, 10)
            faces.append(ff + sum(map(len, vertices)))
            vertices.append(vv)
    if vertices:
        vertices = np.vstack(vertices)
        faces = np.vstack(faces)
        return vertices, faces
    else:
        return (), ()

fn_network = '/home/llandsmeer/Repos/llandsmeer/iopublic/networks/7eff83d2-25a6-460d-ac5f-908305cc7a57.json.gz'

with gzip.open(fn_network) as f:
    network = json.load(f)

try:
    f = np.load('saved.npz')
    vertices = np.array(f['vertices'])
    faces = np.array(f['faces'])
    neuron_ids = np.array(f['neuron_ids'])
    centers = np.array(f['centers'])
except:
    vertices = []
    faces = []
    neuron_ids = []
    centers = []
    for i, neuron in enumerate(network['neurons']):
        dend = vv, ff = mesh_neuron(neuron)
        with open(f'mesh/{i}.obj', 'w') as stream:
            for x, y, z in vv:
                print(f'v {x:.2f} {y:.2f} {z:.2f}', file=stream)
            for f in ff:
                print('f', *(f+1), file=stream)
        if len(ff) > 0:
            faces.append(ff + sum(map(len, vertices)))
            vertices.append(vv)
            centers.extend(((neuron['x'],neuron['y'],neuron['z']),)*len(vv))
            neuron_ids.extend((i,)*len(vv))
            #neuron_ids.extend((neuron['cluster'],)*len(vv))
            print(i)
    neuron_ids = np.array(neuron_ids)
    centers = np.array(centers)
    vertices = np.vstack(vertices)
    faces = np.vstack(faces)
    np.savez_compressed('saved.npz', neuron_ids = neuron_ids, centers = centers, vertices = vertices, faces = faces)

def get_normals(vv, ff):
    vert_normals = np.zeros(vv.shape, np.float32)
    face_normals = np.cross(vv[ff[:,1]] - vv[ff[:,0]], vv[ff[:,2]] - vv[ff[:,0]])
    vert_normals[ff[:,0]] += face_normals[:]
    vert_normals[ff[:,1]] += face_normals[:]
    vert_normals[ff[:,2]] += face_normals[:]
    norm = np.linalg.norm(vert_normals, axis=1)
    return vert_normals / norm.reshape(-1,1)

tubefact = np.exp(-np.linalg.norm(vertices - centers, axis=1)[faces.flatten()].astype(np.float32)**2 / 30**2)
neuron_ids = (neuron_ids[faces.flatten()] + 1) / (len(network['neurons']) + 1)
neuron_ids = neuron_ids.astype(np.float32)
vert_normals = get_normals(vertices, faces)
vert_normals = vert_normals[faces.flatten()].astype(np.float32)
v = vertices[faces.flatten()].astype(np.float32)
#v = v - v.mean(0)
v = v / SCALE
vPosition = v


VERT_SHADER = """#version 330
// simple vertex shader
// https://learnopengl.com/code_viewer_gh.php?code=src/2.lighting/2.1.basic_lighting_diffuse/2.1.basic_lighting.vs
attribute vec3 a_position;
attribute vec3 a_normal;
attribute float neuronid;
attribute float tubefact;
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
uniform   sampler2D u_tex;
uniform float time;
varying out vec3 anormal;
varying out vec3 fragpos;
varying out vec4 v_color;
void main (void) {
    v_color = texture2D(u_tex, vec2(neuronid, time/100.0)).rgba;
    v_color.a *= tubefact;
    vec4 pos = u_model * vec4(a_position, 1.0);
    fragpos = pos.xyz;
    anormal = mat3(transpose(inverse(u_model))) * a_normal;
    gl_Position = u_projection * u_view * pos;
}
"""

FRAG_SHADER = """ // simple fragment shader
// https://learnopengl.com/code_viewer_gh.php?code=src/2.lighting/2.1.basic_lighting_diffuse/2.1.basic_lighting.fs
in vec3 anormal;
in vec3 fragpos;
uniform float time;
varying vec4 v_color;
void main() {
    // if (fragpos.z > 2.0 || v_color.a < 0.05) { discard; }
    float ambientStrength = 0.2;
    vec3 lightPos = vec3(0.0, 500.0, 0.0);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    vec3 ambient = vec3(1.0, 1.0, 1.0) * ambientStrength;
    vec3 norm = normalize(anormal);
    vec3 lightDir = normalize(lightPos - fragpos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec4 result = vec4(ambient + diffuse, 1.0) * v_color;
    //result.a = 1.0;
    gl_FragColor = result;
}
"""

MESH_VERT_SHADER = """#version 330
// simple vertex shader
// https://learnopengl.com/code_viewer_gh.php?code=src/2.lighting/2.1.basic_lighting_diffuse/2.1.basic_lighting.vs
attribute vec3 a_position;
attribute vec3 a_normal;
uniform   vec4 u_color;
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
varying out vec3 fragpos;
varying out vec3 anormal;
varying out vec4 v_color;
void main (void) {
    vec4 pos = u_model * vec4(a_position, 1.0);
    fragpos = pos.xyz;
    anormal = mat3(transpose(inverse(u_model))) * a_normal;
    gl_Position = u_projection * u_view * pos;
    v_color = u_color;
}
"""

MESH_FRAG_SHADER = """ // simple fragment shader
// https://learnopengl.com/code_viewer_gh.php?code=src/2.lighting/2.1.basic_lighting_diffuse/2.1.basic_lighting.fs
in vec3 anormal;
in vec3 fragpos;
uniform float time;
varying vec4 v_color;
void main() {
    if (v_color.a < 0.05) { discard; }
    float ambientStrength = 0.1;
    vec3 lightPos = vec3(0.0, 500.0, 0.0);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    vec3 ambient = vec3(1.0, 1.0, 1.0) * ambientStrength;
    vec3 norm = normalize(anormal);
    vec3 lightDir = normalize(lightPos - fragpos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec4 result = vec4(ambient + diffuse, 1.0) * v_color;
    gl_FragColor = result;
}
"""

#vsoma_color = np.random.random((len(network['neurons']), 10000, 3)).astype(np.float32)
import matplotlib.pyplot
cmap = matplotlib.pyplot.get_cmap('GnBu')
#cmap = matplotlib.pyplot.get_cmap('hsv')
#vsoma = np.random.random((len(network['neurons']), 1000))
vsoma = np.array([ np.linspace(i, (2*i)%1.2+10, 1000)%1 for i in range(len(network['neurons'])) ])
vsoma = np.tanh((vsoma-0.5)*10)
x = (vsoma - vsoma.mean()) / vsoma.std()
x[x<0] = 0
x[x>1] = 1
vsoma_color = cmap(x) * 0.5 + 0.5
#vsoma_color[:,:,3] = 0.3 + 0.7*x**1.5
vsoma_color = vsoma_color.astype(np.float32)


#vsoma_color = np.random.random((1000,1000, 3)).astype(np.float32)

###

def load_obj(filename):
    vertices = []
    faces = []
    normals = []
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            elif parts[0] == 'v':
                x, y, z = parts[1:4]
                vertices.append((float(x), float(y), float(z)))
            elif parts[0] == 'vn':
                x, y, z = parts[1:4]
                normals.append((float(x), float(y), float(z)))
            elif parts[0] == 'f':
                a, b, c = parts[1:4]
                a = int(a.split('//')[0]) - 1
                b = int(b.split('//')[0]) - 1
                c = int(c.split('//')[0]) - 1
                faces.append((a, b, c))
    vertices = np.array(vertices) / SCALE
    faces = np.array(faces)
    normals = np.array(normals)
    #normals = get_normals(vertices, faces)
    vertices = vertices[faces.flatten()].astype(np.float32)
    normals = normals[faces.flatten()].astype(np.float32)
    return vertices, normals

###

class Canvas(app.Canvas):

    def __init__(self):
        super().__init__(keys='interactive')

        # Create program
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        self.view_dist = 8
        self.view = translate((0, 0, -self.view_dist))
        self.model = np.eye(4, dtype=np.float32)
        self.theta = 90
        self.phi = 0
        self.mesh_alpha = 1
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 2.0, 10.0)
        self.mode = 0
        self.angle = 0

        self.tube_triangles = vPosition - vPosition.mean(0)
        self.tube_normals = vert_normals
        self.tube_neuron_ids = neuron_ids
        self.tube_fact = tubefact

        # Set uniform and attribute
        self._program['a_position'] = gloo.VertexBuffer(self.tube_triangles)
        self._program['a_normal'] = gloo.VertexBuffer(self.tube_normals)
        self._program['neuronid'] = gloo.VertexBuffer(self.tube_neuron_ids)
        self._program['tubefact'] = gloo.VertexBuffer(self.tube_fact)
        self._program['u_model'] = self.model
        self._program['u_view'] = self.view
        self._program['u_projection'] = self.projection
        self._program['u_tex'] = gloo.Texture2D(vsoma_color, wrapping='repeat', interpolation='linear') # repeat

        self.programs = []
        for filename in glob.glob('../mesh/*.obj'):
            if 'MAO_left' in filename:
                continue
            mesh_vertices, mesh_normals = load_obj(filename)
            #color = np.array([0.4, 0.5, 0.6]) + np.random.random(3) * 0.1
            color = np.array([0.8, 0.8, 0.8, 1]) + np.random.random(4) * 0.1
            color[3] = 1
            program = gloo.Program(MESH_VERT_SHADER, MESH_FRAG_SHADER)
            program['u_color'] = color
            program['a_position'] = gloo.VertexBuffer(mesh_vertices - vPosition.mean(0))
            program['a_normal'] = gloo.VertexBuffer(mesh_normals)
            program['u_model'] = self.model
            program['u_view'] = self.view
            program['u_projection'] = self.projection
            self.programs.append(program)

        gloo.set_clear_color((0.08, 0.1, 0.1))

        self.start = time.time()
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()
        self.counter = 0

        self.reorder_thread = None
        self.reorder_queue = Queue()

    def on_key_press(self, event):
        if event.key.name == 'Space':
            self.mode = (self.mode + 1 ) % 3

    def reorder(self):
        glpos = (self.projection @ self.view @ self.model) @ np.hstack([
            self.tube_triangles, np.ones((self.tube_triangles.shape[0], 1))]).T
        z0 = glpos[3, 0::3]
        z1 = glpos[3, 1::3]
        z2 = glpos[3, 2::3]
        z = np.min([z0, z1, z2], axis=0)
        o = 3 * z.argsort().repeat(3)
        o[1::3] += 1
        o[2::3] += 2
        a, b, c, d = self.tube_triangles[o], self.tube_normals[o], self.tube_neuron_ids[o], self.tube_fact[o]
        self.reorder_queue.put((a, b, c, d))

    def on_timer(self, event):
        if self.reorder_thread is None:
            t = threading.Thread(target=self.reorder)
            t.start()
            self.reorder_thread = t
        if not self.reorder_queue.empty():
            a, b, c, d = self.reorder_queue.get()
            self.reorder_thread = None
            self._program['a_position'].set_data(a)
            self._program['a_normal'].set_data(b)
            self._program['neuronid'].set_data(c)
            self._program['tubefact'].set_data(d)
        if self.mode == 0:
            self.angle = moveto(self.angle, 0, 2)
            self.mesh_alpha = moveto(self.mesh_alpha, 1, 0.04)
            self.view_dist = moveto(self.view_dist, 9, 0.1)
        elif self.mode == 1:
            self.angle = moveto(self.angle, 90, 2)
            self.mesh_alpha = moveto(self.mesh_alpha, 0, 0.04)
            self.view_dist = moveto(self.view_dist, 5, 0.1)
        elif self.mode == 2:
            self.angle = moveto(self.angle, 90, 2)
            self.mesh_alpha = moveto(self.mesh_alpha, 0, 0.04)
            self.view_dist = moveto(self.view_dist, 1, 0.1)
        self.view = translate((0, 0, -self.view_dist))
        self.phi += .2
        self.model = \
                    rotate(self.angle,(0, 1, 0)) @ \
                    rotate(self.phi,  (0, 0, 1)) @ \
                    rotate(self.theta,(1, 0, 0))
        self._program['u_model'] = self.model
        self._program['u_view'] = self.view
        for program in self.programs:
            program['u_model'] = self.model
            program['u_view'] = self.view
        self.update()
        self.counter += 1

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(-45.0, event.size[0] /
                                      float(event.size[1]), 2.0, 10.0)
        self._program['u_projection'] = self.projection
        for program in self.programs:
            program['u_projection'] = self.projection

    def on_draw(self, event):
        gloo.set_state(blend=True, depth_test=True,
             cull_face=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        elapsed = time.time() - self.start
        self._program['time'] = elapsed
        gloo.clear(depth=True, color=True)
        #gloo.gl.glClear(gloo.gl.GL_COLOR_BUFFER_BIT | gloo.gl.GL_DEPTH_BUFFER_BIT);
        #gloo.gl.glEnable(gloo.gl.GL_DEPTH_TEST)
        #gloo.gl.glEnable(0x809d) # msaa
        #gloo.gl.glEnable(0x864F) # gl depth clamp

        #gloo.gl.glDisable(gloo.gl.GL_BLEND);
        self._program.draw('triangles')
        #gloo.gl.glEnable(gloo.gl.GL_BLEND);
        #gloo.gl.glBlendFunc(gloo.gl.GL_SRC_ALPHA, gloo.gl.GL_ONE_MINUS_SRC_ALPHA);  
        gloo.set_state(cull_face=False)
        for program in self.programs:
            color = np.array(program['u_color']) 
            color[3] = self.mesh_alpha
            program['u_color'] = color
            program.draw('triangles')


if __name__ == '__main__':
    canvas = Canvas()
    app.run()


