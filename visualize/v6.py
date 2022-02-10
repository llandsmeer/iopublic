import gzip
import json
import glob
import time
import vispy_tube
from vispy import gloo
from vispy import app
import numpy as np
from vispy.util.transforms import perspective, translate, rotate

SCALE = 250.0

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
            vv, ff = vispy_tube.mesh_tube(trace, 4.5)
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

MESH_VERT_SHADER = """#version 330
// simple vertex shader
// https://learnopengl.com/code_viewer_gh.php?code=src/2.lighting/2.1.basic_lighting_diffuse/2.1.basic_lighting.vs
attribute vec3 a_position;
attribute vec3 a_normal;
uniform   vec3 u_color;
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
varying out vec3 fragpos;
varying out vec3 anormal;
varying out vec3 v_color;
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
varying vec3 v_color;
void main() {
    float ambientStrength = 0.1;
    vec3 lightPos = vec3(0.0, 500.0, 0.0);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    vec3 ambient = vec3(1.0, 1.0, 1.0) * ambientStrength;
    vec3 norm = normalize(anormal);
    vec3 lightDir = normalize(lightPos - fragpos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 result = (ambient + diffuse) * v_color;
    gl_FragColor = vec4(result, 1.0);
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

        self.view = translate((0, 0, -5))
        self.model = np.eye(4, dtype=np.float32)
        self.theta = 0
        self.phi = 0
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 2.0, 10.0)

        # Set uniform and attribute
        self._program['a_position'] = gloo.VertexBuffer(vPosition - vPosition.mean(0))
        self._program['a_normal'] = gloo.VertexBuffer(vert_normals)
        self._program['neuronid'] = gloo.VertexBuffer(neuron_ids)
        self._program['u_model'] = self.model
        self._program['tubefact'] = gloo.VertexBuffer(tubefact)
        self._program['u_view'] = self.view
        self._program['u_projection'] = self.projection
        self._program['u_tex'] = gloo.Texture2D(vsoma_color, wrapping='repeat', interpolation='linear') # repeat

        self.programs = []
        for filename in glob.glob('../mesh/*.obj'):
            if 'MAO_left' in filename:
                continue
            mesh_vertices, mesh_normals = load_obj(filename)
            #color = np.array([0.4, 0.5, 0.6]) + np.random.random(3) * 0.1
            color = np.array([0.8, 0.8, 0.8]) + np.random.random(3) * 0.1
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

    def on_timer(self, event):
        self.phi += .2
        self.model = np.dot(rotate(self.phi, (0, 0, 1)),
                            rotate(90,       (1, 0, 0)))
        self._program['u_model'] = self.model
        for program in self.programs:
            program['u_model'] = self.model
        self.update()

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(-45.0, event.size[0] /
                                      float(event.size[1]), 2.0, 10.0)
        self._program['u_projection'] = self.projection
        for program in self.programs:
            program['u_projection'] = self.projection

    def on_draw(self, event):
        # gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=False)
        # gloo.set_depth_mask(False)
        elapsed = time.time() - self.start
        self._program['time'] = elapsed
        gloo.clear()
        gloo.gl.glEnable(gloo.gl.GL_DEPTH_TEST)
        gloo.gl.glEnable(0x809d) # msaa
        gloo.gl.glEnable(0x864F) # gl depth clamp
        gloo.gl.glEnable(gloo.gl.GL_BLEND);
        gloo.gl.glBlendFunc(gloo.gl.GL_SRC_ALPHA, gloo.gl.GL_ONE_MINUS_SRC_ALPHA);  

        for program in self.programs:
            program.draw('triangles')
        self._program.draw('triangles')


if __name__ == '__main__':
    canvas = Canvas()
    app.run()


