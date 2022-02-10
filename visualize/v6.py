import gzip
import json
import glob
import time
import vispy_tube
from vispy import gloo
from vispy import app
import numpy as np
from vispy.util.transforms import perspective, translate, rotate

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
            vv, ff = vispy_tube.mesh_tube(trace, 1.5)
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

vertices = []
faces = []
neuron_ids = []
for i, neuron in enumerate(network['neurons']):
    #if i % 100 != 0: continue
    dend = vv, ff = mesh_neuron(neuron)
    with open(f'mesh/{i}.obj', 'w') as stream:
        for x, y, z in vv:
            print(f'v {x:.2f} {y:.2f} {z:.2f}', file=stream)
        for f in ff:
            print('f', *(f+1), file=stream)
    if len(ff) > 0:
        faces.append(ff + sum(map(len, vertices)))
        vertices.append(vv)
        neuron_ids.extend((i,)*len(vv))
        #neuron_ids.extend((neuron['cluster'],)*len(vv))
        print(i)
neuron_ids = np.array(neuron_ids)
vertices = np.vstack(vertices)
faces = np.vstack(faces)

def get_normals(vv, ff):
    vert_normals = np.zeros(vv.shape, np.float32)
    face_normals = np.cross(vv[ff[:,1]] - vv[ff[:,0]], vv[ff[:,2]] - vv[ff[:,0]])
    vert_normals[ff[:,0]] += face_normals[:]
    vert_normals[ff[:,1]] += face_normals[:]
    vert_normals[ff[:,2]] += face_normals[:]
    norm = np.linalg.norm(vert_normals, axis=1)
    return vert_normals / norm.reshape(-1,1)

neuron_ids = (neuron_ids[faces.flatten()] + 1) / (len(network['neurons']) + 1)
neuron_ids = neuron_ids.astype(np.float32)
vert_normals = get_normals(vertices, faces)
vert_normals = vert_normals[faces.flatten()].astype(np.float32)
v = vertices[faces.flatten()].astype(np.float32)
#v = v - v.mean(0)
v = v / 1000
vPosition = v


VERT_SHADER = """#version 330
// simple vertex shader
// https://learnopengl.com/code_viewer_gh.php?code=src/2.lighting/2.1.basic_lighting_diffuse/2.1.basic_lighting.vs
attribute vec3 a_position;
attribute vec3 a_normal;
attribute float neuronid;
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
uniform   sampler2D u_tex;
uniform float time;
varying out vec3 anormal;
varying out vec3 fragpos;
varying out vec3 v_color;
void main (void) {
    fragpos = a_position;
    //anormal = a_normal;
    anormal = mat3(transpose(inverse(u_model))) * a_normal;
    //float r = abs(sin(0.23424*float(neuronid)));
    //float g = abs(sin(0.63432*float(neuronid)));
    //float b = abs(sin(0.93423*float(neuronid)));
    v_color = texture2D(u_tex, vec2(neuronid, time/800.0)).rgb;
    //v_color = vec3(r, g, b) * (0.8 + 0.2*sin(time*g*3.0));
    gl_Position = u_view * u_model * vec4(a_position, 1.0);
    // gl_Position = vec4(a_position, 1.0);
}
"""

FRAG_SHADER = """ // simple fragment shader
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
    fragpos = a_position;
    anormal = a_normal;
    //anormal = mat3(transpose(inverse(u_model))) * a_normal;
    gl_Position = u_view * u_model * vec4(a_position, 1.0);
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
cmap = matplotlib.pyplot.get_cmap('viridis')
vsoma = np.random.random((len(network['neurons']), 1000))
vsoma_color = cmap((vsoma - vsoma.mean()) / vsoma.std())
vsoma_color = vsoma_color.astype(np.float32)
#vsoma_color = np.random.random((1000,1000, 3)).astype(np.float32)

###

def load_obj(filename):
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
    vertices = vertices[faces.flatten()] / 1000.0
    vertices = vertices.astype(np.float32)
    normals = get_normals(vertices, faces)
    normals = vert_normals[faces.flatten()].astype(np.float32)
    return vertices, normals

###

class Canvas(app.Canvas):

    def __init__(self):
        super().__init__(keys='interactive')

        # Create program
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        self.view = translate((0, 0, 0))
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
        self._program['u_view'] = self.view
        self._program['u_projection'] = self.projection
        self._program['u_tex'] = gloo.Texture2D(vsoma_color, wrapping='repeat', interpolation='linear') # repeat

        self.programs = []
        for filename in glob.glob('../mesh/*.obj'):
            if 'MAO_left' in filename:
                continue
            vertices, normals = load_obj(filename)
            color = np.array([0.6, 0.6, 0.6]) + np.random.random(3) * 0.4
            program = gloo.Program(MESH_VERT_SHADER, MESH_FRAG_SHADER)
            program['u_color'] = color
            program['a_position'] = gloo.VertexBuffer(vertices - vPosition.mean(0))
            program['a_normal'] = gloo.VertexBuffer(normals)
            program['u_model'] = self.model
            program['u_view'] = self.view
            program['u_projection'] = self.projection
            self.programs.append(program)

        gloo.set_clear_color('black')

        self.start = time.time()
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def on_timer(self, event):
        self.theta += .2
        self.phi += .2
        self.model = np.dot(rotate(self.theta, (0, 1, 0)),
                            rotate(self.phi,   (0, 0, 1)))
        self._program['u_model'] = self.model
        for program in self.programs:
            program['u_model'] = self.model
        self.update()

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(45.0, event.size[0] /
                                      float(event.size[1]), 2.0, 10.0)
        self._program['u_projection'] = self.projection
        for program in self.programs:
            program['u_projection'] = self.projection

    def on_draw(self, event):
        # gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=False)
        # gloo.set_depth_mask(False)
        elapsed = time.time() - self.start
        self._program['time'] = elapsed
        gloo.gl.glEnable(gloo.gl.GL_DEPTH_TEST)
        gloo.gl.glEnable(0x809d) # msaa
        gloo.gl.glEnable(0x864F) # gl depth clamp
        gloo.clear()
        self._program.draw('triangles')
        for program in self.programs:
            program.draw('triangles')


if __name__ == '__main__':
    canvas = Canvas()
    app.run()


