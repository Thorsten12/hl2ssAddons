import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import trimesh


VERTEX_SHADER = """
#version 120
varying vec3 normal;
varying vec3 frag_pos;

void main() {
    frag_pos = vec3(gl_ModelViewMatrix * gl_Vertex);
    normal = gl_NormalMatrix * gl_Normal;
    gl_Position = ftransform();
}
"""

FRAGMENT_SHADER = """
#version 120
varying vec3 normal;
varying vec3 frag_pos;

void main() {
    // Lichtquelle, die statisch von oben kommt
    vec3 light_pos = vec3(5.0, 10.0, 5.0);  // Position des Lichts
    vec3 light_color = vec3(1.0, 1.0, 1.0); // Weißes Licht

    // Normale des Fragments (Oberflächenrichtung)
    vec3 norm = normalize(normal);

    // Richtung des Lichts relativ zum Fragment
    vec3 light_dir = normalize(light_pos - frag_pos);

    // Berechnung des Diffuse Anteils (abhängig von der Oberfläche)
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * light_color;

    // Berechnung der spekularen Highlights (Reflexion des Lichts)
    vec3 view_dir = normalize(-frag_pos);  // Blickrichtung von der Kamera
    vec3 reflect_dir = reflect(-light_dir, norm);  // Reflektierte Lichtquelle
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0); // Spiegelungseffekt
    vec3 specular = 0.3 * spec * light_color;  // Intensität der Reflexion

    // Berechnung der Ambient (Umgebungslicht) Komponente
    vec3 ambient = 0.2 * light_color;

    // Endfarbe berechnen durch die Summierung der verschiedenen Lichtkomponenten
    vec3 color = ambient + diffuse + specular;

    // Optional: Tiefe basierte Helligkeit (für realistische Schattierung je nach Abstand)
    float distance = length(frag_pos);
    float brightness = clamp(1.5 - (distance / 55.0), 0.0, 1.0);  // Lichtverlauf je nach Entfernung
    color *= brightness;

    // Das Endergebnis: Die endgültige Fragmentfarbe (RGB-Wert)
    gl_FragColor = vec4(color, 1.0);
}

"""
# Fenster-Einstellungen
WIDTH, HEIGHT = 800, 600
FOV = 45
NEAR_PLANE = 0.1
FAR_PLANE = 100.0

# Kamera
camera_pos = [0, 3, 8]
camera_rot = [30, 0, 0]
MOVE_SPEED = 0.2
ROTATE_SPEED = 2.0
mouse_sensitivity = 0.2
last_mouse_pos = None
mouse_dragging = False

# Würfel
cube_size = 0.5
cube_size_min = 0.1
cube_size_max = 2.0
cube_size_step = 0.1
cubes = []
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    # Check for errors
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def create_shader_program():
    vertex = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragment = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    # Check for linking errors
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        raise RuntimeError(glGetProgramInfoLog(program))
    return program
def create_display_list_from_ply(path):
    mesh = trimesh.load(path)
    vertices = mesh.vertices
    faces = mesh.faces

    display_list = glGenLists(1)
    glNewList(display_list, GL_COMPILE)
    glBegin(GL_TRIANGLES)
    for face in faces:
        for idx in face:
            glVertex3fv(vertices[idx])
    glEnd()
    glEndList()
    return display_list

def draw_ply_model_display_list(shader_program, display_list_id, position=(0,0,0), scale=10.0, color=(1.0, 0.8, 0.2)):
    glPushMatrix()
    glTranslatef(*position)
    glScalef(scale, scale, scale)  # <-- hier wird skaliert
    glUseProgram(shader_program)
    glColor3fv(color)
    glCallList(display_list_id)
    glPopMatrix()

def init():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLight(GL_LIGHT0, GL_POSITION, (5, 10, 5, 1))
    glLight(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
    glLight(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))

def set_projection():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(FOV, WIDTH/HEIGHT, NEAR_PLANE, FAR_PLANE)

def set_camera():
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(camera_rot[0], 1, 0, 0)
    glRotatef(camera_rot[1], 0, 1, 0)
    glRotatef(camera_rot[2], 0, 0, 1)
    glTranslatef(-camera_pos[0], -camera_pos[1], -camera_pos[2])

def draw_plane():
    glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_QUADS)
    glNormal3f(0, 1, 0)
    size = 20
    glVertex3f(-size, 0, -size)
    glVertex3f(-size, 0, size)
    glVertex3f(size, 0, size)
    glVertex3f(size, 0, -size)
    glEnd()

    glColor3f(0.3, 0.3, 0.3)
    glBegin(GL_LINES)
    for i in range(-size, size + 1, 2):
        glVertex3f(i, 0.01, -size)
        glVertex3f(i, 0.01, size)
        glVertex3f(-size, 0.01, i)
        glVertex3f(size, 0.01, i)
    glEnd()

def draw_cube(position, size=0.5, color=(0.0, 0.6, 1.0)):
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor3fv(color)
    s = size
    vertices = [
        [s, s, -s], [s, -s, -s], [-s, -s, -s], [-s, s, -s],
        [s, s, s], [s, -s, s], [-s, -s, s], [-s, s, s]
    ]
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 7, 3],
        [1, 5, 6, 2], [0, 4, 5, 1], [3, 7, 6, 2]
    ]
    normals = [
        [0, 0, -1], [0, 0, 1], [0, 1, 0],
        [0, -1, 0], [1, 0, 0], [-1, 0, 0]
    ]
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glNormal3fv(normals[i])
        for v in face:
            glVertex3fv(vertices[v])
    glEnd()
    glPopMatrix()

def create_ray_from_mouse(mouse_pos):
    viewport = glGetIntegerv(GL_VIEWPORT)
    projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
    modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    mouse_x, mouse_y = mouse_pos
    win_y = viewport[3] - mouse_y - 1
    near_point = gluUnProject(mouse_x, win_y, 0.0, modelview_matrix, projection_matrix, viewport)
    far_point = gluUnProject(mouse_x, win_y, 1.0, modelview_matrix, projection_matrix, viewport)
    ray_dir = np.array(far_point) - np.array(near_point)
    ray_dir /= np.linalg.norm(ray_dir)
    return np.array(near_point), ray_dir

def intersect_ray_plane(ray_origin, ray_dir, plane_pos, plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-6:
        return None
    d = np.dot(plane_pos - ray_origin, plane_normal) / denom
    if d < 0:
        return None
    return ray_origin + ray_dir * d

def generate_random_color():
    return (
        np.random.uniform(0.2, 0.8),
        np.random.uniform(0.2, 0.8),
        np.random.uniform(0.2, 0.8)
    )

def draw_crosshair():
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, 0, HEIGHT, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glLineWidth(2)
    glColor3f(1, 1, 1)
    glBegin(GL_LINES)
    glVertex2f(WIDTH/2 - 10, HEIGHT/2)
    glVertex2f(WIDTH/2 + 10, HEIGHT/2)
    glVertex2f(WIDTH/2, HEIGHT/2 - 10)
    glVertex2f(WIDTH/2, HEIGHT/2 + 10)
    glEnd()
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

def rotation_matrix_2d(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def get_2d_vector_from_angle(angle_deg):
    rad = np.radians(angle_deg)
    return np.array([np.cos(rad), np.sin(rad)])

def main():
    global camera_pos, camera_rot, cube_size, last_mouse_pos, mouse_dragging
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Optimiertes 3D-Modell mit Display List")
    init()
    set_projection()
    shader_program = create_shader_program()

    # Optimierter PLY-Loader
    model_display_list = create_display_list_from_ply("C:/Users/admin/Desktop/hl2ss/viewer/spatial_mapping_mesh_simplified_7.ply")

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT: running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    ray_origin, ray_dir = create_ray_from_mouse(event.pos)
                    intersection = intersect_ray_plane(ray_origin, ray_dir, np.array([0,0,0]), np.array([0,1,0]))
                    if intersection is not None:
                        cube_pos = [intersection[0], intersection[1] + cube_size, intersection[2]]
                        cubes.append([cube_pos, cube_size, generate_random_color()])
                elif event.button == 3:
                    mouse_dragging = True
                    last_mouse_pos = event.pos
            elif event.type == MOUSEBUTTONUP:
                if event.button == 3:
                    mouse_dragging = False
            elif event.type == MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                camera_rot[1] -= dx * mouse_sensitivity
                camera_rot[0] -= dy * mouse_sensitivity
                camera_rot[0] = max(-90, min(90, camera_rot[0]))
                last_mouse_pos = event.pos
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    cube_size = max(cube_size_min, cube_size - cube_size_step)
                elif event.key == K_e:
                    cube_size = min(cube_size_max, cube_size + cube_size_step)
                elif event.key == K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        yaw_rad = math.radians(camera_rot[1])
        pitch_rad = math.radians(camera_rot[0])
        look_dir = np.array([
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad)
        ])
        look_dir /= np.linalg.norm(look_dir)
        right_dir = np.cross(look_dir, [0,1,0])
        right_dir /= np.linalg.norm(right_dir)
        up_dir = np.array([0,1,0])
        move_dir = np.array([0.0, 0.0, 0.0])

        if keys[K_w]: move_dir += look_dir * [-1, 0, 1]
        if keys[K_s]: move_dir -= look_dir * [-1, 0, 1]
        if keys[K_d]:
            temp = np.dot(rotation_matrix_2d(0), get_2d_vector_from_angle(camera_rot[1]))
            move_dir += [temp[0], 0, temp[1]]
        if keys[K_a]:
            temp = np.dot(rotation_matrix_2d(0), get_2d_vector_from_angle(camera_rot[1]))
            move_dir -= [temp[0], 0, temp[1]]
        if keys[K_SPACE]: move_dir += up_dir
        if keys[K_LSHIFT]: move_dir -= up_dir
        if np.linalg.norm(move_dir) > 0:
            move_dir /= np.linalg.norm(move_dir)
            camera_pos += move_dir * MOVE_SPEED

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        set_camera()
        #draw_plane()
        for cube in cubes:
            draw_cube(cube[0], cube[1], cube[2])
        draw_ply_model_display_list(shader_program, model_display_list, position=(0, 0.01, 0))
        draw_crosshair()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
