import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import os
import numpy as np
from pynput import keyboard
import multiprocessing as mp
import open3d as o3d
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_sa
import hl2ss_rus
import Config

# Settings --------------------------------------------------------------------
host = Config.HOST  # HoloLens address
buffer_size = 1  # Buffer length in seconds
tpcm = 0.5  # Reduziert von 1 auf 0.5 Punkte pro Kubikmeter
threads = 2  # Number of processing threads
radius = 0.5  # Mapping radius
voxel_size = 0.05  # Voxel-Größe für Downsampling

# Ordner erstellen
output_folder = "meshes"
os.makedirs(output_folder, exist_ok=True)

def get_unique_filename(base_path, filename, ext):
    counter = 1
    unique_filename = f"{filename}_{counter}{ext}"
    while os.path.exists(os.path.join(base_path, unique_filename)):
        counter += 1
        unique_filename = f"{filename}_{counter}{ext}"
    return os.path.join(base_path, unique_filename)

# Fenstereinstellungen
WIDTH, HEIGHT = 800, 600
FOV = 45  # Sichtfeld
NEAR_PLANE = 0.1
FAR_PLANE = 100.0

# Kameraposition und -orientierung
camera_pos = [0, 3, 8]
camera_rot = [30, 0, 0]  # Pitch, Yaw, Roll in Grad

# Bewegungsgeschwindigkeit
MOVE_SPEED = 0.2
ROTATE_SPEED = 2.0

# Maussteuerung
mouse_sensitivity = 0.2
last_mouse_pos = None
mouse_dragging = False

# Würfelgröße
cube_size = 0.5
cube_size_min = 0.1
cube_size_max = 2.0
cube_size_step = 0.1

# Liste der Würfel in der Szene
cubes = []  # Format: [position, size, color]

def init():
    """OpenGL initialisieren"""
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    # Lichtposition und -eigenschaften
    glLight(GL_LIGHT0, GL_POSITION, (5, 10, 5, 1))
    glLight(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
    glLight(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))

def set_projection():
    """Perspektive einstellen"""
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(FOV, WIDTH/HEIGHT, NEAR_PLANE, FAR_PLANE)
    
def set_camera():
    """Kamera positionieren"""
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(camera_rot[0], 1, 0, 0)  # Pitch (X-Achse)
    glRotatef(camera_rot[1], 0, 1, 0)  # Yaw (Y-Achse)
    glRotatef(camera_rot[2], 0, 0, 1)  # Roll (Z-Achse)
    glTranslatef(-camera_pos[0], -camera_pos[1], -camera_pos[2])

def draw_plane():
    """Ebene zeichnen"""
    glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_QUADS)
    glNormal3f(0, 1, 0)  # Normale zeigt nach oben
    size = 20
    glVertex3f(-size, 0, -size)
    glVertex3f(-size, 0, size)
    glVertex3f(size, 0, size)
    glVertex3f(size, 0, -size)
    glEnd()
    
    # Gitternetz zeichnen für bessere Orientierung
    glColor3f(0.3, 0.3, 0.3)
    glBegin(GL_LINES)
    for i in range(-size, size + 1, 2):
        glVertex3f(i, 0.01, -size)
        glVertex3f(i, 0.01, size)
        glVertex3f(-size, 0.01, i)
        glVertex3f(size, 0.01, i)
    glEnd()

def draw_cube(position, size=0.5, color=(0.0, 0.6, 1.0)):
    """Einen Würfel an der angegebenen Position zeichnen"""
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor3fv(color)
    
    vertices = [
        [size, size, -size], [size, -size, -size], [-size, -size, -size], [-size, size, -size],
        [size, size, size], [size, -size, size], [-size, -size, size], [-size, size, size]
    ]
    
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 7, 3],
        [1, 5, 6, 2], [0, 4, 5, 1], [3, 7, 6, 2]
    ]
    
    normals = [
        [0, 0, -1], [0, 0, 1], [0, 1, 0],
        [0, -1, 0], [1, 0, 0], [-1, 0, 0]
    ]
    
    # Würfel zeichnen
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glNormal3fv(normals[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()
    
    glPopMatrix()

def create_ray_from_mouse(mouse_pos):
    """Strahl vom Kamerastandpunkt durch den Mauszeiger erstellen"""
    # Viewport abrufen
    viewport = glGetIntegerv(GL_VIEWPORT)
    
    # Projektion und Modelview-Matrizen abrufen
    projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
    modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    
    # Bildschirmkoordinaten in OpenGL-Koordinaten umwandeln
    mouse_x, mouse_y = mouse_pos
    win_y = viewport[3] - mouse_y - 1
    
    # Nahe Punkt berechnen
    near_point = gluUnProject(mouse_x, win_y, 0.0, 
                              modelview_matrix, projection_matrix, viewport)
    
    # Ferne Punkt berechnen
    far_point = gluUnProject(mouse_x, win_y, 1.0, 
                            modelview_matrix, projection_matrix, viewport)
    
    # Richtungsvektor berechnen
    ray_dir = np.array(far_point) - np.array(near_point)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    
    return np.array(near_point), ray_dir

def intersect_ray_plane(ray_origin, ray_dir, plane_pos, plane_normal):
    """Strahl-Ebenen-Schnitt berechnen"""
    # Normalisieren des Ebenennormals
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Überprüfen ob der Strahl parallel zur Ebene ist
    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-6:
        return None  # Kein Schnittpunkt
    
    # Distanz zur Ebene berechnen
    d = np.dot(plane_pos - ray_origin, plane_normal) / denom
    
    # Prüfen ob der Schnittpunkt vor der Kamera liegt
    if d < 0:
        return None
    
    # Schnittpunkt berechnen
    intersection = ray_origin + ray_dir * d
    
    return intersection

def generate_random_color():
    """Zufällige Farbe generieren"""
    return (
        np.random.uniform(0.2, 0.8),
        np.random.uniform(0.2, 0.8),
        np.random.uniform(0.2, 0.8)
    )

def draw_crosshair():
    """Crosshair zeichnen"""
    glLineWidth(2)
    glBegin(GL_LINES)
    
    # Horizontal (Mitte des Bildschirms)
    glColor3f(1.0, 1.0, 1.0)  # Weiß
    glVertex2f(WIDTH / 2 - 10, HEIGHT / 2)
    glVertex2f(WIDTH / 2 + 10, HEIGHT / 2)
    
    # Vertikal (Mitte des Bildschirms)
    glVertex2f(WIDTH / 2, HEIGHT / 2 - 10)
    glVertex2f(WIDTH / 2, HEIGHT / 2 + 10)
    
    glEnd()

# Drehmatrix für 2D um die Z-Achse (2D)
def rotation_matrix_2d(theta):
    """Drehmatrix um die Z-Achse für 2D"""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def get_2d_vector_from_angle(angle_deg):
    # Umrechnung des Winkels in Bogenmaß
    angle_rad = np.radians(angle_deg)
    
    # Berechnung des 2D-Vektors (Kosinus für X, Sinus für Y)
    x = np.cos(angle_rad)
    y = np.sin(angle_rad)
    
    return np.array([x, y])
def main():
    """Hauptprogramm"""
    global camera_pos, camera_rot, cube_size, last_mouse_pos, mouse_dragging
    
    pygame.init()
    display = (WIDTH, HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Würfel-Platzierung mit Ray Collision")
    
    init()
    set_projection()
    
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Linke Maustaste
                    # Strahl vom Kamerastandpunkt durch den Mauszeiger erstellen
                    ray_origin, ray_dir = create_ray_from_mouse(event.pos)
                    
                    # Ebene definieren (Punkt auf Ebene und Normal)
                    plane_pos = np.array([0, 0, 0])
                    plane_normal = np.array([0, 1, 0])  # Y-Achse nach oben
                    
                    # Schnittpunkt berechnen
                    intersection = intersect_ray_plane(ray_origin, ray_dir, plane_pos, plane_normal)
                    
                    if intersection is not None:
                        # Würfel an Schnittpunkt platzieren
                        cube_pos = [intersection[0], intersection[1] + cube_size, intersection[2]]  # Position anpassen, damit Würfel auf der Ebene steht
                        cubes.append([cube_pos, cube_size, generate_random_color()])
                
                elif event.button == 3:  # Rechte Maustaste
                    mouse_dragging = True
                    last_mouse_pos = event.pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:  # Rechte Maustaste
                    mouse_dragging = False
                
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging and last_mouse_pos is not None:
                    # Kamera mit Maus drehen
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    
                    camera_rot[1] -= dx * mouse_sensitivity
                    camera_rot[0] -= dy * mouse_sensitivity
                    
                    # Neigung begrenzen
                    camera_rot[0] = max(-90, min(90, camera_rot[0]))
                    
                    last_mouse_pos = event.pos
            
            elif event.type == pygame.KEYDOWN:
                # Skalierung des Würfels
                if event.key == K_q:
                    # Würfel verkleinern
                    cube_size = max(cube_size_min, cube_size - cube_size_step)
                elif event.key == K_e:
                    # Würfel vergrößern
                    cube_size = min(cube_size_max, cube_size + cube_size_step)
                elif event.key == K_ESCAPE:
                    running = False
        
        # Bewegungssteuerung mit Tasten
        keys = pygame.key.get_pressed()
        move_speed = MOVE_SPEED
        
        # Blickrichtung berechnen (Pitch & Yaw)
        yaw_rad = math.radians(camera_rot[1])
        pitch_rad = math.radians(camera_rot[0])
        
        # Blickrichtung (vorwärts)
        look_dir = np.array([
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad)
        ])
        look_dir = look_dir / np.linalg.norm(look_dir)
        
        # Rechts-Vektor berechnen (für A/D)
        right_dir = np.cross(look_dir, [0, 1, 0])
        right_dir = right_dir / np.linalg.norm(right_dir)
        
        # Hoch/Runter-Vektor
        up_dir = np.array([0, 1, 0])
        
        # Bewegung zusammensetzen
        move_dir = np.array([0.0, 0.0, 0.0])
        if keys[K_w]:
            move_dir += look_dir *[-1,0,1] 
        if keys[K_s]:
            move_dir -= look_dir *[-1,0,1]
        if keys[K_d]:
            temp = np.dot(rotation_matrix_2d(0), get_2d_vector_from_angle(camera_rot[1]))
            move_dir += [temp[0], 0, temp[1]]
        if keys[K_a]:
            temp = np.dot(rotation_matrix_2d(0), get_2d_vector_from_angle(camera_rot[1]))
            move_dir -= [temp[0], 0, temp[1]]
        if keys[K_SPACE]:
            move_dir += up_dir
        if keys[K_LSHIFT]:
            move_dir -= up_dir
        
        if np.linalg.norm(move_dir) > 0:
            move_dir = move_dir / np.linalg.norm(move_dir)
            camera_pos += move_dir * move_speed
        # Szene rendern
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        set_camera()
        
        draw_plane()
        for cube in cubes:
            draw_cube(cube[0], cube[1], cube[2])
        
        draw_crosshair()
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()


