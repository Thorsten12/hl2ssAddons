import time
import open3d as o3d
import numpy as np
import hl2ss
import hl2ss_sa

def ply_to_sm_mesh(ply_file):
    # PLY-Datei mit Open3D laden
    mesh = o3d.io.read_triangle_mesh(ply_file)

    # Vertex-Positionen (homogene Koordinaten)
    vertex_positions = np.asarray(mesh.vertices, dtype=np.float32)
    vertex_positions = np.hstack((vertex_positions, np.ones((len(vertex_positions), 1), dtype=np.float32)))

    # Dreiecksindizes
    triangle_indices = np.asarray(mesh.triangles, dtype=np.uint32)

    # Vertex-Normalen (falls vorhanden, sonst berechnen)
    if mesh.has_vertex_normals():
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    else:
        mesh.compute_vertex_normals()
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    vertex_normals = np.hstack((vertex_normals, np.ones((len(vertex_normals), 1), dtype=np.float32)))

    # Position Scale: Maximale Distanz zur Mitte
    vertex_position_scale = np.array([np.max(np.abs(vertex_positions[:, :3]))], dtype=np.float32)

    # Pose: Identitätsmatrix (keine Transformation)
    pose = np.eye(4, dtype=np.float32)

    # Bounds: Min und Max der Positionen
    bounds = np.array([vertex_positions[:, :3].min(axis=0), vertex_positions[:, :3].max(axis=0)], dtype=np.float32)

    # Konvertierung der Daten in das _sm_mesh-Format (ohne Binärkonvertierung)
    sm_mesh = hl2ss._sm_mesh(
        vertex_position_scale,
        pose,
        bounds,
        vertex_positions,
        triangle_indices,
        vertex_normals
    )

    return sm_mesh

def sm_mesh_to_sm_manager(path):
    sm_mesh = ply_to_sm_mesh(path)
    update_time = time.time()

    # Load the mesh for ray casting
    mesh = o3d.io.read_triangle_mesh(path)
    
    # Create a proper ray casting scene
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    
    # Use the scene object directly, not bytes
    rcs = scene
    
    # Create the entry object
    test_entry = hl2ss_sa._sm_manager_entry(update_time, sm_mesh, rcs)
    dummy_dict = {
        1: test_entry,
    }
    
    return dummy_dict
"""
Beispielanwendung:
    host = "192.168.137.140"
    triangles_per_cubic_meter = 10
    mesh_threads = 6
    sphere_center = [0, 0, 0]
    sphere_radius = 0.5
    buffer_size = 7
    sm_manager = hl2ss_sa.sm_manager(host, triangles_per_cubic_meter, mesh_threads)
    sm_manager.open()

    
dummy_dict = sm_mesh_to_sm_manager("C:/Users/admin/Desktop/hl2ss/viewer/meshes/spatial_mapping_mesh.ply")
sm_manager.set_surfaces(dummy_dict)
meshes = sm_manager.get_meshes()

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().mesh_show_back_face = True

# Jetzt kann sm_mesh_to_open3d_triangle_mesh() korrekt arbeiten
meshes = [hl2ss_sa.sm_mesh_to_open3d_triangle_mesh(mesh) for mesh in meshes]
first_geometry = True
for mesh in meshes:
    mesh.vertex_colors = mesh.vertex_normals
    vis.add_geometry(mesh, first_geometry)

if len(meshes) > 0:
    first_geometry = False
while True:
    if(keyboard.is_pressed("esc")):
        break
    vis.poll_events()
    vis.update_renderer()"
    ""
    """