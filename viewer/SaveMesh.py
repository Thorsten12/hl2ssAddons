import os
import numpy as np
from pynput import keyboard
import multiprocessing as mp
import open3d as o3d
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_sa

# Settings --------------------------------------------------------------------
host = '192.168.137.140'  # HoloLens address
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

if __name__ == '__main__':
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.space
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().mesh_show_back_face = True

    first_geometry = True

    sm_manager = hl2ss_sa.sm_manager(host, tpcm, threads)
    sm_manager.open()

    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, buffer_size * hl2ss.Parameters_SI.SAMPLE_RATE)
    producer.start(hl2ss.StreamPort.SPATIAL_INPUT)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_si = consumer.create_sink(producer, hl2ss.StreamPort.SPATIAL_INPUT, manager, ...)
    sink_si.get_attach_response()

    combined_mesh = o3d.geometry.TriangleMesh()
    processed_meshes = set()

    while enable:
        sink_si.acquire()
        _, data_si = sink_si.get_most_recent_frame()
        if data_si is None:
            continue

        si = hl2ss.unpack_si(data_si.payload)
        origin = si.get_head_pose().position

        volume = hl2ss.sm_bounding_volume()
        volume.add_sphere(origin, radius)
        sm_manager.set_volumes(volume)

        sm_manager.get_observed_surfaces()
        meshes = sm_manager.get_meshes()
        meshes = [hl2ss_sa.sm_mesh_to_open3d_triangle_mesh(mesh) for mesh in meshes]

        for mesh in meshes:
            mesh.vertex_colors = mesh.vertex_normals
            vis.add_geometry(mesh, first_geometry)

        if len(meshes) > 0:
            first_geometry = False
            
        for i, mesh in enumerate(meshes):
            mesh_center = np.mean(np.asarray(mesh.vertices), axis=0)
            mesh_id = (tuple(mesh_center), len(mesh.vertices))
            
            if mesh_id not in processed_meshes:
                processed_meshes.add(mesh_id)
                combined_mesh += mesh

        vis.poll_events()
        vis.update_renderer()

    print("Führe Voxel-Downsampling durch...")
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(combined_mesh.vertices))
    point_cloud.normals = o3d.utility.Vector3dVector(np.asarray(combined_mesh.vertex_normals))
    downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)

    print("Führe Poisson-Rekonstruktion durch...")
    poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        downsampled_cloud, depth=8, width=0, scale=1.1, linear_fit=False)

    print("Führe Quadric-Dezimierung durch...")
    target_reduction = 0.2  
    simplified_mesh = poisson_mesh.simplify_quadric_decimation(
        int(len(poisson_mesh.triangles) * target_reduction))
    simplified_mesh.compute_vertex_normals()

    file_path = get_unique_filename(output_folder, "spatial_mapping_mesh", ".ply")
    o3d.io.write_triangle_mesh(file_path, combined_mesh)
    print(f"Mesh gespeichert unter: {file_path}")
    print(f"Größe des ursprünglichen Meshes: {len(combined_mesh.vertices)} Vertices, {len(combined_mesh.triangles)} Dreiecke")

    file_path_simplified = get_unique_filename(output_folder, "spatial_mapping_mesh_simplified", ".ply")
    o3d.io.write_triangle_mesh(file_path_simplified, simplified_mesh)
    print(f"Vereinfachtes Mesh gespeichert unter: {file_path_simplified}")
    print(f"Größe des vereinfachten Meshes: {len(simplified_mesh.vertices)} Vertices, {len(simplified_mesh.triangles)} Dreiecke")

    sink_si.detach()
    producer.stop(hl2ss.StreamPort.SPATIAL_INPUT)
    listener.join()
