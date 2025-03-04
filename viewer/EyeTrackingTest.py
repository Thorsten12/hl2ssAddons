import open3d as o3d
import numpy as np
from pynput import keyboard
import multiprocessing as mp
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_sa
import hl2ss_utilities

# Settings
host = '192.168.137.140'  # HoloLens IP address
buffer_size = 2          # Buffer length in seconds
tpcm, threads, radius = 500, 12, 5  # Spatial Mapping parameters

# EET parameters
eet_fps = 30 # 30, 60, 90

# Buffer length in seconds
buffer_length = 5

def create_arrow(start_point, direction, length=0.2):
    """Erstellt einen Pfeil basierend auf Startpunkt, Richtung und Länge"""
    arrow = o3d.geometry.LineSet()
    # Definiere die Start- und Endpunkte des Vektors
    end_point = start_point + np.array(direction) * length
    arrow.points = o3d.utility.Vector3dVector([start_point, end_point])
    arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
    
    # Setze die Farbe des Pfeils
    arrow.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Rote Farbe
    return arrow

def main():
    global enable
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.space
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Open3D Visualization Setup
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = 3  # Bessere Sichtbarkeit
    render_option.background_color = [150, 150, 150]  # Grauer Hintergrund
    render_option.mesh_show_wireframe = False  # Kein Drahtgitter
    render_option.mesh_show_back_face = False  # Rückseiten anzeigen
    

    current_meshes = []

    # Spatial Mapping manager setup
    sm_manager = hl2ss_sa.sm_manager(host, tpcm, threads)
    sm_manager.open()

    # Producer setup (Spatial Input stream + EET)
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, 
                      hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.configure(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, 
                       hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=eet_fps))
    producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, 
                       buffer_size * hl2ss.Parameters_SI.SAMPLE_RATE)
    producer.initialize(hl2ss.StreamPort.EXTENDED_EYE_TRACKER,
                        hl2ss.Parameters_SI.SAMPLE_RATE * buffer_length)
    producer.start(hl2ss.StreamPort.SPATIAL_INPUT)
    producer.start(hl2ss.StreamPort.EXTENDED_EYE_TRACKER)

    # Consumer and sink setup
    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_si = consumer.create_sink(producer, hl2ss.StreamPort.SPATIAL_INPUT, manager, ...)
    sink_eet = consumer.create_sink(producer, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, manager, ...)

    if sink_si is None and sink_eet is None:
        raise RuntimeError("Failed to create sink for Spatial Input stream or EET.")
    sink_si.get_attach_response()
    sink_eet.get_attach_response()

    head_cube = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
    head_cube.paint_uniform_color([1, 0, 0])
    vis.add_geometry(head_cube)

    # Pfeil (Vektor) erstellen
    arrow = create_arrow(np.array([0, 0, 0]), np.array([1, 0, 0]), length=0.5)
    vis.add_geometry(arrow)

    prev_position = None

    # Main processing loop
    while enable:
        sink_si.acquire()
        _, data_si = sink_si.get_most_recent_frame()
        _, data_eet = sink_eet.get_nearest(data_si.timestamp)
        
        if data_si and data_eet:
            si = hl2ss.unpack_si(data_si.payload)
            eet = hl2ss.unpack_eet(data_eet.payload)
            new_position = np.array(si.get_head_pose().position)
            
            # Cube-Bewegung mit Delta
            if prev_position is not None:
                #vis.clear_geometries()
                delta = new_position - prev_position
                head_cube.translate(delta)
                vis.update_geometry(head_cube)
                
                # Vektor aktualisieren, basierend auf der neuen Position des Würfels
                if (eet.left_ray_valid):
                    local_left_ray = hl2ss_utilities.si_ray_to_vector(eet.left_ray.origin, eet.left_ray.direction)
                    local_direction = local_left_ray[0, 3:6]

                    # Winkel in Radiant
                    theta = np.deg2rad(-90)  # -90 Grad, um den Vektor zurückzudrehen

                    # Rotationsmatrix für eine Rotation um die Z-Achse:
                    rotation_matrix = np.array([
                        [np.cos(theta), np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0,              0,             1]
                    ])

                    # Richtungsvektor anpassen:
                    corrected_direction = rotation_matrix.dot(local_direction)

                    #left_ray = hl2ss_utilities.si_ray_transform(local_left_ray, data_eet.pose)
                    try:
                        end_point = eet.left_ray.origin + corrected_direction * 0.5
                        arrow.points = o3d.utility.Vector3dVector([eet.left_ray.origin, end_point])
                        vis.update_geometry(arrow)
                    except:
                        print("new_position shape:", new_position.shape)
                        print("local_left_ray shape:", corrected_direction.shape, corrected_direction)

                """
                direction = delta / np.linalg.norm(delta)  # Normalisierte Richtung
                end_point = new_position + direction * 0.5  # Länge des Pfeils
                arrow.points = o3d.utility.Vector3dVector([new_position, end_point])
                vis.update_geometry(arrow)
                """

            prev_position = new_position.copy()
            
            # Mesh-Update mit Cleanup
            volume = hl2ss.sm_bounding_volume()
            volume.add_sphere(new_position, radius)
            sm_manager.set_volumes(volume)
            sm_manager.get_observed_surfaces()
            
            # Alte Meshes entfernen und neue hinzufügen
            vis.clear_geometries()
            vis.add_geometry(head_cube)
            vis.add_geometry(arrow)  # Pfeil erneut hinzufügen

            # Neue Meshes hinzufügen
            new_meshes = [hl2ss_sa.sm_mesh_to_open3d_triangle_mesh(mesh) for mesh in sm_manager.get_meshes()]
            for mesh in new_meshes:
                mesh.vertex_colors = mesh.vertex_normals
                vis.add_geometry(mesh)
            current_meshes.extend(new_meshes)

            vis.poll_events()
            vis.update_renderer()
    
    # Cleanup
    vis.clear_geometries()
    sink_si.detach()
    producer.stop(hl2ss.StreamPort.SPATIAL_INPUT)
    listener.join()

if __name__ == '__main__':
    main()