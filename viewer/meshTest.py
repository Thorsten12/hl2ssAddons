import open3d as o3d
import numpy as np

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
    # Der Rest des Codes bleibt unverändert

    # Open3D Visualization Setup
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = 3  # Bessere Sichtbarkeit
    render_option.background_color = [150, 150, 150]  # grauer Hintergrund
    render_option.mesh_show_wireframe = False  # Kein Drahtgitter
    render_option.mesh_show_back_face = True  # Rückseiten anzeigen

    current_meshes = []

    # Spatial Mapping manager setup
    sm_manager = hl2ss_sa.sm_manager(host, tpcm, threads)
    sm_manager.open()

    # Producer setup (Spatial Input stream)
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, 
                      hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, 
                       buffer_size * hl2ss.Parameters_SI.SAMPLE_RATE)
    producer.start(hl2ss.StreamPort.SPATIAL_INPUT)

    # Consumer and sink setup
    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_si = consumer.create_sink(producer, hl2ss.StreamPort.SPATIAL_INPUT, manager, ...)

    if sink_si is None:
        raise RuntimeError("Failed to create sink for Spatial Input stream.")
    sink_si.get_attach_response()

    head_cube = None
    prev_position = None

    head_cube = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
    head_cube.paint_uniform_color([1, 0, 0])
    vis.add_geometry(head_cube)

    # Pfeil (Vektor) erstellen
    arrow = create_arrow(np.array([0, 0, 0]), np.array([1, 0, 0]), length=0.5)
    vis.add_geometry(arrow)

    # Main processing loop
    while enable:
        sink_si.acquire()
        _, data_si = sink_si.get_most_recent_frame()
        
        if data_si:
            si = hl2ss.unpack_si(data_si.payload)
            new_position = np.array(si.get_head_pose().position)
            
            # Cube-Bewegung mit Delta
            if prev_position is not None:
                delta = new_position - prev_position
                head_cube.translate(delta)
                vis.update_geometry(head_cube)
                
                # Vektor aktualisieren, basierend auf der neuen Position des Würfels
                arrow_points = np.asarray(arrow.points)
                arrow_points[0] = new_position  # Startpunkt des Vektors an die neue Position des Würfels anpassen
                arrow.points = o3d.utility.Vector3dVector(arrow_points)
                vis.update_geometry(arrow)

            prev_position = new_position.copy()
            
            # Mesh-Update mit Cleanup
            volume = hl2ss.sm_bounding_volume()
            volume.add_sphere(new_position, radius)
            sm_manager.set_volumes(volume)
            
            vis.clear_geometries()  # 🧹 Alte Meshes entfernen
            vis.add_geometry(head_cube)

            # Update spatial mapping volume
            volume = hl2ss.sm_bounding_volume()
            volume.add_sphere(new_position, radius)
            sm_manager.set_volumes(volume)
            sm_manager.get_observed_surfaces()
            # Add new meshes
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
