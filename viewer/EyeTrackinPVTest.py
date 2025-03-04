#------------------------------------------------------------------------------
# This script receives video frames and extended eye tracking data from the 
# HoloLens. The received left, right, and combined gaze pointers are projected
# onto the video frame.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import multiprocessing as mp
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_sa

import open3d as o3d

# Settings --------------------------------------------------------------------

# HoloLens 2 address
host = "192.168.137.140"
# Camera parameters
# See etc/hl2_capture_formats.txt for a list of supported formats
pv_width     = 760
pv_height    = 428
pv_framerate = 30

# EET parameters
eet_fps = 30 # 30, 60, 90

# Marker properties
radius = 5
combined_color = (255, 0, 255)
left_color     = (  0, 0, 255)
right_color    = (255, 0,   0)
thickness = -1

# Buffer length in seconds
buffer_length = 7

# Spatial Mapping manager settings
triangles_per_cubic_meter = 30
mesh_threads = 12
sphere_center = [0, 0, 0]
sphere_radius = 0.5
buffer_size = 5

#------------------------------------------------------------------------------
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
    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Open3D Visualization Setup
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    #render_option.point_size = 3  # Bessere Sichtbarkeit
    #render_option.background_color = [150, 150, 150]  # Grauer Hintergrund
    render_option.mesh_show_wireframe = True  # Kein Drahtgitter
    render_option.mesh_show_back_face = True  # Rückseiten anzeigen

    
    current_meshes = []

    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start Spatial Mapping data manager --------------------------------------
    # Set region of 3D space to sample
    volumes = hl2ss.sm_bounding_volume()
    volumes.add_sphere(sphere_center, sphere_radius)

    # Download observed surfaces
    sm_manager = hl2ss_sa.sm_manager(host, triangles_per_cubic_meter, mesh_threads)
    sm_manager.open()
    
    print("surfaces")
    # Start PV and EET streams ------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=eet_fps))
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, 
                      hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
    producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, 
                       buffer_size * hl2ss.Parameters_SI.SAMPLE_RATE)
    producer.initialize(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss.Parameters_SI.SAMPLE_RATE * buffer_length)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.SPATIAL_INPUT)
    producer.start(hl2ss.StreamPort.EXTENDED_EYE_TRACKER)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, ...)
    sink_si = consumer.create_sink(producer, hl2ss.StreamPort.SPATIAL_INPUT, manager, ...)
    sink_eet = consumer.create_sink(producer, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, manager, ...)
    sink_pv.get_attach_response()
    sink_si.get_attach_response()
    sink_eet.get_attach_response()

    head_cube = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
    head_cube.paint_uniform_color([1, 0, 0])
    vis.add_geometry(head_cube)

    # Pfeil (Vektor) erstellen
    """
    arrow = create_arrow(np.array([0, 0, 0]), np.array([1, 0, 0]), length=0.5)
    arrow2 = create_arrow(np.array([0, 0, 0]), np.array([1, 0, 0]), length=0.5)
    arrow3 = create_arrow(np.array([0, 0, 0]), np.array([1, 0, 0]), length=0.5)
    vis.add_geometry(arrow)
    vis.add_geometry(arrow2)
    vis.add_geometry(arrow3)
    """

    prev_position = None
    cubes = []

    # Main Loop ---------------------------------------------------------------
    while (enable):
        
        # Download observed surfaces ------------------------------------------
        sm_manager.get_observed_surfaces()

        # Wait for PV frame ---------------------------------------------------
        sink_pv.acquire()

        # Get PV frame and nearest (in time) EET frame ------------------------
        _, data_pv = sink_pv.get_most_recent_frame()
        _, data_eet = sink_eet.get_nearest(data_pv.timestamp)
        _, data_si = sink_si.get_nearest(data_pv.timestamp)
        
        

        if data_pv and data_eet and data_si:
            #print(data_eet.pose, data_eet.pose.shape)

            """
            [[-0.0419414  -0.97636074  0.21203884  0.        ]
            [-0.983368    0.00280333 -0.1816026   0.        ]
            [ 0.17671525 -0.21612889 -0.9602394   0.        ]
            [-0.2300689   0.22131944 -0.5359313   1.        ]] (4, 4)
            """
            image = data_pv.payload.image
            eet = hl2ss.unpack_eet(data_eet.payload)
            si = hl2ss.unpack_si(data_si.payload)


            new_position = np.array(si.get_head_pose().position)

            # Mesh-Update mit Cleanup
            volume = hl2ss.sm_bounding_volume()
            volume.add_sphere(new_position, sphere_radius)
            sm_manager.set_volumes(volume)
            sm_manager.get_observed_surfaces()

            try:
                coordinates = [tuple(row[:3]) for row in combined_point]
                for coord in coordinates:
                    head_cube2 = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
                    head_cube2.paint_uniform_color([0, 1, 0])
                    # Verschiebe den Würfel an die gewünschte Position
                    head_cube2.translate(coord)
                    vis.add_geometry(head_cube2)
                    cubes.append(head_cube2)
            except Exception as e:
                print(e)
            # Cube-Bewegung mit Delta
            if prev_position is not None:
                delta = new_position - prev_position
                head_cube.translate(delta)
                vis.update_geometry(head_cube)
                
                # Vektor aktualisieren, basierend auf der neuen Position des Würfels

            # Neue Meshes hinzufügen
            new_meshes = [hl2ss_sa.sm_mesh_to_open3d_triangle_mesh(mesh) for mesh in sm_manager.get_meshes()]
            for mesh in new_meshes:
                mesh.vertex_colors = mesh.vertex_normals
                
                vis.add_geometry(mesh)
            current_meshes.extend(new_meshes)

            vis.poll_events()
            vis.update_renderer()
            # Update PV intrinsics ------------------------------------------------
            # PV intrinsics may change between frames due to autofocus
            
            # Compute world to PV image transformation matrix ---------------------
            
    
            # Draw Left Gaze Pointer ----------------------------------------------
            if (eet.left_ray_valid and False):
                local_left_ray = hl2ss_utilities.si_ray_to_vector(eet.left_ray.origin, eet.left_ray.direction)
                left_ray = hl2ss_utilities.si_ray_transform(local_left_ray, data_eet.pose)
                d = sm_manager.cast_rays(left_ray)
                if (np.isfinite(d)):
                    left_point = hl2ss_utilities.si_ray_to_point(left_ray, d)
                    left_image_point = hl2ss_3dcv.project(left_point, world_to_image)
                    hl2ss_utilities.draw_points(image, left_image_point.astype(np.int32), radius, left_color, thickness)
                    try:
                        #end_point = eet.left_ray.origin + left_ray * 0.5
                        ray_origin = left_ray[:, :3].flatten().tolist()  # (x, y, z) als Liste
                        ray_direction = left_ray[:, 3:6].flatten().tolist()  # (dx, dy, dz) als Liste

                        arrow.points = o3d.utility.Vector3dVector([ray_origin, ray_direction])
                        vis.update_geometry(arrow)
                    except:
                        print("new_position shape:", new_position.shape)
                        print("ray_origin:", ray_origin.shape, ray_origin)
                        print("ray_direction:", ray_direction.shape, ray_direction)
                        #print("local_left_ray shape:", left_ray.shape, left_ray)

                    
                    

            # Draw Right Gaze Pointer ---------------------------------------------
            if (eet.right_ray_valid and False):
                local_right_ray = hl2ss_utilities.si_ray_to_vector(eet.right_ray.origin, eet.right_ray.direction)
                right_ray = hl2ss_utilities.si_ray_transform(local_right_ray, data_eet.pose)
                d = sm_manager.cast_rays(right_ray)
                if (np.isfinite(d)):
                    right_point = hl2ss_utilities.si_ray_to_point(right_ray, d)
                    right_image_point = hl2ss_3dcv.project(right_point, world_to_image)
                    hl2ss_utilities.draw_points(image, right_image_point.astype(np.int32), radius, right_color, thickness)
                    try:
                        #end_point = eet.left_ray.origin + left_ray * 0.5
                        ray_origin = right_ray[:, :3].flatten().tolist()  # (x, y, z) als Liste
                        ray_direction = right_ray[:, 3:6].flatten().tolist()  # (dx, dy, dz) als Liste

                        arrow2.points = o3d.utility.Vector3dVector([ray_origin, ray_direction])
                        vis.update_geometry(arrow2)
                    except:
                        print("new_position shape:", new_position.shape)
                        print("ray_origin:", ray_origin.shape, ray_origin)
                        print("ray_direction:", ray_direction.shape, ray_direction)
                        #print("local_left_ray shape:", left_ray.shape, left_ray)

            pv_intrinsics = np.array([[-data_pv.payload.focal_length[0], 0, 0, 0], [0, data_pv.payload.focal_length[1], 0, 0], [data_pv.payload.principal_point[0], data_pv.payload.principal_point[1], 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        
            pv_extrinsics = np.eye(4, 4, dtype=np.float32)
            R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=pv_extrinsics.dtype)
            pv_intrinsics[0, 0] = -pv_intrinsics[0, 0]
            pv_extrinsics = pv_extrinsics @ R
            pv_intrinsics, pv_extrinsics = (pv_intrinsics, pv_extrinsics)
            world_to_image = np.linalg.inv(data_pv.pose) @ pv_extrinsics @ pv_intrinsics       

            # Draw Combined Gaze Pointer ------------------------------------------
            if (eet.combined_ray_valid):

                """
                pv_intrinsics = hl2ss.create_pv_intrinsics(data_pv.payload.focal_length, data_pv.payload.principal_point)
                pv_extrinsics = np.eye(4, 4, dtype=np.float32)
                pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)
                world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)
                """
                


                local_combined_ray = np.vstack((eet.combined_ray.origin, eet.combined_ray.direction)).reshape((-1, 6))                                                                                                                                                            
                combined_ray = np.hstack((local_combined_ray[:, 0:3] @ data_eet.pose[:3, :3] + data_eet.pose[3, :3].reshape(([1] * (len(local_combined_ray[:, 0:3].shape) - 1)).append(3)), local_combined_ray[:, 3:6] @ data_eet.pose[:3, :3]))
                #d = sm_manager.cast_rays(combined_ray)
                surfaces = sm_manager._get_surfaces()
                n = len(surfaces)
                distances = np.ones(combined_ray.shape[0:-1] + (n if (n > 0) else 1,)) * np.inf
                for index, entry in enumerate(surfaces):
                    distances[..., index] = entry.rcs.cast_rays(combined_ray)['t_hit'].numpy()
                distances = np.min(distances, axis=-1)
                d = distances
                if (np.isfinite(d)):
                    #combined_point = hl2ss_utilities.si_ray_to_point(combined_ray, d)
                    combined_point = (combined_ray[:, 0:3] + d * combined_ray[:, 3:6]).reshape((-1, 3))
                    #combined_image_point = hl2ss_3dcv.project(combined_point, world_to_image)
                    transformTemp = combined_point @ world_to_image[:3, :3] + world_to_image[3, :3].reshape(([1] * (len(combined_point.shape) - 1)).append(3))
                    combined_image_point = transformTemp[..., 0:-1] / transformTemp[..., -1, np.newaxis]
                    #hl2ss_utilities.draw_points(image, combined_image_point.astype(np.int32), radius, combined_color, thickness)
                    for x, y in combined_image_point.astype(np.int32):
                        if (x >= 0 and y >= 0 and x < image.shape[1] and y < image.shape[0]):
                            cv2.circle(image, (x, y), radius, combined_color, thickness)

            #print("Testing")
            #ray_origin, ray_direction = unproject_2d_to_3d((230,240), pv_intrinsics, pv_extrinsics, data_eet.pose)
            #place_cube_at_intersection(ray_origin, ray_direction, 0.5, vis, cubes)


                  
            vis.clear_geometries()
            vis.add_geometry(head_cube)
            #vis.add_geometry(arrow)  # Pfeil erneut hinzufügen
            #vis.add_geometry(arrow2)
            #vis.add_geometry(arrow3)
            prev_position = new_position.copy()

            # Display frame -------------------------------------------------------            
            #cv2.imshow('Video', image)
            #cv2.waitKey(1)

    vis.clear_geometries()
    sink_si.detach()
    producer.stop(hl2ss.StreamPort.SPATIAL_INPUT)
    listener.join()

    # Stop Spatial Mapping data manager ---------------------------------------
    sm_manager.close()

    # Stop PV and EET streams -------------------------------------------------
    sink_pv.detach()
    sink_eet.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.EXTENDED_EYE_TRACKER)

    # Stop PV subsystem -------------------------------------------------------
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.join()


if __name__ == '__main__':
    main()