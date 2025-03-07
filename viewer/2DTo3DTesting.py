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

import hl2ss_rus

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
sphere_radius = 1
buffer_size = 5

key = 0

position = [0, 0, 0]
# Initial rotation in world space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]
# Initial scale in meters
scale = [0.2, 0.2, 0.2]
# Initial color
rgba = [1, 1, 1, 1]

prev_position = None
cubes = []
current_meshes = []

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
def convert2D3D(x_2D, y_2D, screen_width=760, screen_height=428, distance_to_screen=1.0, wert = 0.1):
    x_Mitte = screen_width / 2
    y_Mitte = screen_height / 2

    x_Abstand = x_2D - x_Mitte 
    y_Abstand = y_2D - y_Mitte

    x_Wert = x_Abstand * wert
    y_Wert = y_Abstand * wert

    return [ y_Wert, -x_Wert, distance_to_screen]


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
    render_option.mesh_show_wireframe = True  # Kein Drahtgitter
    render_option.mesh_show_back_face = True  # Rückseiten anzeigen

    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start Spatial Mapping data manager --------------------------------------
    # Set region of 3D space to sample
    volumes = hl2ss.sm_bounding_volume()
    volumes.add_sphere(sphere_center, sphere_radius)

    # Download observed surfaces
    sm_manager = hl2ss_sa.sm_manager(host, triangles_per_cubic_meter, mesh_threads)
    sm_manager.open()
    
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

    ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
    ipc.open()

    

    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.remove_all() # Remove all objects that were created remotely
    display_list.create_primitive(hl2ss_rus.PrimitiveType.Cube) # Create a cube, server will return its id
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the cube
    display_list.set_world_transform(key, position, rotation, scale) # Set the world transform of the cube
    display_list.set_color(key, rgba) # Set the color of the cube
    display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the cube visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    key = results[2] # Get the cube id, created by the 3rd command in the list

    # Main Loop ---------------------------------------------------------------
    with keyboard.Listener(on_press=on_press) as listener:
        while enable:
        
    
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

                meshSM = sm_manager.get_meshes()
    
                try:
                    coordinates = [tuple(row[:3]) for row in combined_point]
                    for coord in coordinates:
                        head_cube2 = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
                        head_cube2.paint_uniform_color([0, 1, 0])
                        # Verschiebe den Würfel an die gewünschte Position
                        head_cube2.translate(coord)
                        vis.add_geometry(head_cube2)
                        cubes.append(head_cube2)
    
                        
                        #coord Transform
                        coord = tuple(-x if i in {2} else x for i, x in enumerate(coord))
                        display_list = hl2ss_rus.command_buffer()
                        display_list.begin_display_list()
                        display_list.set_world_transform(key, coord, rotation, scale)
                        display_list.end_display_list()
                        ipc.push(display_list)
                        results = ipc.pull(display_list)
    
                except Exception as e:
                    print(e)
                # Cube-Bewegung mit Delta
                if prev_position is not None:
                    delta = new_position - prev_position
                    head_cube.translate(delta)
                    vis.update_geometry(head_cube)
                    
                    # Vektor aktualisieren, basierend auf der neuen Position des Würfels
    
                # Neue Meshes hinzufügen
                new_meshes = [hl2ss_sa.sm_mesh_to_open3d_triangle_mesh(mesh) for mesh in meshSM]
                for mesh in new_meshes:
                    mesh.vertex_colors = mesh.vertex_normals
                    
                    vis.add_geometry(mesh)
                current_meshes.extend(new_meshes)
    
                vis.poll_events()
                vis.update_renderer()
             
                pv_intrinsics = np.array([[-data_pv.payload.focal_length[0], 0, 0, 0], [0, data_pv.payload.focal_length[1], 0, 0], [data_pv.payload.principal_point[0], data_pv.payload.principal_point[1], 1, 0], [0, 0, 0, 1]], dtype=np.float32)
            
                pv_extrinsics = np.eye(4, 4, dtype=np.float32)
                R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=pv_extrinsics.dtype)
                pv_intrinsics[0, 0] = -pv_intrinsics[0, 0]
                pv_extrinsics = pv_extrinsics @ R
                pv_intrinsics, pv_extrinsics = (pv_intrinsics, pv_extrinsics)
                world_to_image = np.linalg.inv(data_pv.pose) @ pv_extrinsics @ pv_intrinsics       
                
                indexTest = indexTest + 1
                # Draw Combined Gaze Pointer ------------------------------------------
                if (True):
                
                    """
                    pv_intrinsics = hl2ss.create_pv_intrinsics(data_pv.payload.focal_length, data_pv.payload.principal_point)
                    pv_extrinsics = np.eye(4, 4, dtype=np.float32)
                    pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)
                    world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)
                    """
                    
                    #vector = convert2D3D(0, 0, wert= 0.0013)
                    #print(vector)
                    local_combined_ray = np.vstack((eet.combined_ray.origin, eet.combined_ray.direction)).reshape((-1, 6))  
                    #local_combined_ray = np.array([[0, 0, 0,  vector[0],  vector[1],  vector[2] ]], dtype=np.float32)
                    #print(type(local_combined_ray), local_combined_ray)                                                                                                                                                          
                    combined_ray = np.hstack((local_combined_ray[:, 0:3] @ data_eet.pose[:3, :3] + data_eet.pose[3, :3].reshape(([1] * (len(local_combined_ray[:, 0:3].shape) - 1)).append(3)), local_combined_ray[:, 3:6] @ data_eet.pose[:3, :3]))
                    #local_combined_ray = np.array([[0.05398782, -0.06293464, -0.04105301,  0.04428716,  0.12401385,  0.9912918 ]], dtype=np.float32)
                    
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
                        #print(type(combined_point), combined_point)
                        #combined_image_point = hl2ss_3dcv.project(combined_point, world_to_image)
                        transformTemp = combined_point @ world_to_image[:3, :3] + world_to_image[3, :3].reshape(([1] * (len(combined_point.shape) - 1)).append(3))
                        combined_image_point = transformTemp[..., 0:-1] / transformTemp[..., -1, np.newaxis]
                        #hl2ss_utilities.draw_points(image, combined_image_point.astype(np.int32), radius, combined_color, thickness)
                        for x, y in combined_image_point.astype(np.int32):
                            if (x >= 0 and y >= 0 and x < image.shape[1] and y < image.shape[0]):
                                cv2.circle(image, (x, y), radius, combined_color, thickness)
    
                """
                u, v = 320, 240  # Bildmitte
                origin, direction = image_to_world_ray(u, v, data_pv)
                temp = origin, direction
                #place_cube_at_intersection(origin, direction, 0.5, vis, cubes)
                #print(type(temp), temp)
                try:
                    head_cube3 = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
                    head_cube3.paint_uniform_color([0, 0, 1])
                    # Verschiebe den Würfel an die gewünschte Position
                    head_cube3.translate(origin)
                    vis.add_geometry(head_cube3)
                    cubes.append(head_cube3)
                except:
                    print("new_position shape:", new_position.shape)
                    print("ray_origin:", origin.shape, origin)
                    print("ray_direction:", direction.shape, direction)
                    #print("local_left_ray shape:", left_ray.shape, left_ray)
    
                    """
    
                      
                vis.clear_geometries()
                vis.add_geometry(head_cube)
                #vis.add_geometry(arrow)  # Pfeil erneut hinzufügen
                #vis.add_geometry(arrow2)
                #vis.add_geometry(arrow3)
                prev_position = new_position.copy()
    
                # Display frame -------------------------------------------------------            
                cv2.imshow('Video', image)
                cv2.waitKey(1)

    command_buffer = hl2ss_rus.command_buffer()
    command_buffer.remove(key) # Destroy cube
    ipc.push(command_buffer)
    results = ipc.pull(command_buffer)
    
    ipc.close()

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