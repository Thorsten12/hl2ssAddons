#------------------------------------------------------------------------------
# This is our Current Script to send over distance, eyetracking, spatial Input and Video Data
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
import csv
import os
import datetime
import threading


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
buffer_length = 5

# Spatial Mapping manager settings
triangles_per_cubic_meter = 1000
mesh_threads = 2
sphere_center = [0, 0, 0]
sphere_radius = 5

#------------------------------------------------------------------------------

if __name__ == '__main__':
    # Keyboard events ---------------------------------------------------------
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start Spatial Mapping data manager --------------------------------------
    # Set region of 3D space to sample
    volumes = hl2ss.sm_bounding_volume()
    volumes.add_sphere(sphere_center, sphere_radius)

    print("volumes")
    # Download observed surfaces
    sm_manager = hl2ss_sa.sm_manager(host, triangles_per_cubic_meter, mesh_threads)
    print("2")
    sm_manager.open()
    print("3")
    sm_manager.set_volumes(volumes)
    print("4")  
    sm_manager.get_observed_surfaces()
    print("5")
    
    print("surfaces")
    # Start PV and EET streams ------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.configure(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=eet_fps))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss.Parameters_RM_DEPTH_AHAT.FPS * 5)
    producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss.Parameters_SI.SAMPLE_RATE * buffer_length)
    producer.initialize(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss.Parameters_SI.SAMPLE_RATE * buffer_length)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)
    producer.start(hl2ss.StreamPort.EXTENDED_EYE_TRACKER)
    producer.start(hl2ss.StreamPort.SPATIAL_INPUT)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, ...)
    sink_ahat = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_AHAT, manager, None)
    sink_eet = consumer.create_sink(producer, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, manager, None)
    sink_si = consumer.create_sink(producer, hl2ss.StreamPort.SPATIAL_INPUT, manager, None)
    sink_pv.get_attach_response()
    sink_ahat.get_attach_response()
    sink_eet.get_attach_response()
    sink_si.get_attach_response()

    def get_unique_folder(base_name):
        index = 0
        while True:
            folder_name = f"{base_name}{index}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                os.makedirs(os.path.join(folder_name, 'images'))
                os.makedirs(os.path.join(folder_name, 'depth'))
                os.makedirs(os.path.join(folder_name, 'AB'))

                return folder_name
            index += 1

    def format_list(lst):
        """ Hilfsfunktion, um Listen/Arrays als CSV-kompatible Strings zu formatieren """
        if isinstance(lst, (list, np.ndarray)):  # Falls Liste oder NumPy-Array
            return " ".join(map(str, lst))  # Konvertiere jedes Element zu String und verbinde mit Leerzeichen
        return lst  # Falls kein Array, einfach zurückgeben

    data_folder = get_unique_folder("data")
    csv_filename = os.path.join(data_folder, 'data.csv')
    image_folder = os.path.join(data_folder, 'images')
    depth_folder = os.path.join(data_folder, 'depth')
    ab_folder = os.path.join(data_folder, 'AB')


    print(f"Writing data to: {csv_filename}")
    print(f"Saving images to: {image_folder}")
    print(f"Saving Depth to: {depth_folder}")
    print(f"Saving AB to: {ab_folder}")

    def save_image(image_data, filename):
        cv2.imwrite(f"{filename}.jpg", image_data)

    def save_depth(arr, filename):
        np.save(filename, arr, allow_pickle=True, fix_imports=True)

    left_image_point = []
    right_image_point = []
    combined_image_point =[]

    null_arr = [-1,-1]

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=";")
        csv_writer.writerow([
            'Timestamp',
            'LeftImagePoint',
            'RightImagePoint',
            'CombinedImagePoint',
            'Focal_lenght',
            'Principal_point',
            'Intrinsics',
            'Extrinsics'
            'LeftGazeOrigin',
            'LeftGazeDirection',
            'RightGazeOrigin',
            'RightGazeDirection',
            'CombinedGazeOrigin',
            'CombinedGazeDirection',
            'Position',
            'Forward',
            'Up',
            'image_filename',
            'depth_filename',
            'ab_filename'
        ])

        # Main Loop ---------------------------------------------------------------
        while (enable):
            # Download observed surfaces ------------------------------------------
            sm_manager.get_observed_surfaces()
            # Wait for PV frame ---------------------------------------------------
            sink_pv.acquire()
            # Get PV frame and nearest (in time) EET frame ------------------------

            _, data_ahat = sink_ahat.get_most_recent_frame()
            #if ((data_ahat is None) or (not hl2ss.is_valid_pose(data_ahat.pose))):
                    #continue

            _, data_pv = sink_pv.get_nearest(data_ahat.timestamp)
            #if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
                #continue

            _, data_eet = sink_eet.get_nearest(data_ahat.timestamp)
            #if ((data_eet is None) or (not hl2ss.is_valid_pose(data_eet.pose))):
                #continue

            _, data_si = sink_si.get_nearest(data_ahat.timestamp)
            #if (data_si is None):
                #continue
            image = data_pv.payload.image
            eet = hl2ss.unpack_eet(data_eet.payload)
            si = hl2ss.unpack_si(data_si.payload)
            depth = data_ahat.payload.depth

            image_filename = os.path.join(image_folder, f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npy")
            depth_filename = os.path.join(depth_folder, f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npy")
            ab_filename = os.path.join(ab_folder, f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")

            threading.Thread(target=save_image, args=(image, image_filename)).start()
            threading.Thread(target=save_depth, args=(depth, depth_filename)).start()
            threading.Thread(target=save_depth, args=(data_ahat.payload.ab, ab_filename)).start()
            # Update PV intrinsics ------------------------------------------------
            # PV intrinsics may change between frames due to autofocus
            pv_intrinsics = hl2ss.create_pv_intrinsics(data_pv.payload.focal_length, data_pv.payload.principal_point)
            pv_extrinsics = np.eye(4, 4, dtype=np.float32)
            pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

            

            # Compute world to PV image transformation matrix ---------------------
            try: 
                world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)

                # Draw Left Gaze Pointer ----------------------------------------------
                if (eet.left_ray_valid):
                    local_left_ray = hl2ss_utilities.si_ray_to_vector(eet.left_ray.origin, eet.left_ray.direction)
                    left_ray = hl2ss_utilities.si_ray_transform(local_left_ray, data_eet.pose)
                    d = sm_manager.cast_rays(left_ray)
                    if (np.isfinite(d)):
                        left_point = hl2ss_utilities.si_ray_to_point(left_ray, d)
                        left_image_point = hl2ss_3dcv.project(left_point, world_to_image)
                        #hl2ss_utilities.draw_points(image, left_image_point.astype(np.int32), radius, left_color, thickness)
                    
                else:
                    left_image_point = null_arr

              #print(left_image_point)            

              # Draw Right Gaze Pointer ---------------------------------------------
                if (eet.right_ray_valid):
                    local_right_ray = hl2ss_utilities.si_ray_to_vector(eet.right_ray.origin, eet.right_ray.direction)
                    right_ray = hl2ss_utilities.si_ray_transform(local_right_ray, data_eet.pose)
                    d = sm_manager.cast_rays(right_ray)
                    if (np.isfinite(d)):
                        right_point = hl2ss_utilities.si_ray_to_point(right_ray, d)
                        right_image_point = hl2ss_3dcv.project(right_point, world_to_image)
                        #hl2ss_utilities.draw_points(image, right_image_point.astype(np.int32), radius, right_color, thickness)

                else:
                    right_image_point = null_arr

                #print(right_image_point)     
                # Draw Combined Gaze Pointer ------------------------------------------
                if (eet.combined_ray_valid):
                    local_combined_ray = hl2ss_utilities.si_ray_to_vector(eet.combined_ray.origin, eet.combined_ray.direction)
                    combined_ray = hl2ss_utilities.si_ray_transform(local_combined_ray, data_eet.pose)
                    d = sm_manager.cast_rays(combined_ray)
                    if (np.isfinite(d)):
                        combined_point = hl2ss_utilities.si_ray_to_point(combined_ray, d)
                        combined_image_point = hl2ss_3dcv.project(combined_point, world_to_image)
                        #hl2ss_utilities.draw_points(image, combined_image_point.astype(np.int32), radius, combined_color, thickness)

                else:
                    combined_image_point = null_arr

            except:
                left_image_point, right_image_point, combined_image_point = None

            #print(combined_image_point)

            start_x, start_z = None, None


            if si.is_valid_head_pose():
                head_pose = si.get_head_pose()
                position = head_pose.position
                forward = head_pose.forward
                up = head_pose.up
            else:
                position = [0, 0, 0]
                forward = [0, 0, 0]
                up = [0, 0, 0]

            csv_writer.writerow([
                datetime.datetime.now().isoformat(),
                format_list(left_image_point),
                format_list(right_image_point),
                format_list(combined_image_point),

                pv_intrinsics, 
                pv_extrinsics,

                format_list(data_pv.payload.focal_length), 
                format_list(data_pv.payload.principal_point),

                format_list(eet.left_ray.origin),
                format_list(eet.left_ray.direction),

                format_list(eet.right_ray.origin),
                format_list(eet.right_ray.direction),

                format_list(eet.combined_ray.origin),
                format_list(eet.combined_ray.direction),

                format_list(position),
                format_list(forward),
                format_list(up),

                image_filename,
                depth_filename,
                ab_filename
            ])            
            
            # Show the image
            #cv2.imshow('Video', image)
            #cv2.waitKey(1)
            print(f"Data recorded at {datetime.datetime.now().isoformat()}")
        # Shutdown ---------------------------------------------------------------
    sm_manager.close()

    sink_pv.detach()
    sink_eet.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)
    producer.stop(hl2ss.StreamPort.EXTENDED_EYE_TRACKER)
    producer.stop(hl2ss.StreamPort.SPATIAL_INPUT)
    

    # Stop keyboard events ----------------------------------------------------
    listener.join()
    

    """
    3d mapping
    """   
    import open3d as o3d
    import hl2ss_sa
    # Maximum triangles per cubic meter
    tpcm = 1000

    # Data format
    vpf = hl2ss.SM_VertexPositionFormat.R32G32B32A32Float
    tif = hl2ss.SM_TriangleIndexFormat.R32Uint
    vnf = hl2ss.SM_VertexNormalFormat.R32G32B32A32Float

    # Include normals
    normals = True

    # include bounds
    bounds = False

    # Maximum number of active threads (on the HoloLens) to compute meshes
    threads = 2

    # Region of 3D space to sample (bounding box)
    # All units are in meters
    center  = [0.0, 0.0, 0.0] # Position of the box
    extents = [8.0, 8.0, 8.0] # Dimensions of the box

    #------------------------------------------------------------------------------ 

    # Download meshes -------------------------------------------------------------
    client_3D = hl2ss_lnm.ipc_sm(host, hl2ss.IPCPort.SPATIAL_MAPPING)

    client_3D.open()

    client_3D.create_observer()

    volumes = hl2ss.sm_bounding_volume()
    volumes.add_box(center, extents)
    client_3D.set_volumes(volumes)

    surface_infos = client_3D.get_observed_surfaces()
    tasks = hl2ss.sm_mesh_task()
    for surface_info in surface_infos:
        tasks.add_task(surface_info.id, tpcm, vpf, tif, vnf, normals, bounds)

    meshes = client_3D.get_meshes(tasks, threads)

    client_3D.close()

    print(f'Observed {len(surface_infos)} surfaces')

    # Combine all meshes into one -------------------------------------------------
    combined_mesh = o3d.geometry.TriangleMesh()

    for index, mesh in meshes.items():
        id_hex = surface_infos[index].id.hex()
        timestamp = surface_infos[index].update_time

        if mesh is None:
            print(f'Task {index}: surface id {id_hex} compute mesh failed')
            continue

        mesh.unpack(vpf, tif, vnf)

        hl2ss_3dcv.sm_mesh_normalize(mesh)

        open3d_mesh = hl2ss_sa.sm_mesh_to_open3d_triangle_mesh(mesh)
        open3d_mesh = hl2ss_sa.open3d_triangle_mesh_swap_winding(open3d_mesh)
        open3d_mesh.vertex_colors = open3d_mesh.vertex_normals

        # Offset the indices of the combined mesh
        triangles = np.asarray(open3d_mesh.triangles) + len(combined_mesh.vertices)

        # Append vertices and triangles to the combined mesh
        combined_mesh.vertices.extend(open3d_mesh.vertices)
        combined_mesh.triangles.extend(o3d.utility.Vector3iVector(triangles))
        combined_mesh.vertex_colors.extend(open3d_mesh.vertex_colors)

    # Speichern des kombinierten Meshs mit inkrementeller Benennung -----------------------------------

    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    output_dir = data_folder
    os.makedirs(output_dir, exist_ok=True)

    # Nächsten verfügbaren Dateinamen finden
    file_index = 1
    filename = os.path.join(output_dir, f'combined_mesh{file_index}.ply')
    while os.path.exists(filename):
        file_index += 1
        filename = os.path.join(output_dir, f'combined_mesh{file_index}.ply')

    # Speichern des kombinierten Meshs
    o3d.io.write_triangle_mesh(filename, combined_mesh)
    print(f'Saved combined mesh to {filename}')

