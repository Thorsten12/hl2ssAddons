#------------------------------------------------------------------------------
# This script receives video frames and extended eye tracking data from the 
# HoloLens. The received left, right, and combined gaze pointers are projected
# onto the video frame.
# Press esc to stop.
#------------------------------------------------------------------------------

import os
import signal
import sys
import threading
import time
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

stop_threads = False



# Lock für den Zugriff auf sm_manager
sm_manager_lock = threading.Lock()

def convert2D3D(x_2D, y_2D, screen_width=760, screen_height=428, distance_to_screen=1.0, wert = 0.1):
    x_Mitte = screen_width / 2
    y_Mitte = screen_height / 2

    x_Abstand = x_2D - x_Mitte 
    y_Abstand = y_2D - y_Mitte

    x_Wert = x_Abstand * wert
    y_Wert = y_Abstand * wert

    return [ y_Wert, -x_Wert, distance_to_screen]

def stop_threads_gracefully():
    global stop_thread
    stop_thread = True  # Flag setzen, um die Threads zu stoppen


def mesh_update(new_position, sm_manager, sphere_radius = 1):
    """ Funktion für das Mesh-Update mit sm_manager """
    with sm_manager_lock:
        # Mesh-Update mit Cleanup
        volume = hl2ss.sm_bounding_volume()
        volume.add_sphere(new_position, sphere_radius)
        sm_manager.set_volumes(volume)
        sm_manager.get_observed_surfaces()
    return sm_manager

def fetchData(queue, sink_pv, sink_eet, sink_si):
    global stop_threads
    while not stop_threads:
        try:
            sink_pv.acquire()
            
            # Holen der neuesten Sensordaten
            _, data_pv = sink_pv.get_most_recent_frame()
            _, data_eet = sink_eet.get_nearest(data_pv.timestamp)
            _, data_si = sink_si.get_nearest(data_pv.timestamp)

            # Extrahieren der Sensordaten
            image = data_pv.payload.image
            eet = hl2ss.unpack_eet(data_eet.payload)
            si = hl2ss.unpack_si(data_si.payload)
            new_position = np.array(si.get_head_pose().position)

            while(queue.qsize() != 0):
                _ = queue.get()
            # Daten in die Queue packen
            queue.put({
                "image": image,
                "eet": eet,
                "si": si,
                "new_position": new_position,
                "data_pv": data_pv,
                "data_eet": data_eet
            })
        
        except Exception as e:
            print(f"Fehler in fetchData: {e}")
            time.sleep(0.1)

def process_mesh_data(queue, sm_manager, shared_queue,):
    """ Thread zum Verarbeiten der Mesh-Daten """
    last_time = time.time()  # Startzeit für die FPS-Berechnung
    global stop_threads
    while not stop_threads:
        try:
            if not queue.empty():
                data = queue.get()  # Das Dictionary abrufen

                # Die Werte aus dem Dictionary extrahieren
                eet = data["eet"]
                new_position = data["new_position"]
                data_eet = data["data_eet"]
                data_pv = data["data_pv"]

                # Berechnung im Thread durchführen
                sm_manager = mesh_update(new_position, sm_manager)

                #pv_intrinsics = np.array([[-data_pv.payload.focal_length[0], 0, 0, 0], [0, data_pv.payload.focal_length[1], 0, 0], [data_pv.payload.principal_point[0], data_pv.payload.principal_point[1], 1, 0], [0, 0, 0, 1]], dtype=np.float32) 
                #pv_extrinsics = np.eye(4, 4, dtype=np.float32)
                #R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=pv_extrinsics.dtype)
                #pv_intrinsics[0, 0] = -pv_intrinsics[0, 0]
                #pv_extrinsics = pv_extrinsics @ R
                #pv_intrinsics, pv_extrinsics = (pv_intrinsics, pv_extrinsics)
                #world_to_image = np.linalg.inv(data_pv.pose) @ pv_extrinsics @ pv_intrinsics       

                # Berechnung des combined_point
                local_combined_ray = np.vstack((eet.combined_ray.origin, eet.combined_ray.direction)).reshape((-1, 6))                                                                                     
                combined_ray = np.hstack((local_combined_ray[:, 0:3] @ data_eet.pose[:3, :3] + data_eet.pose[3, :3].reshape(([1] * (len(local_combined_ray[:, 0:3].shape) - 1)).append(3)), local_combined_ray[:, 3:6] @ data_eet.pose[:3, :3]))
                surfaces = sm_manager._get_surfaces()
                n = len(surfaces)
                distances = np.ones(combined_ray.shape[0:-1] + (n if (n > 0) else 1,)) * np.inf
                for index, entry in enumerate(surfaces):
                    distances[..., index] = entry.rcs.cast_rays(combined_ray)['t_hit'].numpy()
                distances = np.min(distances, axis=-1)
                d = distances
                if (np.isfinite(d)):
                    combined_point = (combined_ray[:, 0:3] + d * combined_ray[:, 3:6]).reshape((-1, 3)).flatten().tolist()
                    #print("reingegeben", combined_point)
                    # Der combined_point an die Queue übergeben, damit andere Threads darauf zugreifen können
                    shared_queue.put(combined_point)
                    
                #time.sleep(0.01)  # Verhindert 100% CPU-Auslastung

                # FPS Berechnung: Zeit seit dem letzten Datenpaket
                current_time = time.time()
                delta_time = current_time - last_time  # Zeitdifferenz seit letztem Frame
                last_time = current_time  # Zeitstempel für das nächste Frame setzen

                if delta_time > 0:  # Verhindert Division durch Null
                    fps = 1.0 / delta_time
                    print(f"FPS: {fps:.2f}")  # FPS-Ausgabe

        except:
            ...


        #time.sleep(0.01)  # Verhindert 100% CPU-Auslastung      

# -------  HoloLensVisualization
def HLV_thread(shared_queue, display_list, ipc, key, rotation, scale): 
    cubes = []

    while True:
        try:
            if not shared_queue.empty():
                data = shared_queue.get()
                coord = [-x if i == 2 else x for i, x in enumerate(data)]

                try:
                    #head_cube2 = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
                    #head_cube2.paint_uniform_color([0, 1, 0])
                    # Verschiebe den Würfel an die gewünschte Position
                    #head_cube2.translate(coord)
                    #vis.add_geometry(head_cube2)
                    #cubes.append(head_cube2)
                    #coord Transform
                    display_list = hl2ss_rus.command_buffer()
                    display_list.begin_display_list()
                    display_list.set_world_transform(key, coord, rotation, scale)
                    display_list.end_display_list()
                    ipc.push(display_list)
                    results = ipc.pull(display_list)
                except:
                    ...

        except Exception as e:
            ...
            # Cube-Bewegung mit Delta
            #if prev_position is not None:
                #delta = new_position - prev_position
                #head_cube.translate(delta)
                #vis.update_geometry(head_cube)




def main():
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

    ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
    ipc.open()

    key = 0

    position = [0, 0, 0]
    # Initial rotation in world space (x, y, z, w) as a quaternion
    rotation = [0, 0, 0, 1]
    # Initial scale in meters
    scale = [0.2, 0.2, 0.2]
    # Initial color
    rgba = [1, 1, 1, 1]

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


    #mp.set_start_method("spawn")
    queue = manager.Queue(1)
    shared_queue = manager.Queue(1)

    # Starte den Datensammler-Prozess
    fetch_process = mp.Process(target=fetchData, args=(queue, sink_pv, sink_eet, sink_si))
    fetch_process.start()
    
    mesh_thread = threading.Thread(target=process_mesh_data, args=(queue, sm_manager, shared_queue), name="MeshThread")
    mesh_thread.start()

    cube_thread = threading.Thread(target=HLV_thread, args=(shared_queue, display_list, ipc, key, rotation, scale))
    cube_thread.start()

    def on_keyboard_interrupt():
        print("Shutting down gracefully...")
        stop_threads = True
        
        # Stop data collection first
        if fetch_process.is_alive():
            fetch_process.terminate()
            fetch_process.join(timeout=2)
        
        # Close IPC connection
        print("Closing IPC connection...")
        ipc.close()
        
        # Detach sinks
        print("Detaching sinks...")
        sink_si.detach()
        sink_pv.detach()
        sink_eet.detach()
        
        # Stop producers
        print("Stopping stream producers...")
        producer.stop(hl2ss.StreamPort.SPATIAL_INPUT)
        producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.stop(hl2ss.StreamPort.EXTENDED_EYE_TRACKER)
        
        # Close the spatial mapping manager
        print("Closing spatial mapping manager...")
        sm_manager.close()
        
        # Stop subsystem
        print("Stopping PV subsystem...")
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        
        print("Shutdown complete")
        
    try:
        # Instead of just joining, implement a proper wait loop that can be interrupted
        while not stop_threads:
            time.sleep(0.1)  # Small sleep to avoid CPU hogging
    except KeyboardInterrupt:
        on_keyboard_interrupt()
        sys.exit()


if __name__ == '__main__':
    main()