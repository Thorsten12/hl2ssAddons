#------------------------------------------------------------------------------
# This script receives video frames and extended eye tracking data from the 
# HoloLens. The received left, right, and combined gaze pointers are projected
# onto the video frame.
# Press esc to stop.
#------------------------------------------------------------------------------

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


def fetchData(queue, sink_pv, sink_eet, sink_si, sm_manager, sphere_radius):
    while True:
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

            # Mesh-Update mit Cleanup
            volume = hl2ss.sm_bounding_volume()
            volume.add_sphere(new_position, sphere_radius)
            sm_manager.set_volumes(volume)
            sm_manager.get_observed_surfaces()
            meshSM = sm_manager.get_meshes()  # Nur das Mesh speichern

            # Daten in die Queue packen
            queue.put({
                "image": image,
                "eet": eet,
                "si": si,
                "new_position": new_position,
                "meshSM": meshSM  # Nur das Mesh, nicht das ganze sm_manager Objekt!
            })
        
        except Exception as e:
            print(f"Fehler in fetchData: {e}")

def main():
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

    #mp.set_start_method("spawn")
    queue = manager.Queue()

    # Starte den Datensammler-Prozess
    fetch_process = mp.Process(target=fetchData, args=(queue, sink_pv, sink_eet, sink_si, sm_manager, sphere_radius))
    fetch_process.start()
    

    try:
        last_time = time.time()  # Startzeit für die erste Messung

        while True:
            if not queue.empty():
                data = queue.get()

                # FPS Berechnung: Zeit seit dem letzten Datenpaket
                current_time = time.time()
                delta_time = current_time - last_time  # Zeitdifferenz seit letztem Frame
                last_time = current_time  # Zeitstempel für das nächste Frame setzen

                if delta_time > 0:  # Verhindert Division durch Null
                    fps = 1.0 / delta_time

                    print(f"data: {data} - FPS: {fps:.2f}")
                
            
            #time.sleep(0.01)  # Leichte Pause, um CPU-Last zu reduzieren

    except KeyboardInterrupt:
        print("Beende den Prozess...")
        fetch_process.terminate()



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