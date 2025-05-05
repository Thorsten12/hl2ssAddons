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
import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import threading

import SM_convert
import Config

import datetime
import csv


# Settings --------------------------------------------------------------------

# HoloLens 2 address
host = Config.HOST
# Camera parameters
# See etc/hl2_capture_formats.txt for a list of supported formats
pv_width     = 760
pv_height    = 428
pv_framerate = 30

# Globale Variablen für den Slider
x_val, y_val, z_val = 0.5, 0.5, 1

# GUI Variablen
fig = None
slider_x = None
slider_y = None

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
triangles_per_cubic_meter = 10
mesh_threads = 6
sphere_center = [0, 0, 0]
sphere_radius = 0.5
buffer_size = 7

stop_threads = False

# --- Für Datenspeicherung ---
left_image_points = []
combined_image_points = []
right_image_points = []
null_array = [-1, -1]
save_data = True  

def get_unique_folder(base_name):
    index = 0
    while True:
        folder_name = f"{base_name}{index}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            os.makedirs(os.path.join(folder_name, 'images'))
            return folder_name
        index += 1

def save_image(image_data, filename):
    if image_data is None:
        print(f"Kein Bilddaten zum Speichern für {filename}")
        return

    success = cv2.imwrite(f"{filename}.jpg", image_data)
    if success:
        print(f"Bild erfolgreich gespeichert: {filename}.jpg")
    else:
        print(f"FEHLER: Bild konnte NICHT gespeichert werden: {filename}.jpg")


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

def process_mesh_data(queue, sm_manager, shared_queue):
    print("Mesh Data Thread gestartet.")
    
    last_time = time.time()  # Startzeit für die FPS-Berechnung
    global stop_threads

    while not stop_threads:
        try:
            #print("Versuche, Daten aus der Hauptqueue zu holen...")
            data = queue.get()  # Das Dictionary abrufen
            data_pv = data["data_pv"]
            eet = data["eet"]
            data_eet = data["data_eet"]
             # Update PV intrinsics ------------------------------------------------
            # PV intrinsics may change between frames due to autofocus
            pv_intrinsics = hl2ss.create_pv_intrinsics(data_pv.payload.focal_length, data_pv.payload.principal_point)
            pv_extrinsics = np.eye(4, 4, dtype=np.float32)
            pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

            # Compute world to PV image transformation matrix ---------------------
            world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)
            

            try:
                if (eet.combined_ray_valid):
                    local_combined_ray = hl2ss_utilities.si_ray_to_vector(eet.combined_ray.origin, eet.combined_ray.direction)
                    combined_ray = hl2ss_utilities.si_ray_transform(local_combined_ray, data_eet.pose)
                    d = sm_manager.cast_rays(combined_ray)
                    if (np.isfinite(d)):
                        combined_point = hl2ss_utilities.si_ray_to_point(combined_ray, d)
                        combined_image_point = hl2ss_3dcv.project(combined_point, world_to_image)
                        
                        # Den combined_point an die Queue übergeben
                        shared_queue.put(combined_image_point)
                        #print("combined_point erfolgreich in shared_queue gespeichert.")

            except Exception as ml_queue_error:
                print("Fehler beim Verarbeiten der ML_queue:")
                traceback.print_exc()
                continue

        except Exception as e:
            print("Ein Fehler ist aufgetreten!")
            traceback.print_exc()
            time.sleep(0.01)  # Verhindert 100% CPU-Auslastung

def pv_view(queue, shared_queue, csv_writer, image_folder):
    global stop_threads
    while not stop_threads:
        try:
            data = queue.get()
            image = data["image"]
            combined_image_point = shared_queue.get()
            # --- In Bild zeichnen ---
            if combined_image_point is not None:
                hl2ss_utilities.draw_points(image, combined_image_point.astype(np.int32), radius, combined_color, thickness)

            # --- Bild speichern ---
            
            if save_data:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(image_folder, str(timestamp))
                save_image(image, filename)
                # In CSV schreiben
                csv_writer.writerow([
                    timestamp,
                    combined_image_point,
                    os.path.basename(filename) + ".jpg"
                ])
                

            # --- Bild anzeigen ---
            cv2.imshow('Video', image)
            cv2.waitKey(1)

        except Exception as e:
            print(f"Fehler in pv_view: {e}")
            time.sleep(0.1)


def on_keyboard_interrupt():
    print("Shutting down gracefully...")
    global stop_threads
    stop_threads = True
    
    # Alle Threads und Prozesse stoppen
    for thread in threading.enumerate():
        if thread != threading.current_thread():
            if thread.is_alive():
                print(f"Stopping thread: {thread.name}")
                
    # Andere Ressourcen freigeben
    cv2.destroyAllWindows()
    

def main():
    global stop_threads
    
    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start Spatial Mapping data manager --------------------------------------
    # Download observed surfaces
    sm_manager = hl2ss_sa.sm_manager(host, triangles_per_cubic_meter, mesh_threads)
    sm_manager.open()

    # Set Surfaces
    dummy_dict = SM_convert.sm_mesh_to_sm_manager("C:/Users/Tsyri/OneDrive/Desktop/hl2ssAddons/hl2ss2_27_04/hl2ssAddons/viewer/meshes/spatial_mapping_mesh_1.ply")
    sm_manager.set_surfaces(dummy_dict)
    
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

    queue = manager.Queue(1)
    shared_queue = manager.Queue(1)
    # Ordner und CSV vorbereiten
    data_folder = get_unique_folder("data")
    csv_filename = os.path.join(data_folder, 'data.csv')
    image_folder = os.path.join(data_folder, 'images')
    
    print(f"Writing data to: {csv_filename}")
    print(f"Saving images to: {image_folder}")
    
    # CSV Datei erstellen und Header schreiben
    csvfile = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csvfile, delimiter=";")
    csv_writer.writerow([
        'TimeStamp',
        'CombinedImagePoint',
        'image_filename'
    ])

    # Starte den Datensammler-Prozess
    fetch_process = mp.Process(target=fetchData, args=(queue, sink_pv, sink_eet, sink_si))
    fetch_process.start()
    
    # Starte die Worker-Threads
    mesh_thread = threading.Thread(target=process_mesh_data, args=(queue, sm_manager, shared_queue), name="MeshThread")
    mesh_thread.start()

    pv_thread = threading.Thread(target=pv_view, args=(queue, shared_queue, csv_writer, image_folder))
    pv_thread.start()
    
    
    

    try:
        # Hauptschleife mit GUI-Update
        while not stop_threads:
            time.sleep(0.01)  # Kleine Pause, um CPU-Auslastung zu verringern
            
            # Überprüfe, ob alle Threads noch laufen
            if not mesh_thread.is_alive() or not pv_thread.is_alive():
                print("Ein Thread ist beendet. Stoppe Programm...")
                stop_threads = True
                
    except KeyboardInterrupt:
        print("Tastatur-Unterbrechung erkannt.")
    except Exception as e:
        print(f"Fehler in der Hauptschleife: {e}")
        traceback.print_exc()
    finally:
        # Aufräumen
        stop_threads = True
        
        time.sleep(1)
        # Auf Threads warten
         # Clear all queues to prevent blocking on queue.get()
        while not queue.empty():
            try:
                queue.get_nowait()
            except:
                pass

        while not shared_queue.empty():
            try:
                shared_queue.get_nowait()
            except:
                pass

        print("Warte auf Beendigung der Threads...")
        mesh_thread.join(timeout=2)
        pv_thread.join(timeout=2)
        
        # Fetch-Prozess beenden
        if fetch_process.is_alive():
            fetch_process.terminate()
            fetch_process.join(timeout=2)
        
        try:
            csvfile.close()
        except:
            pass
        # Sinks trennen
        print("Trenne Sinks...")
        sink_si.detach()
        sink_pv.detach()
        sink_eet.detach()
        
        # Produzenten stoppen
        print("Stoppe Stream-Produzenten...")
        producer.stop(hl2ss.StreamPort.SPATIAL_INPUT)
        producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.stop(hl2ss.StreamPort.EXTENDED_EYE_TRACKER)
        
        # Spatial Mapping Manager schließen
        print("Schließe Spatial Mapping Manager...")
        sm_manager.close()
        
        # PV-Subsystem stoppen
        print("Stoppe PV-Subsystem...")
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        
        # OpenCV-Fenster schließen
        cv2.destroyAllWindows()
        
        # Matplotlib-Fenster schließen
        plt.close('all')
        
        print("Herunterfahren abgeschlossen")


if __name__ == '__main__':
    main()