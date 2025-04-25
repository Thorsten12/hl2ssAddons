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
import time

import SM_convert
import Config

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

def convert2D3D(x_2D, y_2D, screen_width=760, screen_height=428, distance_to_screen=1.0, wert = 0.1):
        x_Mitte = screen_width / 2
        y_Mitte = screen_height / 2

        x_Abstand = x_2D - x_Mitte 
        y_Abstand = y_2D - y_Mitte

        x_Wert = x_Abstand * wert
        y_Wert = y_Abstand * wert

        return [ y_Wert, x_Wert, distance_to_screen]

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

def process_mesh_data(queue, sm_manager, shared_queue, ML_queue):
    print("Mesh Data Thread gestartet.")
    
    last_time = time.time()  # Startzeit für die FPS-Berechnung
    global stop_threads

    while not stop_threads:
        try:
            print("Versuche, Daten aus der Hauptqueue zu holen...")
            data = queue.get()  # Das Dictionary abrufen
            print(f"Daten aus der Hauptqueue erhalten: {data}")

            try:
                print("Versuche, Mittelpunkt-Daten aus ML_queue zu holen...")
                Mittelpunkt = ML_queue.get(timeout=3)
                print(f"Daten aus ML_queue erhalten: {Mittelpunkt}")

                # Die Werte aus dem Dictionary extrahieren
                new_position = data["new_position"]
                data_eet = data["data_eet"]
                print(f"Extrahierte Werte - new_position: {new_position}, data_eet: {data_eet}")

                # Berechnung des combined_point
                local_combined_ray = np.array([[0, 0, 0, Mittelpunkt[0], Mittelpunkt[1], Mittelpunkt[2]]], dtype=np.float32)
                print(f"local_combined_ray: {local_combined_ray}")
                combined_ray = np.hstack((
                    local_combined_ray[:, 0:3] @ data_eet.pose[:3, :3] + 
                    data_eet.pose[3, :3].reshape(([1] * (len(local_combined_ray[:, 0:3].shape) - 1)).append(3)), 
                    local_combined_ray[:, 3:6] @ data_eet.pose[:3, :3]
                ))
                print("combined_ray:", combined_ray)
                try:
                    print("Versuche, Oberflächen aus sm_manager zu holen...")
                    print(type(sm_manager))
                    surfaces = sm_manager._get_surfaces()
                    print(surfaces, type(surfaces))
                    print(f"Anzahl der Oberflächen erhalten: {len(surfaces)}")
                except Exception as sm_error:
                    print("Fehler beim Abrufen der Oberflächen aus sm_manager:")
                    traceback.print_exc()

                n = len(surfaces)
                #distances = np.ones(combined_ray.shape[0:-1] + (n if (n > 0) else 1,)) * np.inf

                #distances = surfaces.cast_rays(combined_ray)['t_hit'].numpy() # distances[..., index] = entry.rcs.cast_rays(combined_ray)['t_hit'].numpy()

                #d = distances
                d = sm_manager.cast_rays(combined_ray)
                print(f"Minimale Distanz: {d}")

                if np.isfinite(d).all():
                    combined_point = (combined_ray[:, 0:3] + d * combined_ray[:, 3:6]).reshape((-1, 3)).flatten().tolist()
                    print(f"Berechneter combined_point: {combined_point}")

                    # Den combined_point an die Queue übergeben
                    shared_queue.put(combined_point)
                    print("combined_point erfolgreich in shared_queue gespeichert.")

            except Exception as ml_queue_error:
                print("Fehler beim Verarbeiten der ML_queue:")
                traceback.print_exc()
                continue

        except Exception as e:
            print("Ein Fehler ist aufgetreten!")
            traceback.print_exc()
            time.sleep(0.01)  # Verhindert 100% CPU-Auslastung

def pv_view(queue, ML_queue):
    global stop_threads, x_val, y_val, z_val
    while not stop_threads:
        try:
            data = queue.get()
            image = data["image"]
            #print("got FullData!")

            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Definiere den Bereich für die Blautöne
            # Orange liegt typischerweise im Bereich von Hue: 10 bis 25
            lower_orange_red = np.array([0, 100, 100])   # Dunkelrot bis tiefes Orange
            upper_orange_red = np.array([30, 255, 255])  # Sattes Orange bis leicht rötlich

            # Maske erstellen, die nur orange Pixel enthält
            orange_mask = cv2.inRange(hsv_image, lower_orange_red, upper_orange_red)
            # Finde die Konturen der orangefarbenen Bereiche
            contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 1000
            valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
            
            # Wenn Konturen vorhanden sind
            if valid_contours :
                # Wähle die größte Kontur (die mit der höchsten Fläche)
                largest_contour = max(valid_contours, key=cv2.contourArea)

                # Berechne den Mittelpunkt der größten Kontur
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:  # Vermeidet Division durch Null
                    cx = int(M["m10"] / M["m00"])  # Mittelpunkt x
                    cy = int(M["m01"] / M["m00"])  # Mittelpunkt y
      
                x_val2 = cx / pv_width
                y_val2 = -cy / pv_height + 1

                pixel_x2 = int((1 - (x_val2 + 0.107 - ((0.5 - x_val2) * 0.31))) * pv_width) # Die Werte, sonst scheint 
                pixel_y2 = int((1 - (y_val2 + 0.031 - ((0.5 - y_val2) * 0.31))) * pv_height) # nicht zu allignen

                """

                pixel_x = int((1 - (x_val + 0.107 - ((0.5 - x_val) * 0.31))) * pv_width) # Die Werte, sonst scheint 
                pixel_y = int((1 - (y_val + 0.031 - ((0.5 - y_val) * 0.31))) * pv_height) # es nicht ganz allignet zu sein

                vis_pixel_x = int(x_val * pv_width)
                vis_pixel_y = int((1 - y_val) * pv_height)
                
                
                """
                # Zeichne einen Kreis an der aktuellen Slider-Position
                cv2.circle(image, (cx, cy), 7, (0, 255, 255), -1)
                
                # Berechne den 3D-Vektor aus Slider-Werten
                Vector = convert2D3D(pixel_x2, pixel_y2, wert=0.0013)
                print(Vector)
                ML_queue.put(Vector)
               
                # Visualisiere den Mittelpunkt
                cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)  # Kontur zeichnen
                cv2.circle(image, (cx, cy), 7, (0, 255, 255), -1)  # Mittelpunkt als rotes Kreis markieren
            else:
                # Berechne die Pixelkoordinaten aus den Slider-Werten
                #print("Val:", x_val, y_val)
                print("no MIttelpnkt")

                ...
                
                
            # Display frame
            cv2.imshow('Video', image)
            cv2.waitKey(1)

        except Exception as e:
            print(f"Fehler in pv_view: {e}")
            time.sleep(0.1)

# -------  HoloLensVisualization
def HLV_thread(shared_queue, display_list, ipc, key, rotation, scale): 
    global stop_threads
    while not stop_threads:
        try:
            data = shared_queue.get()
            coord = [-x if i == 2 else x for i, x in enumerate(data)]
            
            display_list = hl2ss_rus.command_buffer()
            display_list.begin_display_list()
            display_list.set_world_transform(key, coord, rotation, scale)
            display_list.end_display_list()
            ipc.push(display_list)
            results = ipc.pull(display_list)

        except Exception as e:
            print(f"Fehler in HLV_thread: {e}")
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
    dummy_dict = SM_convert.sm_mesh_to_sm_manager("C:/Users/admin/Desktop/hl2ss/viewer/meshes/spatial_mapping_mesh_1.ply")
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

    ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
    ipc.open()

    key = 0

    position = [0, 0, 0]
    # Initial rotation in world space (x, y, z, w) as a quaternion
    rotation = [0, 0, 0, 1]
    # Initial scale in meters
    scale = [0.05, 0.05, 0.05]
    # Initial color
    rgba = [1, 1, 1, 1]

    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.remove_all() # Remove all objects that were created remotely
    display_list.create_primitive(hl2ss_rus.PrimitiveType.Sphere) # Create a sphere, server will return its id
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the sphere
    display_list.set_world_transform(key, position, rotation, scale) # Set the world transform of the sphere
    display_list.set_color(key, rgba) # Set the color of the sphere
    display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the sphere visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    key = results[2] # Get the sphere id, created by the 3rd command in the list

    queue = manager.Queue(1)
    shared_queue = manager.Queue(1)
    ML_queue = manager.Queue(1)


    # Starte den Datensammler-Prozess
    fetch_process = mp.Process(target=fetchData, args=(queue, sink_pv, sink_eet, sink_si))
    fetch_process.start()
    
    # Starte die Worker-Threads
    mesh_thread = threading.Thread(target=process_mesh_data, args=(queue, sm_manager, shared_queue, ML_queue), name="MeshThread")
    mesh_thread.start()

    cube_thread = threading.Thread(target=HLV_thread, args=(shared_queue, display_list, ipc, key, rotation, scale))
    cube_thread.start()

    pv_thread = threading.Thread(target=pv_view, args=(queue, ML_queue))
    pv_thread.start()
    
    try:
        # Hauptschleife mit GUI-Update
        while not stop_threads:
            time.sleep(0.01)  # Kleine Pause, um CPU-Auslastung zu verringern
            
            # Überprüfe, ob alle Threads noch laufen
            if not mesh_thread.is_alive() or not cube_thread.is_alive() or not pv_thread.is_alive():
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

        while not ML_queue.empty():
            try:
                ML_queue.get_nowait()
            except:
                pass
        print("Warte auf Beendigung der Threads...")
        mesh_thread.join(timeout=2)
        cube_thread.join(timeout=2)
        pv_thread.join(timeout=2)
        
        # Fetch-Prozess beenden
        if fetch_process.is_alive():
            fetch_process.terminate()
            fetch_process.join(timeout=2)
        
        # IPC Verbindung schließen
        print("Schließe IPC Verbindung...")
        ipc.close()
        
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