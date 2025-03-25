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

# Settings --------------------------------------------------------------------

# HoloLens 2 address
host = "192.168.137.140"
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

# Lock für den Zugriff auf sm_manager
sm_manager_lock = threading.Lock()

def convert2D3D(x_2D, y_2D, screen_width=760, screen_height=428, distance_to_screen=1.0, wert = 0.1):
        x_Mitte = screen_width / 2
        y_Mitte = screen_height / 2

        x_Abstand = x_2D - x_Mitte 
        y_Abstand = y_2D - y_Mitte

        x_Wert = x_Abstand * wert
        y_Wert = y_Abstand * wert

        return [ y_Wert, x_Wert, distance_to_screen]

# Funktion zur Aktualisierung der Werte
def update_x(val):
    global x_val
    x_val = val

def update_y(val):
    global y_val
    y_val = val

def update_z(val):
    global z_val
    z_val = val

# Initialisierung der GUI im Hauptthread
def init_gui():
    global fig, slider_x, slider_y, slider_z

    fig, ax = plt.subplots(figsize=(4, 3))
    plt.subplots_adjust(left=0.1, bottom=0.45)  # Platz für drei Slider schaffen

    # Slider für X-Wert
    ax_slider_x = plt.axes([0.1, 0.3, 0.8, 0.05])
    slider_x = Slider(ax_slider_x, 'X', 0.0, 1.0, valinit=x_val)
    slider_x.on_changed(update_x)

    # Slider für Y-Wert
    ax_slider_y = plt.axes([0.1, 0.2, 0.8, 0.05])
    slider_y = Slider(ax_slider_y, 'Y', 0.0, 1.0, valinit=y_val)
    slider_y.on_changed(update_y)

    # Neuer Slider für Z-Wert
    ax_slider_z = plt.axes([0.1, 0.1, 0.8, 0.05])
    slider_z = Slider(ax_slider_z, 'Z', 0.0, 1.0, valinit=z_val)
    slider_z.on_changed(update_z)

    # GUI im nicht-blockierenden Modus anzeigen
    plt.ion()
    plt.show(block=False)

# Update der GUI im Hauptthread
def update_gui():
    if fig is not None:
        plt.pause(0.001)  # Erlaubt der GUI zu aktualisieren ohne zu blockieren

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

def process_mesh_data(queue, sm_manager, shared_queue, ML_queue):
    print("mesh data started")
    """ Thread zum Verarbeiten der Mesh-Daten """
    last_time = time.time()  # Startzeit für die FPS-Berechnung
    global stop_threads
    while not stop_threads:
        try:
            #print("Versuche, Daten aus der Queue zu holen...")
            data = queue.get()  # Das Dictionary abrufen
            try:
                Mittelpunkt = ML_queue.get(timeout=1)

                #print(f"Daten erhalten: {data}")

                # Die Werte aus dem Dictionary extrahieren
                new_position = data["new_position"]
                data_eet = data["data_eet"]
                #print(f"Extrahierte Werte - eet: {eet}, new_position: {new_position}, data_eet: {data_eet}, data_pv: {data_pv}")

                # Berechnung im Thread durchführen
                sm_manager = mesh_update(new_position, sm_manager)
                #print("Mesh erfolgreich aktualisiert.")

                # Berechnung des combined_point
                #local_combined_ray = np.vstack((eet.combined_ray, eet.combined_ray.direction)).reshape((-1, 6))
                local_combined_ray = np.array([[0, 0, 0,  Mittelpunkt[0],  Mittelpunkt[1],  Mittelpunkt[2] ]], dtype=np.float32)
                #print(f"local_combined_ray: {local_combined_ray}")

                combined_ray = np.hstack((
                    local_combined_ray[:, 0:3] @ data_eet.pose[:3, :3] + 
                    data_eet.pose[3, :3].reshape(([1] * (len(local_combined_ray[:, 0:3].shape) - 1)).append(3)), 
                    local_combined_ray[:, 3:6] @ data_eet.pose[:3, :3]
                ))
                #print(f"combined_ray: {combined_ray}")

                surfaces = sm_manager._get_surfaces()
                #print(f"Anzahl der Oberflächen: {len(surfaces)}")

                n = len(surfaces)
                distances = np.ones(combined_ray.shape[0:-1] + (n if (n > 0) else 1,)) * np.inf

                for index, entry in enumerate(surfaces):
                    distances[..., index] = entry.rcs.cast_rays(combined_ray)['t_hit'].numpy()
                #print(f"Distanzen berechnet: {distances}")

                distances = np.min(distances, axis=-1)
                d = distances
                #print(f"Minimale Distanz: {d}")

                if np.isfinite(d).all():
                    combined_point = (combined_ray[:, 0:3] + d * combined_ray[:, 3:6]).reshape((-1, 3)).flatten().tolist()
                    #print(f"Berechneter combined_point: {combined_point}")

                    # Den combined_point an die Queue übergeben, damit andere Threads darauf zugreifen können
                    shared_queue.put(combined_point)
                    #print("combined_point in shared_queue gespeichert.")

                # FPS Berechnung
                """
                current_time = time.time()
                delta_time = current_time - last_time  # Zeitdifferenz seit letztem Frame
                last_time = current_time  # Zeitstempel für das nächste Frame setzen

                if delta_time > 0:  # Verhindert Division durch Null
                    fps = 1.0 / delta_time
                    print(f"FPS: {fps:.2f}")  # FPS-Ausgabe"
                    """
            except:
                print("Queue empty")
                continue
            
        except Exception as e:
            print("Ein Fehler ist aufgetreten!")
            traceback.print_exc()  # Zeigt den kompletten Stack-Trace für genaues Debugging
            time.sleep(0.01)  # Verhindert 100% CPU-Auslastung      

def pv_view(queue, ML_queue):
    global stop_threads, x_val, y_val, z_val
    while not stop_threads:
        try:
            data = queue.get()
            image = data["image"]
            print("got FullData!")

            #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Definiere den Bereich für die Blautöne
            # Orange liegt typischerweise im Bereich von Hue: 10 bis 25
            #lower_orange = np.array([10, 100, 100])  # Untere Grenze des Orangetons
            #upper_orange = np.array([25, 255, 255])  # Obere Grenze des Orangetons

            # Maske erstellen, die nur orange Pixel enthält
            #orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
            # Finde die Konturen der orangefarbenen Bereiche
            #contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #min_area = 1000
            #valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
            
            # Wenn Konturen vorhanden sind
            if False and  valid_contours :
                # Wähle die größte Kontur (die mit der höchsten Fläche)
                largest_contour = max(valid_contours, key=cv2.contourArea)

                # Berechne den Mittelpunkt der größten Kontur
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:  # Vermeidet Division durch Null
                    cx = int(M["m10"] / M["m00"])  # Mittelpunkt x
                    cy = int(M["m01"] / M["m00"])  # Mittelpunkt y
                print(cx,cy)
                x, y = int(760 / 2), int(428 / 2)
                Vector = convert2D3D(x, y)
                Vector2 = convert2D3D(cx, cy)
                print("1; ", Vector)
                print("2;", Vector2)
                ML_queue.put(Vector2)
                
                # Visualisiere den Mittelpunkt
                cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)  # Kontur zeichnen
                cv2.circle(image, (x, y), 7, (0, 0, 255), -1)  # Mittelpunkt als rotes Kreis markieren
                cv2.circle(image, (cx, cy), 7, (0, 255, 255), -1)  # Mittelpunkt als rotes Kreis markieren
            else:
                wert = 0.1
                # Berechne die Pixelkoordinaten aus den Slider-Werten
                print("Val:", x_val, y_val)

                pixel_x = int((1 - (x_val + 0.107 - ((0.5 - x_val) * 0.31))) * pv_width) # Die Werte, sonst scheint 
                pixel_y = int((1 - (y_val + 0.031 - ((0.5 - y_val) * 0.31))) * pv_height) # es nicht ganz allignet zu sein

                vis_pixel_x = int(x_val * pv_width)
                vis_pixel_y = int((1 - y_val) * pv_height)


                print("Pixel:", pixel_x, pixel_y)
                
                # Zeichne einen Kreis an der aktuellen Slider-Position
                cv2.circle(image, (vis_pixel_x, vis_pixel_y), 7, (0, 255, 255), -1)
                
                # Berechne den 3D-Vektor aus Slider-Werten
                Vector = convert2D3D(pixel_x, pixel_y, wert=0.0013)
                ML_queue.put(Vector)
                print("put in MLQueue:", Vector)
                # Zeichne ein Fadenkreuz in der Mitte
                center_x, center_y = int(pv_width / 2), int(pv_height / 2)
                cv2.line(image, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
                cv2.line(image, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)
                
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
    
    # GUI schließen, falls sie offen ist
    if plt.fignum_exists(1):
        plt.close('all')

def main():
    global stop_threads
    
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

    # Initialisiere das GUI im Hauptthread
    init_gui()

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
            update_gui()  # Update GUI im Hauptthread
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