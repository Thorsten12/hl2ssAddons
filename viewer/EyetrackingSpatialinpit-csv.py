import hl2ss
import hl2ss_lnm
from pynput import keyboard
import csv
import datetime
import os
import threading
import time

# Einstellungen
host = '192.168.137.174'
fps = 30
base_folder_name = 'data'

# Eyetracker initialisieren
enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

def get_unique_folder(base_name):
    index = 0
    while True:
        folder_name = f"{base_name}{index}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name
        index += 1

# Verzögerter Verbindungsaufbau
def initialize_clients():
    global client_eye_tracker, client_spatial
    
    # Eye Tracker initialisieren
    client_eye_tracker = hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=fps)
    client_eye_tracker.open()
    print("Eye Tracker verbunden.")
    time.sleep(0.5)  # Verzögerung

    # Spatial Input initialisieren
    client_spatial = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
    client_spatial.open()
    print("Spatial Input verbunden.")

# Listener für Tastatureingaben
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Verbindung zu HoloLens-Datenströmen herstellen
initialize_clients()

data_folder = get_unique_folder(base_folder_name)
csv_filename = os.path.join(data_folder, 'data.csv')

print(f"Writing data to: {csv_filename}")

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([
        'Timestamp', 'HoloLens Eye Tracker Timestamp', 'Calibration Valid',
        'Combined Gaze Valid', 'Combined Gaze Origin X', 'Combined Gaze Origin Y', 'Combined Gaze Origin Z',
        'Combined Gaze Direction X', 'Combined Gaze Direction Y', 'Combined Gaze Direction Z',
        'Left Gaze Valid', 'Left Gaze Origin X', 'Left Gaze Origin Y', 'Left Gaze Origin Z',
        'Left Gaze Direction X', 'Left Gaze Direction Y', 'Left Gaze Direction Z',
        'Right Gaze Valid', 'Right Gaze Origin X', 'Right Gaze Origin Y', 'Right Gaze Origin Z',
        'Right Gaze Direction X', 'Right Gaze Direction Y', 'Right Gaze Direction Z',
        'Left Eye Openness Valid', 'Left Eye Openness',
        'Right Eye Openness Valid', 'Right Eye Openness',
        'Vergence Distance Valid', 'Vergence Distance',
        'HoloLens Position X', 'HoloLens Position Y', 'HoloLens Position Z',
        'HoloLens Forward X', 'HoloLens Forward Y', 'HoloLens Forward Z',
        'HoloLens Up X', 'HoloLens Up Y', 'HoloLens Up Z'
    ])

    while enable:
        try:
            # Eye Tracking Daten abrufen
            data_eye_tracking = client_eye_tracker.get_next_packet()
            eet = hl2ss.unpack_eet(data_eye_tracking.payload)

            # Spatial Input Daten abrufen
            data_spatial = client_spatial.get_next_packet()
            si = hl2ss.unpack_si(data_spatial.payload)

            # Extrahieren der Position und Orientierung
            if si.is_valid_head_pose():
                head_pose = si.get_head_pose()
                position = head_pose.position
                forward = head_pose.forward
                up = head_pose.up
            else:
                position = [0, 0, 0]
                forward = [0, 0, 0]
                up = [0, 0, 0]

            # CSV-Daten schreiben
            csv_writer.writerow([
                datetime.datetime.now().isoformat(),
                data_eye_tracking.timestamp,
                eet.calibration_valid,
                eet.combined_ray_valid,
                *eet.combined_ray.origin,
                *eet.combined_ray.direction,
                eet.left_ray_valid,
                *eet.left_ray.origin,
                *eet.left_ray.direction,
                eet.right_ray_valid,
                *eet.right_ray.origin,
                *eet.right_ray.direction,
                eet.left_openness_valid,
                eet.left_openness,
                eet.right_openness_valid,
                eet.right_openness,
                eet.vergence_distance_valid,
                eet.vergence_distance,
                *position,  # Position
                *forward,   # Forward-Vektor
                *up         # Up-Vektor
            ])

            print(f"Data recorded at {datetime.datetime.now().isoformat()}")
        except Exception as e:
            print(f"Fehler beim Datenabruf: {e}")

client_eye_tracker.close()
client_spatial.close()
listener.join()
