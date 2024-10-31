import cv2
import numpy as np
import hl2ss
import hl2ss_lnm
from pynput import keyboard
import csv
import datetime
import os
import threading
from Tools import hl2ss_3dcv

# Einstellungen
host = '192.168.137.174'
fps = 30
base_folder_name = 'data'

# Kameraeinstellungen
mode = hl2ss.StreamMode.MODE_1
width = 1920
height = 1080
framerate = fps
profile = hl2ss.VideoProfile.H265_MAIN
decoded_format = 'bgr24'

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
            os.makedirs(os.path.join(folder_name, 'images'))
            return folder_name
        index += 1

def save_image(image_data, filename):
    cv2.imwrite(filename, image_data)

listener = keyboard.Listener(on_press=on_press)
listener.start()

# Verbindung zu HoloLens-Datenströmen herstellen
client_eye_tracker = hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=fps)
client_eye_tracker.open()

hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=False, shared=False)

client_video = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, decoded_format=decoded_format)
client_video.open()

client_spatial = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
client_spatial.open()

data_folder = get_unique_folder(base_folder_name)
csv_filename = os.path.join(data_folder, 'data.csv')
image_folder = os.path.join(data_folder, 'images')

print(f"Writing data to: {csv_filename}")
print(f"Saving images to: {image_folder}")

# CSV-Datei schreiben mit den umbenannten Spaltennamen
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
        'HoloLens Video Timestamp', 'Focal Length X', 'Focal Length Y',
        'Principal Point X', 'Principal Point Y',
        'Image Filename',
        'HoloLens Position X', 'HoloLens Position Y', 'HoloLens Position Z',
        'HoloLens Forward X', 'HoloLens Forward Y', 'HoloLens Forward Z',
        'HoloLens Up X', 'HoloLens Up Y', 'HoloLens Up Z'
    ])

    # Erfassen des Startpunkts für Positions-Offset
    start_x, start_z = None, None

    while enable:
        data_eye_tracking = client_eye_tracker.get_next_packet()
        eet = hl2ss.unpack_eet(data_eye_tracking.payload)

        data_video = client_video.get_next_packet()
        data_spatial = client_spatial.get_next_packet()
        si = hl2ss.unpack_si(data_spatial.payload)

        image_filename = os.path.join(image_folder, f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
        threading.Thread(target=save_image, args=(data_video.payload.image, image_filename)).start()

        # Extrahieren der Position und Orientierung aus si
        if si.is_valid_head_pose():
            head_pose = si.get_head_pose()
            position = head_pose.position
            forward = head_pose.forward
            up = head_pose.up
        else:
            position = [0, 0, 0]
            forward = [0, 0, 0]
            up = [0, 0, 0]

        # Startwerte für Position X und Z setzen, wenn sie noch nicht festgelegt sind
        if start_x is None and start_z is None:
            start_x, start_z = position[0], position[2]

        # Offset der Position X und Z anwenden
        position[0] -= start_x
        position[2] -= start_z

        # Pfadname anpassen
        image_filename = image_filename.replace('data2\\images\\', '')

        # Schreiben der Zeile in die CSV-Datei
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
            data_video.timestamp,
            *data_video.payload.focal_length,
            *data_video.payload.principal_point,
            image_filename,
            *position,  # Position
            *forward,   # Forward-Vektor
            *up         # Up-Vektor
        ])

        print(f"Data recorded at {datetime.datetime.now().isoformat()}")

client_eye_tracker.close()
client_video.close()
client_spatial.close()
listener.join()
