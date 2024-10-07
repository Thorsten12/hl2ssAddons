#------------------------------------------------------------------------------
# This script receives video from the HoloLens front RGB camera and plays it.
# The camera supports various resolutions and framerates. See
# https://github.com/jdibenes/hl2ss/blob/main/etc/pv_configurations.txt
# for a list of supported formats. The default configuration is 1080p 30 FPS. 
# The stream supports three operating modes: 0) video, 1) video + camera pose, 
# 2) query calibration (single transfer).
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm

import csv
import os
import time

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.168.137.174"

# Operating mode
# 0: video
# 1: video + camera pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Enable Mixed Reality Capture (Holograms)
enable_mrc = False

# Enable Shared Capture
# If another program is already using the PV camera, you can still stream it by
# enabling shared mode, however you cannot change the resolution and framerate
shared = False

# Camera parameters
# Ignored in shared mode
width     = 1920
height    = 1080
framerate = 30

# Base CSV file name
base_csv_filename = 'video_data'

# Framerate denominator (must be > 0)
# Effective FPS is framerate / divisor
divisor = 1 

# Video encoding profile and bitrate (None = default)
profile = hl2ss.VideoProfile.H265_MAIN
bitrate = None

# Decoded format
# Options include:
# 'bgr24'
# 'rgb24'
# 'bgra'
# 'rgba'
# 'gray8'
decoded_format = 'bgr24'

#------------------------------------------------------------------------------

def get_unique_filename(base_name):
    index = 0
    while True:
        if index == 0:
            filename = f"{base_name}.csv"
        else:
            filename = f"{base_name}_{index}.csv"
        
        if not os.path.exists(filename):
            return filename
        index += 1


hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)

if (mode == hl2ss.StreamMode.MODE_2):
    data = hl2ss_lnm.download_calibration_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width, height, framerate)
    print('Calibration')
    print(f'Focal length: {data.focal_length}')
    print(f'Principal point: {data.principal_point}')
    print(f'Radial distortion: {data.radial_distortion}')
    print(f'Tangential distortion: {data.tangential_distortion}')
    print('Projection')
    print(data.projection)
    print('Intrinsics')
    print(data.intrinsics)
    print('RigNode Extrinsics')
    print(data.extrinsics)
    print(f'Intrinsics MF: {data.intrinsics_mf}')
    print(f'Extrinsics MF: {data.extrinsics_mf}')
else:
    enable = True

    def on_press(key):
        global enable
        enable = key != keyboard.Key.esc
        return enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, divisor=divisor, profile=profile, bitrate=bitrate, decoded_format=decoded_format)
    client.open()

    csv_filename = get_unique_filename(base_csv_filename)
    print(f"Writing data to: {csv_filename}")

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'Data'])
        while (enable):
            data = client.get_next_packet()
            output = f"""

            Frame captured at {data.timestamp}
            Focal length: {data.payload.focal_length}
            Principal point: {data.payload.principal_point}
            Exposure Time: {data.payload.exposure_time}
            Exposure Compensation: {data.payload.exposure_compensation}
            Lens Position (Focus): {data.payload.lens_position}
            Focus State: {data.payload.focus_state}
            ISO Speed: {data.payload.iso_speed}
            White Balance: {data.payload.white_balance}
            ISO Gains: {data.payload.iso_gains}
            White Balance Gains: {data.payload.white_balance_gains}
            Pose: {data.pose}
            """
            csv_writer.writerow([time.time(), output.strip()])
        
    client.close()
    listener.join()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
