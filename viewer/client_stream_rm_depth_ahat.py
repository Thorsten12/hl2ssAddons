#------------------------------------------------------------------------------
# This script receives video from the HoloLens depth camera in ahat mode and 
# plays it. The resolution is 512x512 @ 45 FPS. The stream supports three 
# operating modes: 0) video, 1) video + rig pose, 2) query calibration (single 
# transfer). Depth and AB data are scaled for visibility. The ahat and long 
# throw streams cannot be used simultaneously.
# Press esc to stop.
# See https://github.com/jdibenes/hl2ss/tree/main/extensions before setting
# profile_z to hl2ss.DepthProfile.ZDEPTH (lossless* compression).
#------------------------------------------------------------------------------

from pynput import keyboard

import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.168.1.7"

# Operating mode
# 0: video
# 1: video + rig pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Framerate denominator (must be > 0)
# Effective framerate is framerate / divisor
divisor = 1 

# Depth encoding profile, AB encoding profile and bitrate (None = default)
# SAME: use same compression as AB
#     AB RAW: 
#         - data streamed as-is (most accurate)
#         - very low framerate (uncompressed)
#     AB H264/H265:
#         - reduced depth resolution (from 1mm to 4mm)
#         - noisy due to lossy video compression
#         - full framerate
# ZDEPTH: use ZDepth lossless* compression
#     - increased minimum range (objects close to the camera get truncated)
#     - full framerate
#     - requires building the pyzdepth extension (one time only)
profile_z  = hl2ss.DepthProfile.SAME
profile_ab = hl2ss.VideoProfile.H265_MAIN
bitrate    = None

#------------------------------------------------------------------------------

if (mode == hl2ss.StreamMode.MODE_2):
    data = hl2ss_lnm.download_calibration_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT)
    print('Calibration data')
    print('Image point to unit plane')
    print(data.uv2xy)
    print('Extrinsics')
    print(data.extrinsics)
    print(f'Scale: {data.scale}')
    print(f'Alias: {data.alias}')
    print('Undistort map')
    print(data.undistort_map)
    print('Intrinsics (undistorted only)')
    print(data.intrinsics)
    quit()

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT, mode=mode, divisor=divisor, profile_z=profile_z, profile_ab=profile_ab, bitrate=bitrate)
client.open()

max_depth = 1056
max_uint8 = 255

while (enable):
    data = client.get_next_packet()

    print(f'Frame captured at {data.timestamp}')
    print(f'Sensor Ticks: {data.payload.sensor_ticks}')
    print(f'Pose')
    print(data.pose)

    depth = data.payload.depth
    ab = data.payload.ab
    
    cv2.imshow('Depth', cv2.applyColorMap(((depth / max_depth) * max_uint8).astype(np.uint8), cv2.COLORMAP_JET)) # Scaled for visibility
    cv2.imshow('AB', np.sqrt(ab).astype(np.uint8)) # Scaled for visibility
    cv2.waitKey(1)

client.close()
listener.join()
