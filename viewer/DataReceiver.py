from pynput import keyboard
import multiprocessing as mp
import numpy as np
import cv2
from Tools import hl2ss_imshow
from Tools import hl2ss
from Tools import hl2ss_lnm
from Tools import hl2ss_utilities
from Tools import hl2ss_mp
from Tools import hl2ss_3dcv
from Tools import hl2ss_sa
import Config
import datetime
from pylsl import StreamInfo, StreamOutlet
import pandas as pd
import time
import json
import base64

# Settings --------------------------------------------------------------------

# HoloLens 2 address
host = Config.HOST

# Camera parameters
pv_width = 760
pv_height = 428
pv_framerate = 30

# EET parameters
eet_fps = 30  # 30, 60, 90

# Marker properties
radius = 5
combined_color = (255, 0, 255)
left_color = (0, 0, 255)
right_color = (255, 0, 0)
thickness = -1

# Buffer length in seconds
buffer_length = 5

# Spatial Mapping manager settings
triangles_per_cubic_meter = 1000
mesh_threads = 2
sphere_center = [0, 0, 0]
sphere_radius = 5
dataframe = []

# ------------------------------------------------------------------------------

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    print(dataframe)
    return enable

if __name__ == '__main__':
    # Keyboard events ---------------------------------------------------------
    print("started")
    enable = True
    index = 0

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("listener Started")
    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start Spatial Mapping data manager --------------------------------------
    volumes = hl2ss.sm_bounding_volume()
    volumes.add_sphere(sphere_center, sphere_radius)
    print("volumes")
    
    # Download observed surfaces
    sm_manager = hl2ss_sa.sm_manager(host, triangles_per_cubic_meter, mesh_threads)
    sm_manager.open()
    sm_manager.set_volumes(volumes)
    sm_manager.get_observed_surfaces()
    print("suracves")
    # Start PV and EET streams ------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO,
                       hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height,
                                       framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.EXTENDED_EYE_TRACKER,
                       hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=eet_fps))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
    producer.initialize(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss.Parameters_SI.SAMPLE_RATE * buffer_length)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.EXTENDED_EYE_TRACKER)
    print("producer")
    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_eet = consumer.create_sink(producer, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, manager, None)
    sink_pv.get_attach_response()
    sink_eet.get_attach_response()

    # Erstellen Sie Listen zum Sammeln der Daten
    gaze_data = []
    image_data = []
    print("yo")

    # Main Loop ---------------------------------------------------------------
    while enable:
        print("loop")
        
        # Download observed surfaces ------------------------------------------
        sm_manager.get_observed_surfaces()

        # Wait for PV frame ---------------------------------------------------
        sink_pv.acquire()
        left_image_point = [np.nan, np.nan]
        right_image_point = [np.nan, np.nan]
        combined_image_point = [np.nan, np.nan]

        # Get PV frame and nearest (in time) EET frame ------------------------
        _, data_pv = sink_pv.get_most_recent_frame()
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            continue

        _, data_eet = sink_eet.get_nearest(data_pv.timestamp)
        if ((data_eet is None) or (not hl2ss.is_valid_pose(data_eet.pose))):
            continue

        image = data_pv.payload.image
        eet = hl2ss.unpack_eet(data_eet.payload)

        # Update PV intrinsics ------------------------------------------------
        pv_intrinsics = hl2ss.create_pv_intrinsics(data_pv.payload.focal_length, data_pv.payload.principal_point)
        pv_extrinsics = np.eye(4, 4, dtype=np.float32)
        pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

        # Compute world to PV image transformation matrix ---------------------
        world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(
            pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)

        # Draw Left Gaze Pointer ----------------------------------------------
        if (eet.left_ray_valid):
            local_left_ray = hl2ss_utilities.si_ray_to_vector(eet.left_ray.origin, eet.left_ray.direction)
            left_ray = hl2ss_utilities.si_ray_transform(local_left_ray, data_eet.pose)
            d = sm_manager.cast_rays(left_ray)
            if (np.isfinite(d)):
                left_point = hl2ss_utilities.si_ray_to_point(left_ray, d)
                left_image_point = hl2ss_3dcv.project(left_point, world_to_image).flatten()

        # Draw Right Gaze Pointer ---------------------------------------------
        if (eet.right_ray_valid):
            local_right_ray = hl2ss_utilities.si_ray_to_vector(eet.right_ray.origin, eet.right_ray.direction)
            right_ray = hl2ss_utilities.si_ray_transform(local_right_ray, data_eet.pose)
            d = sm_manager.cast_rays(right_ray)
            if (np.isfinite(d)):
                right_point = hl2ss_utilities.si_ray_to_point(right_ray, d)
                right_image_point = hl2ss_3dcv.project(right_point, world_to_image).flatten()

        # Draw Combined Gaze Pointer ------------------------------------------
        if (eet.combined_ray_valid):
            local_combined_ray = hl2ss_utilities.si_ray_to_vector(eet.combined_ray.origin, eet.combined_ray.direction)
            combined_ray = hl2ss_utilities.si_ray_transform(local_combined_ray, data_eet.pose)
            d = sm_manager.cast_rays(combined_ray)
            if (np.isfinite(d)):
                combined_point = hl2ss_utilities.si_ray_to_point(combined_ray, d)
                combined_image_point = hl2ss_3dcv.project(combined_point, world_to_image).flatten()

        # Flatten the image array
        image_array_flat = np.array(image).flatten().astype(np.float32)
        image_bytes = image_array_flat.tobytes()

        # Convert bytes to a string (base64 encoding)
        image_string = base64.b64encode(image_bytes).decode('utf-8')

        # Create a list for the data
        sample_data = [
            combined_image_point[0],
            combined_image_point[1],
            left_image_point[0],
            left_image_point[1],
            time.time(),
            index
        ]

        # Ensure all values are real numbers (floats or integers)
        sample_data = [float(x) for x in sample_data]

        # Add the data to the lists
        gaze_data.append(sample_data)
        image_data.append([image_string, str(float(index))])

        index += 1

    # Nach der Hauptschleife, laden Sie die Daten in den LSL
    """
    info_gaze = StreamInfo("EyeGaze", "Gaze", 6, 30, 'float32', 'EyeTracker-SerialNumber')
    info_image = StreamInfo("EyeImage", "Image", 2, 30, 'string', 'EyeTracker-SerialNumber')

    outlet_gaze = StreamOutlet(info_gaze)
    outlet_image = StreamOutlet(info_image)

    for gaze_sample, image_sample in zip(gaze_data, image_data):
        outlet_gaze.push_sample(gaze_sample)
        outlet_image.push_sample(image_sample)
        time.sleep(1/30)  # Simulieren Sie die ursprüngliche Aufnahmerate

    print("Alle Daten wurden in den LabStreamingLayer geladen.")
    """
    df_eet = pd.DataFrame(gaze_data)
    df_image = pd.DataFrame(image_data)
    df_eet.to_csv("gaze_data.csv", index=False)
    df_image.to_csv("image.csv", index=False)

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
