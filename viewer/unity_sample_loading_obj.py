# Import the notwendigen Libraries
from pynput import keyboard
import hl2ss
import hl2ss_lnm
import hl2ss_rus
import random
import math
import time
import numpy as np

# Settings --------------------------------------------------------------------
# HoloLens address
host = '192.168.137.174'

# Scale factor (to adjust the size of all elements)
scale_factor = 0.002  # Set this to change the size of the cubes
scaling_factor = 0.001  # Factor to scale down the OBJ model (10x smaller)
scipping_factor = 50

# Initial color (white)
rgba = [1, 1, 1, 1]

# Animation parameters
enable = True

# Specify the target position (replace with your coordinates)
target_position = [0, 0, 0]  # Change these coordinates as needed

# -----------------------------------------------------------------------------
def on_press(key):
    global enable
    if key == keyboard.Key.esc:
        cleanup()
        enable = False
    elif key == keyboard.Key.backspace:
        cleanup()
    return enable

def load_obj(filename):
    """Load OBJ file and return vertices."""
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Line starts with 'v' indicates a vertex
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def create_cubes_from_vertices(vertices):
    """Create cubes from vertex positions."""
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list()
    display_list.remove_all()  # Remove any existing objects
    index = 0
    for vertex in vertices:
        if index < scipping_factor:
            index += 1
        else:
            index = 0
            # Scale down the vertex position by the scaling factor
            scaled_vertex = [coord * scaling_factor for coord in vertex]
            # Place the cube at the target position by adding the target position coordinates
            positioned_vertex = [scaled_vertex[i] + target_position[i] for i in range(3)]

            display_list.create_primitive(hl2ss_rus.PrimitiveType.Cube)
            display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast)
            # Set the position of the cube to the adjusted vertex position
            display_list.set_world_transform(0, positioned_vertex, [0, 0, 0, 1], [scale_factor, scale_factor, scale_factor])
            display_list.set_color(0, rgba)
            display_list.set_active(0, hl2ss_rus.ActiveState.Active)
            display_list.set_target_mode(hl2ss_rus.TargetMode.UseID)

    display_list.end_display_list()
    return display_list

def cleanup():
    """Remove all cubes and close the IPC connection."""
    command_buffer = hl2ss_rus.command_buffer()
    command_buffer.remove_all()  # Destroy all cubes
    ipc.push(command_buffer)
    ipc.pull(command_buffer)
    ipc.close()  # Close the IPC connection

# ----------------------------------------------------------------------------- 
listener = keyboard.Listener(on_press=on_press)
listener.start()

ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

# Load the OBJ file and create cubes from its vertices
vertices = load_obj('models/model.obj')  # Update with the actual path to your OBJ file
display_list = create_cubes_from_vertices(vertices)
ipc.push(display_list)
results = ipc.pull(display_list)

# Store cube ids (if needed)
cube_keys = results[2::3]  # Get ids of created cubes

# Keep the program running
while enable:
    time.sleep(1)  # Optional sleep to keep the program active

# Cleanup at the end if it hasn't been called already
cleanup()
listener.join()
