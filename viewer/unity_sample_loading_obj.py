# Import the necessary libraries
from pynput import keyboard
import hl2ss
import hl2ss_lnm
import hl2ss_rus
import random
import math
import time
import numpy as np
from PIL import Image  # To handle image textures
import struct  # For packing RGBA values

# Settings --------------------------------------------------------------------
# HoloLens address
host = '192.168.137.179'

# Scale factor (to adjust the size of all elements)
global scale_factor
global scaling_factor
global max_cubes
global enable  # Global variable for enabling/disabling functionality
global ipc  # Global variable for IPC connection
global listener  # Global variable for keyboard listener

scale_factor = 0.001  # Set this to change the size of the cubes
scaling_factor = 0.002  # Factor to scale down the OBJ model (10x smaller)
max_cubes = 500  # Maximum number of cubes

# Initial color (white)
rgba = [1, 1, 1, 1]  # Default RGBA (white with full opacity)

# Animation parameters
enable = True

# Specify the target position (replace with your coordinates)
global target_position
target_position = [0, 0, 0]  # Change these coordinates as needed

model = input("welches model würdest du reinladen wollen:")

def on_press(key):
    global enable

    if key == keyboard.Key.esc:
        cleanup()
        exit(0)  # Gracefully exit the program
    elif key == keyboard.Key.backspace:
        handle_scale_change(new_scale_factor = scale_factor / 2)
    elif key == keyboard.Key.space:
        handle_scale_change(new_scale_factor = scale_factor * 2)
    elif key == keyboard.Key.down:
        handle_scale_change(new_scaling_factor = scaling_factor / 2)
    elif key == keyboard.Key.up:
        handle_scale_change(new_scaling_factor = scaling_factor * 2)
    elif hasattr(key, 'char') and key.char == '-':  # Ensure key.char exists
        handle_scale_change(new_max_cubes=max_cubes / 2)
    elif hasattr(key, 'char') and key.char == '+':  # Ensure key.char exists
        handle_scale_change(new_max_cubes=max_cubes * 2)

    elif hasattr(key, 'char') and key.char == 'w':  # Ensure key.char exists
        handle_position_change(new_x = 0.1)
    elif hasattr(key, 'char') and key.char == 's':  # Ensure key.char exists
        handle_position_change(new_x = -0.1)  
    elif hasattr(key, 'char') and key.char == 'a':  # Ensure key.char exists
        handle_position_change(new_y = 0.1)
    elif hasattr(key, 'char') and key.char == 'd':  # Ensure key.char exists
        handle_position_change(new_y = -0.1) 
    elif hasattr(key, 'char') and key.char == 'q':  # Ensure key.char exists
        handle_position_change(new_z = 0.1)
    elif hasattr(key, 'char') and key.char == 'e':  # Ensure key.char exists
        handle_position_change(new_z = -0.1)     

    return enable


def handle_scale_change(new_scale_factor=None, new_max_cubes=None, new_scaling_factor=None):
    global scale_factor
    global max_cubes
    global scaling_factor
    if new_scale_factor is not None:
        scale_factor = new_scale_factor
    if new_max_cubes is not None:
        max_cubes = new_max_cubes
    if new_scaling_factor is not None:
        scaling_factor = new_scaling_factor
    print("scale_factor:", scale_factor, "scaling_factor:", scaling_factor, "max_cubes:", max_cubes)
    # First, clean up the current connection and stop the program
    stop()

    # Restart with new parameters
    start()

def handle_position_change(new_x=None, new_y=None, new_z = None):
    global target_position
    if new_x is not None:
        target_position[0] += new_x
    if new_y is not None:
        target_position[1] += new_y
    if new_z is not None:
        target_position[2] += new_z    
    print("target Postion:", target_position)
    # First, clean up the current connection and stop the program
    stop()

    # Restart with new parameters
    start()


def load_obj(filename):
    """Load OBJ file and return vertices and texture coordinates."""
    vertices = []
    texture_coords = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Line starts with 'v' indicates a vertex
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('vt '):  # Line starts with 'vt' indicates a texture coordinate
                parts = line.split()
                texture_coords.append([float(parts[1]), float(parts[2])])
    return np.array(vertices), np.array(texture_coords)


def load_texture(filename):
    """Load an image texture using PIL."""
    # Check if the file is either PNG or JPG (JPEG)
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        return Image.open(filename)
    else:
        raise ValueError("Unsupported texture format. Only PNG and JPG are allowed.")


def get_texture_color(texture, uv_coords):
    """Get the color from the texture based on UV coordinates."""
    width, height = texture.size
    # Convert UV coordinates (range 0-1) to pixel coordinates
    u = int(uv_coords[0] * width) % width
    v = int(uv_coords[1] * height) % height
    # Get the pixel color at the UV coordinates
    color = texture.getpixel((u, v))
    # Normalize the color values (0-255 to 0-1)
    return [c / 255.0 for c in color[:3]]  # Ignore alpha if present


def create_cubes_with_color_texture(vertices, texture_coords, color_texture):
    """Create cubes from vertex positions and apply the color texture."""
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list()
    display_list.remove_all()  # Remove any existing objects
    index = 0

    scipping_factor = 1 if max_cubes > len(vertices) else len(vertices) / max_cubes

    for i, vertex in enumerate(vertices):
        if index < scipping_factor:
            index += 1
        else:
            index = 0
            # Scale down the vertex position by the scaling factor
            scaled_vertex = [coord * scaling_factor for coord in vertex]
            # Place the cube at the target position by adding the target position coordinates
            positioned_vertex = [scaled_vertex[j] + target_position[j] for j in range(3)]

            # Check if the current index is within the bounds of texture_coords
            if i < len(texture_coords):
                # Get the color from the texture using UV coordinates
                uv = texture_coords[i]  # UV coordinates for this vertex
                color = get_texture_color(color_texture, uv)

                # Ensure rgba has 4 elements (RGBA)
                rgba = color + [1] * (4 - len(color))  # Fill missing values with alpha = 1 (full opacity)

                display_list.create_primitive(hl2ss_rus.PrimitiveType.Cube)
                display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast)
                # Set the position of the cube to the adjusted vertex position
                display_list.set_world_transform(0, positioned_vertex, [0, 0, 0, 1], [scale_factor, scale_factor, scale_factor])
                display_list.set_color(0, rgba)  # Apply the texture-based color

                # Safely pack the color and send to HoloLens
                key = 0  # Replace with a real key if needed
                packed_color = struct.pack('<Iffff', key, rgba[0], rgba[1], rgba[2], rgba[3])
                display_list.set_active(0, hl2ss_rus.ActiveState.Active)
                display_list.set_target_mode(hl2ss_rus.TargetMode.UseID)

    display_list.end_display_list()
    return display_list


def cleanup():
    """Remove all cubes and close the IPC connection."""
    global ipc  # Use the global ipc variable
    if ipc:  # Ensure ipc is not None
        command_buffer = hl2ss_rus.command_buffer()
        command_buffer.remove_all()  # Destroy all cubes
        ipc.push(command_buffer)
        ipc.pull(command_buffer)
        ipc.close()  # Close the IPC connection


def start():
    global ipc  # Use the global ipc variable
    global listener  # Use the global listener variable

    # Create the cubes and apply the color texture
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
    ipc.open()

    # Load the OBJ file and the texture
    vertices, texture_coords = load_obj(f'models/{model}/obj.obj')  # Update with the actual path to your OBJ file

    # Use either a PNG or JPG texture (modify this path to use your actual texture file)
    color_texture = load_texture(f'models/{model}/texture.png')  # Can be either .png or .jpg/.jpeg

    display_list = create_cubes_with_color_texture(vertices, texture_coords, color_texture)
    ipc.push(display_list)
    results = ipc.pull(display_list)

    # Keep the program running
    while enable:
        time.sleep(1)  # Optional sleep to keep the program active


def stop():
    """Stop the program and close the listener and ipc."""
    global listener
    if listener:  # Check if listener is active
        listener.stop()

    cleanup()  # Ensure resources are cleaned up

start()

