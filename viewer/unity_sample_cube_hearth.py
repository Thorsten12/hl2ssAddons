import math
import time
import random
import numpy as np
from pynput import keyboard
import hl2ss
import hl2ss_lnm
import hl2ss_rus

# Settings --------------------------------------------------------------------
host = '192.168.137.179'
num_cubes = 50
scale_factor = 0.02
scale = [scale_factor, scale_factor, scale_factor]
rgba = [1, 1, 1, 1]
enable = True
animation_steps = 200
stay_duration = 2.0
heart_scale = 0.5 * scale_factor
fps = 60

# ----------------------------------------------------------------------------- 
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

def heart_shape(t):
    x = heart_scale * 16 * math.sin(t) ** 3
    y = heart_scale * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
    z = 0
    return [x, y, z]

def random_position():
    range_limit = 10 * scale_factor
    return [random.uniform(-range_limit, range_limit) for _ in range(3)]

def ease_in_out_cubic(t):
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

def lerp(start, end, t):
    return [s + (e - s) * t for s, e in zip(start, end)]

def lerp_color(start, end, t):
    return [s + (e - s) * t for s, e in zip(start, end)]

# ----------------------------------------------------------------------------- 
listener = keyboard.Listener(on_press=on_press)
listener.start()

ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

# Create cubes
cube_keys = []
max_attempts = 3
for attempt in range(max_attempts):
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list()
    display_list.remove_all()

    for _ in range(num_cubes):
        display_list.create_primitive(hl2ss_rus.PrimitiveType.Cube)
        display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast)
        display_list.set_world_transform(0, random_position(), [0, 0, 0, 1], scale)
        display_list.set_color(0, rgba)
        display_list.set_active(0, hl2ss_rus.ActiveState.Active)
        display_list.set_target_mode(hl2ss_rus.TargetMode.UseID)

    display_list.end_display_list()
    ipc.push(display_list)
    results = ipc.pull(display_list)

    if len(results) >= num_cubes * 3:
        cube_keys = results[2::3]
        break
    else:
        print(f"Attempt {attempt + 1}: Failed to create all cubes. Retrying...")
        time.sleep(1)

# Check if cubes were created successfully
if len(cube_keys) == 0:
    print("Failed to create cubes after multiple attempts. Exiting.")
    ipc.close()
    listener.join()
    exit()

# Save initial positions for later use
initial_positions = [random_position() for _ in range(num_cubes)]

# Animation loop
while enable:
    # Morph to heart shape
    end_positions = [heart_shape(2 * math.pi * i / num_cubes) for i in range(num_cubes)]

    for step in range(animation_steps):
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()

        t = ease_in_out_cubic(step / animation_steps)

        for i, key in enumerate(cube_keys):
            # Ensure we do not exceed the bounds
            if i < len(initial_positions) and i < len(end_positions):
                new_position = lerp(initial_positions[i], end_positions[i], t)
                display_list.set_target_mode(hl2ss_rus.TargetMode.UseID)
                display_list.set_world_transform(key, new_position, [0, 0, 0, 1], scale)

                # Gradual color change to red while moving to heart
                new_color = lerp_color(rgba, [1, 0, 0, 1], t)
                display_list.set_color(key, new_color)

        display_list.end_display_list()
        ipc.push(display_list)
        ipc.pull(display_list)
        time.sleep(1 / fps)

    # Stay in heart shape
    time.sleep(stay_duration)

    # Morph back to initial positions
    start_positions = end_positions
    end_positions = initial_positions.copy()

    for step in range(animation_steps):
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()

        t = ease_in_out_cubic(step / animation_steps)

        for i, key in enumerate(cube_keys):
            # Ensure we do not exceed the bounds
            if i < len(start_positions) and i < len(end_positions):
                new_position = lerp(start_positions[i], end_positions[i], t)
                display_list.set_target_mode(hl2ss_rus.TargetMode.UseID)
                display_list.set_world_transform(key, new_position, [0, 0, 0, 1], scale)

                # Gradual color change back to white while moving back
                new_color = lerp_color([1, 0, 0, 1], rgba, t)
                display_list.set_color(key, new_color)

        display_list.end_display_list()
        ipc.push(display_list)
        ipc.pull(display_list)
        time.sleep(1 / fps)

    time.sleep(1)  # Optional pause before the next cycle

# Cleanup
command_buffer = hl2ss_rus.command_buffer()
command_buffer.remove_all()
ipc.push(command_buffer)
ipc.pull(command_buffer)

ipc.close()
listener.join()
