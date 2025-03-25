#------------------------------------------------------------------------------
# This script adds a cube to the Unity scene and animates it.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import hl2ss
import hl2ss_lnm
import hl2ss_rus

import open3d as o3d

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.137.140'

# Initial position in world space (x, y, z) in meters
position = [0, 0, 0]

# Initial rotation in world space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Initial scale in meters
scale = [0.2, 0.2, 0.2]

# Initial color
rgba = [1, 1, 1, 1]

#------------------------------------------------------------------------------

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

key = 0
file_path = "C:/Users/admin/Desktop/hl2ss/viewer/meshes/spatial_mapping_mesh_1.ply"
ply_mesh = o3d.io.read_triangle_mesh(file_path)
display_list = hl2ss_rus.command_buffer()
display_list.begin_display_list() # Begin command sequence
#display_list.say("programm Startet ")

#display_list.say("trying to Load PLY.data")

display_list.load_ply(ply_mesh)
display_list.end_display_list() # End command sequence
ipc.push(display_list) # Send commands to server
results = ipc.pull(display_list) # Get results from server

print(f'Created cube with id {key}')
while enable:
    

    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list()
    
    display_list.set_world_transform(key, position, rotation, scale)
    display_list.end_display_list()
    ipc.push(display_list)
    results = ipc.pull(display_list)


command_buffer = hl2ss_rus.command_buffer()
command_buffer.remove(key) # Destroy cube
ipc.push(command_buffer)
results = ipc.pull(command_buffer)

ipc.close()

listener.join()
