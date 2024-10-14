import open3d as o3d
import hl2ss
import hl2ss_lnm
import hl2ss_3dcv
import hl2ss_sa
import numpy as np
import pyvista as pv
import struct

from pynput import keyboard
import hl2ss
import hl2ss_lnm
import hl2ss_rus
import time

# Funktion zur Konvertierung von bytearrays zu numpy arrays
def bytearray_to_numpy(byte_array, dtype, element_size):
    count = len(byte_array) // element_size
    return np.frombuffer(byte_array, dtype=dtype, count=count)

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.137.179'

# Maximum triangles per cubic meter
tpcm = 1000

# Data format
vpf = hl2ss.SM_VertexPositionFormat.R32G32B32A32Float
tif = hl2ss.SM_TriangleIndexFormat.R32Uint
vnf = hl2ss.SM_VertexNormalFormat.R32G32B32A32Float

# Include normals
normals = True

# include bounds
bounds = False

# Maximum number of active threads (on the HoloLens) to compute meshes
threads = 2

# Region of 3D space to sample (bounding box)
# All units are in meters
center  = [0.0, 0.0, 0.0] # Position of the box
extents = [8.0, 8.0, 8.0] # Dimensions of the box

position = [0, 0, 0]

# Initial rotation in world space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Initial scale in meters
scale = [0.05, 0.05, 0.05]

# Initial color
rgba = [1, 1, 1, 1]

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

si = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
si.open()

key = 0

display_list = hl2ss_rus.command_buffer()
display_list.begin_display_list() # Begin command sequence
display_list.remove_all() # Remove all objects that were created remotely
display_list.create_primitive(hl2ss_rus.PrimitiveType.Cube) # Create a cube, server will return its id
display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the cube
display_list.set_world_transform(key, position, rotation, scale) # Set the world transform of the cube
display_list.set_color(key, rgba) # Set the color of the cube
display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the cube visible
display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
display_list.end_display_list() # End command sequence
ipc.push(display_list) # Send commands to server
results = ipc.pull(display_list) # Get results from server
key = results[2] # Get the cube id, created by the 3rd command in the list

print(f'Created cube with id {key}')

# Download meshes -------------------------------------------------------------
client = hl2ss_lnm.ipc_sm(host, hl2ss.IPCPort.SPATIAL_MAPPING)

client.open()
client.create_observer()

volumes = hl2ss.sm_bounding_volume()
volumes.add_box(center, extents)
client.set_volumes(volumes)

surface_infos = client.get_observed_surfaces()
tasks = hl2ss.sm_mesh_task()

for surface_info in surface_infos:
    tasks.add_task(surface_info.id, tpcm, vpf, tif, vnf, normals, bounds)

meshes = client.get_meshes(tasks, threads)
client.close()

print(f'Observed {len(surface_infos)} surfaces')

# Display meshes --------------------------------------------------------------

for index, mesh in meshes.items():
    id_hex = surface_infos[index].id.hex()
    timestamp = surface_infos[index].update_time

    if mesh is None:
        print(f'Task {index}: surface id {id_hex} compute mesh failed')
        continue

    mesh.unpack(vpf, tif, vnf)
    for i in mesh.vertex_positions:
        print(i)
    


   # Extrahiere nur die x, y, z Koordinaten (ignoriere den letzten Wert)
points = mesh.vertex_positions[:, :3]
for i in points:
    print(i)

"""# Erstelle einen PyVista Point Cloud Mesh
point_cloud = pv.PolyData(points)

# Erstelle ein Plot-Objekt
plotter = pv.Plotter()

# Füge die Punktwolke hinzu
plotter.add_mesh(point_cloud, point_size=15, render_points_as_spheres=True, color='red')

# Plot anzeigen
plotter.show()
"""

while (enable):
    data = mesh.vertex_positions

    for pos in data:
        position = pos
        # Cube-Position aktualisieren
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list()
        display_list.set_world_transform(key, position, rotation, scale)
        display_list.end_display_list()
        ipc.push(display_list)
        """results = ipc.pull(display_list)"""
        time.sleep(0.1)

command_buffer = hl2ss_rus.command_buffer()
command_buffer.remove(key) # Destroy cube
ipc.push(command_buffer)
results = ipc.pull(command_buffer)

ipc.close()

listener.join()
