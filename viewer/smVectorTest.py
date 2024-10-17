import open3d as o3d
import hl2ss
import hl2ss_lnm
import hl2ss_3dcv
import hl2ss_sa
import hl2ss_rus
import numpy as np
import cv2
from pynput import keyboard

import time

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.137.174'
enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client_si = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
client_si.open()


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
#------------------------------------------------------------------------------
# Download meshes -------------------------------------------------------------
# See
# https://learn.microsoft.com/en-us/windows/mixed-reality/develop/native/spatial-mapping-in-directx
# for details

client_sm = hl2ss_lnm.ipc_sm(host, hl2ss.IPCPort.SPATIAL_MAPPING)
client_sm.open()
client_sm.create_observer()
volumes = hl2ss.sm_bounding_volume()
volumes.add_box(center, extents)
client_sm.set_volumes(volumes)
surface_infos = client_sm.get_observed_surfaces()
tasks = hl2ss.sm_mesh_task()
for surface_info in surface_infos:
    tasks.add_task(surface_info.id, tpcm, vpf, tif, vnf, normals, bounds)
meshes = client_sm.get_meshes(tasks, threads)
client_sm.close()
print(f'Observed {len(surface_infos)} surfaces')
# Display meshes --------------------------------------------------------------
open3d_meshes = []
for index, mesh in meshes.items():
    id_hex = surface_infos[index].id.hex()
    timestamp = surface_infos[index].update_time
    if (mesh is None):
        print(f'Task {index}: surface id {id_hex} compute mesh failed')
        continue
    mesh.unpack(vpf, tif, vnf)
    # Surface timestamps are given in Windows FILETIME (utc)
    print(f'Task {index}: surface id {id_hex} @ {timestamp} has {mesh.vertex_positions.shape[0]} vertices {mesh.triangle_indices.shape[0]} triangles {mesh.vertex_normals.shape[0]} normals')
    hl2ss_3dcv.sm_mesh_normalize(mesh)
    
    open3d_mesh = hl2ss_sa.sm_mesh_to_open3d_triangle_mesh(mesh)
    open3d_mesh = hl2ss_sa.open3d_triangle_mesh_swap_winding(open3d_mesh)
    open3d_mesh.vertex_colors = open3d_mesh.vertex_normals
    open3d_meshes.append(open3d_mesh)
# Raycasting function
def raycast(scene, meshes, origin, direction, max_distance=10.0, add_sphere=False):
    ray = np.concatenate([origin, direction])
    rays_tensor = o3d.core.Tensor([ray], dtype=o3d.core.Dtype.Float32)
    
    t_hit = scene.cast_rays(rays_tensor)['t_hit'].numpy()[0]
    
    if t_hit != np.inf and t_hit <= max_distance:
        collision_point = origin + direction * t_hit
        if add_sphere:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(collision_point)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
            meshes.append(sphere)
        return collision_point
    return None
# Create RaycastingScene
scene = o3d.t.geometry.RaycastingScene()
# Add meshes to the scene
for mesh in open3d_meshes:
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        # Convert legacy TriangleMesh to Tensor TriangleMesh
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(tmesh)
# Perform raycasting
# Perform raycasting

""" cube at raycast """
# Initial position in world space (x, y, z) in meters
position = [0, 0, 0]
# Initial rotation in world space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]
# Initial scale in meters
scale = [0.15, 0.15, 0.15]
# Initial color
rgba = [1, 1, 1, 1]
ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()
num_cubes = 2
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

while enable:
    data = client_si.get_next_packet()
    si = hl2ss.unpack_si(data.payload)
    head_pose = si.get_head_pose()  

    print(f'Head pose: Position={head_pose.position} Forward={head_pose.forward} Up={head_pose.up}')    
    origin = np.array(head_pose.position)
    direction1 = np.array([0,0,1])
    print(origin, direction1)
    raycast_point = raycast(scene, open3d_meshes, origin, direction1)
    position = [raycast_point[0], raycast_point[1], raycast_point[2]]
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list()
    display_list.set_world_transform(key, position, rotation, scale)
    display_list.end_display_list()
    ipc.push(display_list)
    time.sleep(0.5)
    """results = ipc.pull(display_list)"""
    """
    command_buffer = hl2ss_rus.command_buffer()
    command_buffer.remove(key) # Destroy cube
    ipc.push(command_buffer)
    results = ipc.pull(command_buffer)"""
    
listener.join()
ipc.close()