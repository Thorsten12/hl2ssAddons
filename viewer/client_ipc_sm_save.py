import open3d as o3d
import hl2ss
import hl2ss_lnm
import hl2ss_3dcv
import hl2ss_sa
import numpy as np
import os

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.137.174'

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

# Combine all meshes into one -------------------------------------------------
combined_mesh = o3d.geometry.TriangleMesh()

for index, mesh in meshes.items():
    id_hex = surface_infos[index].id.hex()
    timestamp = surface_infos[index].update_time

    if mesh is None:
        print(f'Task {index}: surface id {id_hex} compute mesh failed')
        continue

    mesh.unpack(vpf, tif, vnf)

    hl2ss_3dcv.sm_mesh_normalize(mesh)
    
    open3d_mesh = hl2ss_sa.sm_mesh_to_open3d_triangle_mesh(mesh)
    open3d_mesh = hl2ss_sa.open3d_triangle_mesh_swap_winding(open3d_mesh)
    open3d_mesh.vertex_colors = open3d_mesh.vertex_normals

    # Offset the indices of the combined mesh
    triangles = np.asarray(open3d_mesh.triangles) + len(combined_mesh.vertices)
    
    # Append vertices and triangles to the combined mesh
    combined_mesh.vertices.extend(open3d_mesh.vertices)
    combined_mesh.triangles.extend(o3d.utility.Vector3iVector(triangles))
    combined_mesh.vertex_colors.extend(open3d_mesh.vertex_colors)

# Save combined mesh with incremental naming -----------------------------------

# Ensure the output directory exists
output_dir = '3dData'
os.makedirs(output_dir, exist_ok=True)

# Find the next available filename
file_index = 1
filename = os.path.join(output_dir, f'combined_mesh{file_index}.ply')
while os.path.exists(filename):
    file_index += 1
    filename = os.path.join(output_dir, f'combined_mesh{file_index}.ply')

# Save the combined mesh
o3d.io.write_triangle_mesh(filename, combined_mesh)
print(f'Saved combined mesh to {filename}')
