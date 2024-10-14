import open3d as o3d
import hl2ss
import hl2ss_lnm
import hl2ss_3dcv
import hl2ss_sa
import numpy as np
import pyvista as pv
import struct

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

# Include normals (Änderung: Normals auf False setzen)
normals = True

# include bounds
bounds = False

# Maximum number of active threads (on the HoloLens) to compute meshes
threads = 2

# Region of 3D space to sample (bounding box)
# All units are in meters
center  = [0.0, 0.0, 0.0] # Position of the box
extents = [8.0, 8.0, 8.0] # Dimensions of the box

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

all_vertices = []
all_faces = []
vertex_offset = 0

for index, mesh in meshes.items():
    id_hex = surface_infos[index].id.hex()
    timestamp = surface_infos[index].update_time

    if mesh is None:
        print(f'Task {index}: surface id {id_hex} compute mesh failed')
        continue

    mesh.unpack(vpf, tif, vnf)
    
    vertex_positions = mesh.vertex_positions[:, :3]
    triangle_indices = mesh.triangle_indices
    
    num_vertices = len(vertex_positions)
    
    if num_vertices % 3 != 0:
        print(f"Warnung: Die Anzahl der Vertices ({num_vertices}) ist nicht durch 3 teilbar!")
        continue
    
    triangle_indices = triangle_indices.reshape((-1, 3))
    
    # Füge Vertices zum Gesamtarray hinzu
    all_vertices.append(vertex_positions)
    
    # Passe die Indizes an und füge sie zum Gesamtarray hinzu
    adjusted_indices = triangle_indices + vertex_offset
    all_faces.append(adjusted_indices)
    
    # Aktualisiere den Vertex-Offset für das nächste Mesh
    vertex_offset += num_vertices

# Kombiniere alle Vertices und Faces
combined_vertices = np.vstack(all_vertices)
combined_faces = np.vstack(all_faces)

# Erstelle das PyVista-Format für Faces
pv_faces = np.hstack([[3] + list(face) for face in combined_faces]).astype(int)

# Erstelle das kombinierte PyVista-Mesh
combined_mesh = pv.PolyData(combined_vertices, pv_faces)

# Visualisiere das kombinierte Mesh ohne Normalen
plotter = pv.Plotter()
plotter.add_mesh(combined_mesh)
plotter.show()
