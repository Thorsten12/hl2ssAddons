import pyvista as pv
import numpy as np
import pandas as pd
import re
import open3d as o3d

def get_vec(forward, up):
    combined = (np.array(forward) + np.array(up))
    combined = combined / np.linalg.norm(combined)
    return combined

def extract_vector(vector_string):
    values = re.findall(r'-?\d+\.?\d*', vector_string)
    return np.array([float(value) for value in values])

# Read mesh
mesh = pv.read("C:/Users/admin/Desktop/hl2ssAddons/hl2ssAddons/Tim/data2/combined_mesh1.ply")
#mesh = mesh.rotate_x(90)
#mesh = mesh.rotate_z(9)

index = 0
# Data
df = pd.DataFrame(pd.read_csv("C:/Users/admin/Desktop/hl2ssAddons/hl2ssAddons/Tim/data2/data.csv", delimiter=";"))

# Create plotter object and set camera
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="gray", opacity=0.5)

# Add a red sphere at the HoloLens position
hololens_pos = extract_vector(df["Position"].iloc[index])
sphere = pv.Sphere(radius=0.05, center=hololens_pos)
plotter.add_mesh(sphere, color='red')

# Extract forward and up vectors
forward = extract_vector(df["Forward"].iloc[index])
up = extract_vector(df["Up"].iloc[index])

# Calculate gaze direction
gaze_vec = get_vec(forward, up)

# Create a sphere to represent the gaze direction endpoint
gaze_endpoint = hololens_pos + gaze_vec
gaze_sphere = pv.Sphere(radius=0.05, center=gaze_endpoint)
plotter.add_mesh(gaze_sphere, color='green')

# Create a plane oriented according to the gaze direction
gaze_plane_center = hololens_pos + gaze_vec
gaze_plane = pv.Plane(
    center=gaze_plane_center,
    direction=gaze_vec,
    i_size=0.76,
    j_size=0.428
)
plotter.add_mesh(gaze_plane, color="green", opacity=0.7)

# Create plane based on Forward vector (now using gaze_vec)
forward_plane_center = hololens_pos + gaze_vec * 1.5  # Slightly further than gaze
forward_plane = pv.Plane(
    center=forward_plane_center,
    direction=gaze_vec,
    i_size=0.76,
    j_size=0.428,
)
plotter.add_mesh(forward_plane, color="yellow", opacity=0.7)

# Create plane based on CombinedGazeDirection (now using gaze_vec)
combined_direction_plane_center = hololens_pos + gaze_vec * 0.8  # Slightly closer than gaze
combined_direction_plane = pv.Plane(
    center=combined_direction_plane_center,
    direction=gaze_vec,
    i_size=0.76,
    j_size=0.428,
)
plotter.add_mesh(combined_direction_plane, color="blue", opacity=0.7)

# Create plane based on CombinedGazeOrigin (now using gaze_vec)
combined_origin_plane_center = hololens_pos + gaze_vec * 1.2  # Between gaze and forward
combined_origin_plane = pv.Plane(
    center=combined_origin_plane_center,
    direction=gaze_vec,
    i_size=0.76,
    j_size=0.428,
)
plotter.add_mesh(combined_origin_plane, color="orange", opacity=0.7)

# Show plot
plotter.show()
