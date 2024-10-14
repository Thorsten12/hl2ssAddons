import pyvista as pv

# Lade das .obj-Modell
mesh = pv.read("C:/Users/Tsyri/OneDrive/Desktop/hl2ssAddons/hl2ssAddons/viewer/models/robo/obj.obj")

# Visualisiere das Modell
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show()
