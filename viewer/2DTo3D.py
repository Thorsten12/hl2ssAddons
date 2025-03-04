import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider

def Transform(x, y):
    ...
def on_click(event):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        print(f'Geklickt auf Position: ({x:.2f}, {y:.2f})')
        Transform(x, y)

def zeige_bild(index, data, ax, fig):
    ax.clear()
    ax.imshow(data[index])
    ax.set_title(f"Bild {index + 1}/{len(data)}")
    ax.axis("off")
    fig.canvas.draw_idle()

def main():
    ordner1 = "C:/Users/admin/Desktop/hl2ssAddons/hl2ssAddons/viewer/data10/images"
    dateien1 = sorted([f for f in os.listdir(ordner1) if f.endswith(".jpg")])

    if not dateien1:
        print("Keine Bilder gefunden!")
        return
    
    data = [mpimg.imread(os.path.join(ordner1, f)) for f in dateien1]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Zeige das erste Bild
    zeige_bild(0, data, ax, fig)

    # Erstelle einen Slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position des Sliders
    slider = Slider(ax_slider, 'Bild', 0, len(data)-1, valinit=0, valstep=1)

    def update(val):
        zeige_bild(int(slider.val), data, ax, fig)

    slider.on_changed(update)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__ == '__main__':
    main()
