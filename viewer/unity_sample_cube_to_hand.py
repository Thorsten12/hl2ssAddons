#------------------------------------------------------------------------------
# This script adds a cube to the Unity scene and animates it.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import hl2ss
import hl2ss_lnm
import hl2ss_rus

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.137.179'

# Initial position in world space (x, y, z) in meters
position = [0, 0, 0]

# Initial rotation in world space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Initial scale in meters
scale = [0.05, 0.05, 0.05]

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

z = 0
delta = 0.01


# Hauptschleife
while (enable):
    data = si.get_next_packet()
    si_data = hl2ss.unpack_si(data.payload)


    if (si_data.is_valid_hand_left()):
        hand_left = si_data.get_hand_left()
        pose = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.Wrist)
        if abs(pose.orientation[2]) < 0.9:
            position = [pose.position[0], pose.position[1], -pose.position[2]]
            rotation = [-pose.orientation[0], -pose.orientation[1], pose.orientation[2], pose.orientation[3]]
            print(pose.orientation)
        else:
            print("wrist over 90 rotated")
        #print(f'Left wrist pose: Position={pose.position} Orientation={pose.orientation} Radius={pose.radius} Accuracy={pose.accuracy}')
    else:
        print('No left hand data')
        pass

    # Cube-Position aktualisieren
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list()
    display_list.set_world_transform(key, position, rotation, scale)
    display_list.end_display_list()
    ipc.push(display_list)
    """results = ipc.pull(display_list)"""


command_buffer = hl2ss_rus.command_buffer()
command_buffer.remove(key) # Destroy cube
ipc.push(command_buffer)
results = ipc.pull(command_buffer)

ipc.close()

listener.join()
