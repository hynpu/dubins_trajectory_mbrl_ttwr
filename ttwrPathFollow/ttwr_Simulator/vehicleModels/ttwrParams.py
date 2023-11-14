import numpy as np

# map limits
map_x_limit = 40
map_y_limit = 40

# host control limits
maxSteeringAngle = np.pi / 6 # 30 degree
jackKnifeAngle = np.deg2rad(30) # np.pi / 6 

# ttwr system parameters
L1 = 3
L2 = 1.5 #1.5 3
L3 = 2 #2 4
dt = 0.1

# host dimensions
host_length = 4.7
host_width = 1.8

# trailer dimensions
trailer_length = 2.2 # 2.2 5.2
trailer_width = 1.3
trailer_rear_overhang = 1.0 #1.0 5.0
trailer_front_overhang = trailer_length - trailer_rear_overhang

# wheel dimensions
wheel_radius = 0.4

# path resolution
path_res = 0.1

