import numpy as np
import matplotlib.pyplot as plt

#Geometry
num_points = 51
x_start = 0
x_stop = 1
num_timesteps = 1000
t_start = 0
t_stop = 1 

colo_initial = 30
colo_boundary = 30
colo_interior_pts = int(0.2*(num_points * num_timesteps))

# Set random seed for reproducible results
np.random.seed(1234)

# Get a uniform sample locations for initial condition data
init_t_coord = np.ones((colo_initial,1), dtype=int).ravel() * 0
init_x_coord = np.random.randint(0, num_points-1, colo_initial)

#Plot initial condition locations in red:
plt.scatter(init_t_coord, init_x_coord, marker='X',c="red")

# Get a uniform sample location for left and right spatial boundary data
x0_bound_t_coord = np.random.randint(0, num_timesteps-1, colo_boundary)
x0_bound_x_coord = np.ones((colo_boundary,1), dtype=int).ravel() * 0
xstop_bound_t_coord = x0_bound_t_coord
xstop_bound_x_coord = np.ones((colo_boundary,1), dtype=int).ravel() * (num_points-1)

#Plot boundary condition locations in blue:
plt.scatter(x0_bound_t_coord, x0_bound_x_coord, marker='X',c="blue")
plt.scatter(xstop_bound_t_coord, xstop_bound_x_coord, marker='X',c="blue")

# Get a uniform sample location for interior data (80% of all remaining data)
x_interior_t_coord = np.random.randint(0,num_timesteps-1,colo_interior_pts)
x_interior_x_coord = np.random.randint(0,num_points-1,colo_interior_pts)

#Plot boundary condition locations in green:
plt.scatter(x_interior_t_coord, x_interior_x_coord, marker='x',c="green", alpha=0.1)

U = np.load("./1d-heat.npy")
plt.imshow(U.transpose(), aspect=8, alpha=0.7)