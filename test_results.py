from system import System
from auxiliary_code.ellipsoids import ellipsoid_plot_3D
from auxiliary_code.ellipsoids import ellipsoid_plot_2D_projections
import matplotlib.pyplot as plt
import numpy as np
import os

## ======== WEIGHTS AND BIASES IMPORT ========

files = sorted(os.listdir(os.path.abspath(__file__ + "/../weights")))
W = []
b = []
for f in files:
  if f.startswith('W') and f.endswith('.csv'):
    W.append(np.loadtxt(os.path.abspath(__file__ + "/../weights/" + f), delimiter=','))
  elif f.startswith('b') and f.endswith('.csv'):
    b.append(np.loadtxt(os.path.abspath(__file__ + "/../weights/" + f), delimiter=','))

# Weights and biases reshaping
W[-1] = W[-1].reshape((1, len(W[-1])))

## ======== TRIGGERING MATRICES IMPORT ========

path = 'results'
files = sorted(os.listdir(os.path.abspath(__file__ + "/../" + path)))
bigX = []
for f in files:
  if f.startswith('bigX') and f.endswith('.npy'):
    bigX.append(np.load(os.path.abspath(__file__ + "/../" + path + "/" + f)))

## ======== SYSTEM INITIALIZATION ========
# Parameters: Weights, biases, triggering matrices, reference, gamma threshold for R matrix

gamma_threshold = 0.86

s = System(W, b, bigX, 0.0, gamma_threshold)

# P matrix import for lyapunov function
P = np.load(path + '/P.npy')

volume = 4/3*np.pi/np.sqrt(np.linalg.det(P))
print(f"Volume of ellipsoid: {volume:.2f}")

# Maximum disturbance bound on the position theta in degrees
ref_bound = 5 * np.pi / 180

# Flag to decide wether start in a random initial configuration such that the initial state is inside the ellipsoid or not
random_start = False

# Loop to find a random initial state inside the ellipsoid
if random_start:

  # Flag to check if the initial state is inside the ellipsoid
  in_ellip = False

  while not in_ellip:
    # Random initial state and disturbance
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    vtheta = np.random.uniform(-s.max_speed, s.max_speed)
    ref = np.random.uniform(-ref_bound, ref_bound)

    # Initial state definition and system initialization
    x0 = np.array([[theta], [vtheta], [0.0]])
    s = System(W, b, bigX, ref, gamma_threshold)

    # Check if the initial state is inside the ellipsoid, specifically on the border for plotting purposes
    if (x0 - s.xstar).T @ P @ (x0 - s.xstar) <= 1.0 and (x0).T @ P @ (x0) >= 0.9:

      # Initial eta0 computation with respect to the initial state
      eta0 = ((1 - (x0 - s.xstar).T @ P @ (x0 - s.xstar)) / (s.nlayers * 2))[0][0]
      
      # Initial eta0 value update in the system
      s.eta = np.ones(s.nlayers) * eta0
      
      # Initial state update in the system
      s.state = x0

      # Flag variable update to stop the search
      in_ellip = True


# Fixed initial condition, relative to plots in paper
else:
  theta = -10.67 * np.pi / 180
  vtheta = -2.04
  ref = -1.13 * np.pi / 180
  eta0 = 0.001

  x0 = np.array([[theta], [vtheta], [0.0]])
  s = System(W, b, bigX, ref, gamma_threshold)
  s.state = x0
  s.eta = np.ones(s.nlayers) * eta0

print(f"Initial state: theta0 = {theta*180/np.pi:.2f} deg, vtheta0 = {vtheta:.2f} rad/s, constant disturance = {ref*180/np.pi:.2f} deg")
print(f"Initial eta0: {eta0:.2f}")

# Simulation loop
# Empty lists to store the values of the simulation
states = []
inputs = []
events = []
etas = []
lyap = []

# Flag to stop the simulation
stop_run = False

# Counter of the number of steps
nsteps = 0

# Magnitude of the Lyapunov function to stop the simulation
lyap_magnitude = 1e-15
# lyap_magnitude = 1e-400

# Maximum number of steps to stop the simulation
max_steps = 5000

# Simulation loop
while not stop_run:
  # Counter update
  nsteps += 1

  # Step computation
  state, u, e, eta = s.step()

  # Values storage
  states.append(state)
  inputs.append(u)
  events.append(e)
  etas.append(eta)
  lyap_value = (state - s.xstar).T @ P @ (state - s.xstar)
  for i in range(s.nlayers):
    lyap_value += 2 * eta[i]
  lyap.append(lyap_value)
  
  # Stop condition
  if lyap[-1] < lyap_magnitude or nsteps > max_steps:
    stop_run = True

# Data processing

# Initial state added manually to the states list
states = np.insert(states, 0, x0, axis=0)
# Last state removed to have the same size as the other lists
states = np.delete(states, -1, axis=0)
# First state component converted to degrees
states = np.squeeze(np.array(states))
states[:, 0] *= 180 / np.pi
s.xstar[0] *= 180 / np.pi

# Initial input added manually to the inputs list, set to 0
inputs = np.insert(inputs, 0, np.array(0.0), axis=0)
# Last input removed to have the same size as the other lists
inputs = np.delete(inputs, -1, axis=0)
# Inputs multiplied by the maximum torque to have the real value
inputs = np.squeeze(np.array(inputs)) * s.max_torque

events = np.squeeze(np.array(events))
etas = np.squeeze(np.array(etas))
lyap = np.squeeze(np.array(lyap))

# Check of decrement of the Lyapunov function
lyap_diff = np.diff(lyap)
if np.all(lyap_diff <= 1e-25):
  print("Lyapunov function is always decreasing.")
else:
  print("Lyapunov function is not always decreasing.")

# Data visualization
timegrid = np.arange(0, nsteps)

# Triggering percentage computation
trigger = []
for i in range(s.nlayers):
  layer_trigger = np.sum(events[:, i]) / nsteps * 100
  trigger.append(layer_trigger)

for i in range(s.nlayers):
  print(f"Layer {i+1} has been triggered {trigger[i]:.1f}% of times")

# print(f"Lambda: {s.lambda1}")
overall_trigger = sum(trigger[i] * s.neurons[i] for i in range(s.nlayers)) / (s.nphi)
print(f"Overall update rate: {overall_trigger:.1f}%")

# Replace every non event value from 0 to None for ease of plotting
for i, event in enumerate(events):
  for id, ev in enumerate(event):
    if not ev:
      events[i][id] = None

# Control input plot
plot_cut = 200
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axs[0].plot(timegrid[:plot_cut], inputs[:plot_cut], label=r'u')
# Ustar plot
axs[0].plot(timegrid[:plot_cut], np.squeeze(timegrid[:plot_cut] * 0 + s.ustar * s.max_torque), 'r--', label=r'$u_*$')
axs[0].plot(timegrid[:plot_cut], inputs[:plot_cut] * events[:plot_cut, 3], marker='o', markerfacecolor='none', linestyle='None', label='Events')
# axs[0].set_xlabel('Time steps',fontsize=14)
axs[0].set_ylabel(r'Torque (N m)',fontsize=14)
axs[0].legend(fontsize=14, loc='upper right', ncols=3)
axs[0].grid(True)

# Theta plot
axs[1].plot(timegrid[:plot_cut], states[:plot_cut, 0], label=r'$\theta$')
# Theta star plot
axs[1].plot(timegrid[:plot_cut], timegrid[:plot_cut] * 0 + s.xstar[0], 'r--', label=r'$\theta_*$')
axs[1].plot(timegrid[:plot_cut], states[:plot_cut, 0] * events[:plot_cut, 3], marker='o', markerfacecolor='none', linestyle='None', label='Events')
# axs[1].set_xlabel('Time steps',fontsize=14)
axs[1].set_ylabel(r'$\theta$ (deg)',fontsize=14)
axs[1].legend(fontsize=14, loc='upper right', ncols=3)
axs[1].grid(True)

# V plot
axs[2].plot(timegrid[:plot_cut], states[:plot_cut, 1], label=r'$\dot \theta$')
# V star plot
axs[2].plot(timegrid[:plot_cut], timegrid[:plot_cut] * 0 + s.xstar[1], 'r--', label=r'$\dot \theta_*$')
axs[2].plot(timegrid[:plot_cut], states[:plot_cut, 1] * events[:plot_cut, 3], marker='o', markerfacecolor='none', linestyle='None', label='Events')
# axs[2].set_xlabel('Time steps',fontsize=14)
axs[2].set_ylabel(r'$\dot \theta$ (rad/s)',fontsize=14)
axs[2].legend(fontsize=14, loc='lower right', ncols=3)
axs[2].grid(True)

# Integrator state plot
axs[3].plot(timegrid[:plot_cut], states[:plot_cut, 2], label=r'z')
# Integrator state star plot
axs[3].plot(timegrid[:plot_cut], timegrid[:plot_cut] * 0 + s.xstar[2], 'r--', label=r'$z_*$')
axs[3].plot(timegrid[:plot_cut], states[:plot_cut, 2] * events[:plot_cut, 3], marker='o', markerfacecolor='none', linestyle='None', label='Events')
axs[3].set_xlabel('Time steps',fontsize=14)
axs[3].set_ylabel(r'z',fontsize=14)
axs[3].legend(fontsize=14, loc='lower right', ncols=3)
axs[3].grid(True)
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))


# Lyapunov function plot
lyap_cut = 100
lyap_diff = lyap_diff[:lyap_cut]
axs[0].plot(timegrid[1:lyap_cut+1], timegrid[1:lyap_cut+1] * 0 - (lyap_diff - np.max(lyap_diff))/(np.min(lyap_diff) - np.max(lyap_diff)), 'r', label=r'$\Delta V(x, \boldsymbol{\eta})$')
axs[0].plot(timegrid[:lyap_cut], lyap[:lyap_cut], label=r'$V(x, \boldsymbol{\eta})$', markersize = 5)
axs[0].set_xlabel('Time steps', fontsize=14)
axs[0].legend(fontsize=14)
axs[0].grid(True)

# Eta plots
eta_cut = 100
for i in range(s.nlayers):
  axs[1].plot(timegrid[:eta_cut], etas[:eta_cut, i], label=fr'$\eta^{{{i+1}}}$')
axs[1].legend(fontsize=14)
axs[1].set_xlabel('Time steps', fontsize=14)
axs[1].grid(True)
plt.show()


# Event plot
event_cut = 300
colors = ['r', 'g', 'b', 'c']
body = [':', '-.', '--', '-']
heads = ['s', 'd', 'x', 'v']

# Create a figure
fig, ax = plt.subplots(figsize=(11, 4))

plot_events = events[:event_cut]

for i in range(s.nlayers):
  plot_events[:, i] *= i + 1

plot_events = plot_events[::-1]

for i in range(s.nlayers - 1, -1, -1):
  ax.stem(np.arange(event_cut), plot_events[:, i], linefmt=colors[i] + body[i], markerfmt=colors[i] + heads[i], basefmt="", label=f'ETM {i+1}')

# Display the plot
plt.ylim(0, 5)
plt.xlabel('Time steps', fontsize=14)
plt.legend(fontsize=14, loc='upper center', ncol=4)
plt.grid(True)
plt.show()

# Ellipsoid plot

# 3D ROA plot
fig, ax = ellipsoid_plot_3D(P, False, color='yellow', legend=r'ROA approximation $\mathcal{E}(P, x_*)$')

# 2D ROA projections
ellipsoid_plot_2D_projections(P, plane='xy', offset=-8, ax=ax, color='b', legend=r'Projections of $\mathcal{E}(P, x_*)$')
ellipsoid_plot_2D_projections(P, plane='xz', offset=8, ax=ax, color='b', legend=None)
ellipsoid_plot_2D_projections(P, plane='yz', offset=-35, ax=ax, color='b', legend=None)

# 3D evolution of the states 
ax.plot(states[:, 0] - s.xstar[0], states[:, 1] - s.xstar[1], states[:, 2] - s.xstar[2], 'b')

# Initial point plot
ax.plot(states[0, 0] - s.xstar[0], states[0, 1] - s.xstar[1], states[0, 2] - s.xstar[2], marker='o', markersize=5, color='c')

# Equilibrium point plot
ax.plot(0, 0, 0, marker='o', markersize=5, color='r', label='Equilibrium point')


# 2D evolution of the states
ax.plot(states[:, 0] - s.xstar[0], states[:, 1]  - s.xstar[1], -8, 'b')
ax.plot(states[:, 0] - s.xstar[0], 8, states[:, 2]  - s.xstar[2], 'b')
ax.plot(-35, states[:, 1] - s.xstar[1], states[:, 2]  - s.xstar[2], 'b')

# Projected equilibrium point plot
ax.plot(0, 0, -8, marker='o', markersize=5, color='r')
ax.plot(0, 8, 0, marker='o', markersize=5, color='r')
ax.plot(-35, 0, 0, marker='o', markersize=5, color='r')

# Projected initial point plot
ax.plot(states[0, 0] - s.xstar[0], states[0, 1] - s.xstar[1], -8, marker='o', markersize=5, color='c', label='Initial point')
ax.plot(states[0, 0] - s.xstar[0], 8, states[0, 2] - s.xstar[2], marker='o', markersize=5, color='c')
ax.plot(-35, states[0, 1] - s.xstar[1], states[0, 2] - s.xstar[2], marker='o', markersize=5, color='c')

plt.legend(fontsize=14)
plt.show()