import numpy as np
import torch.nn as nn
import torch
from scipy.linalg import block_diag
from scipy.optimize import fsolve
import warnings

class System():
  def __init__(self, W, b, Omega, ref, gamma):

    # Ignore useless warnings from torch
    warnings.filterwarnings("ignore", category=UserWarning)

    ## ========== SYSTEM DEFINITION ==========

    # State of the system variable 
    self.state = None

    # Constants
    self.g = 9.81
    self.m = 0.15
    self.l = 0.5
    self.mu = 0.05
    self.dt = 0.02
    self.max_torque = 5
    self.max_speed = 8.0
    self.constant_reference = ref
    
    # Dimensions of state, input of the system and non-linearity
    self.nx = 3
    self.nu = 1
    self.nq = 1

    # State matrices in the form x⁺ = A x + B u + C phi * D ref
    self.A = np.array([
        [1,                       self.dt,                                0],
        [self.g*self.dt/self.l,   1-self.mu*self.dt/(self.m*self.l**2),   0],
        [1,                       0,                                      1]
    ])
    self.B = np.array([
        [0],
        [self.dt * self.max_torque/(self.m*self.l**2)],
        [0]
    ])
    self.C = np.array([
      [0],
      [self.g / self.l * self.dt],
      [0]
    ])
    self.D = np.array([
      [0],
      [0],
      [-1]
    ])
    self.E = np.array([
      [1, 0, 0]
    ])

    ## ========== NETWORK PARAMETERS ==========

    # Weights and biases
    self.W = W
    self.b = b

    ## NN-related variables
    self.nlayers = len(self.W) # Considering the final saturation of the input
    self.neurons = []
    for layer in self.b:
      try:
        self.neurons.append(len(layer))
      except:
        self.neurons.append(1)

    # List of layers of the neural network
    self.layers = []
    for i in range(self.nlayers):
      layer = nn.Linear(self.W[i].shape[1], self.W[i].shape[0])
      layer.weight = nn.Parameter(torch.tensor(self.W[i]))
      layer.bias = nn.Parameter(torch.tensor(self.b[i]))
      self.layers.append(layer)

    # Activation function definition to be called upon event
    self.func = nn.Hardtanh()
    
    # Total number of neurons and activation functions
    self.nphi = 0
    for weight in self.W:
      self.nphi += weight.shape[0]

    # Saturation bound
    self.bound = 1

    # NN matrices in the form [u, v] = N [x, w, 1]
    N = block_diag(*self.W)
    self.Nux = np.zeros((self.nu, self.nx))
    self.Nuw = np.concatenate([np.zeros((self.nu, self.nphi - 1)), np.eye(self.nu)], axis=1)
    self.Nub = np.array([[0.0]])
    self.Nvx = N[:, :self.nx]
    self.Nvw = np.concatenate([N[:, self.nx:], np.zeros((self.nphi, self.nu))], axis=1)
    self.Nvb = np.concatenate([b_i.reshape(-1, 1) for b_i in self.b], axis=0)

    self.N = [self.Nux, self.Nuw, self.Nub, self.Nvx, self.Nvw, self.Nvb]

    ## ========== EQUILIBRIUM COMPUTATION ==========
    
    # Useful matrices for LMI and equilibria computation
    self.R = np.linalg.inv(np.eye(*self.Nvw.shape) - self.Nvw)
    self.Rw = self.Nux + self.Nuw @ self.R @ self.Nvx
    self.Rb = self.Nuw @ self.R @ self.Nvb + self.Nub

    # Equilibrium computation with implicit form
    def implicit_function(x):
      x = x.reshape(3, 1)
      I = np.eye(self.A.shape[0])
      K = np.array([[1.0, 0.0, 0.0]])
      to_zero = np.squeeze((-I + self.A + self.B @ self.Rw - self.C @ K) @ x + self.C * np.sin(K @ x) + self.D * self.constant_reference + self.B @ self.Rb)
      return to_zero

    self.xstar = fsolve(implicit_function, np.array([[self.constant_reference], [0.0], [0.0]])).reshape(3,1)

    # Equilibrium of the input value
    self.ustar = self.Rw @ self.xstar + self.Rb

    # Equilibrium state for hidden layers
    wstar = self.R @ self.Nvx @ self.xstar + self.R @ self.Nvb
    wstar_split = np.split(wstar, np.cumsum(self.neurons[:-1]))
    wstar1, wstar2, wstar3, wstar4 = wstar_split
    self.wstar = [wstar1, wstar2, wstar3, wstar4]

    ## ========== ETM PARAMETERS ==========

    # Initial value for ETM states
    self.eta = np.ones(self.nlayers) * 0.0
    self.rho = np.ones(self.nlayers) * gamma

    # Layer state buffer, initialized to arbitrary high value to trigger an event on initialization
    self.last_w = []
    for i, neuron in enumerate(self.neurons):
      self.last_w.append(np.ones((neuron, 1))*1e8)

    # List to store ETM triggering matrices
    self.bigX = Omega
  
  ## ========== MODULES DEFINITION ==========
  
  # Function to compute the forward pass of the neural network taking into account all ETMs
  def forward(self):
    # Empty vector to keep track of events
    e = np.zeros(self.nlayers)

    # Reshape state for input
    x = self.state.reshape(1, self.W[0].shape[1])

    # Empty vector to store the values needed for the dynamic of the dynamic ETM thresholds
    val = np.zeros(self.nlayers)

    # Iteration for each layer
    for l in range(self.nlayers):

      # Particular case for the first layer since the input is the state
      if l == 0:
        input = torch.tensor(x)
      else:
        if omega is None:
        # fake omega for code robustness
          omega = np.zeros((1, self.W[l].shape[1]))
        
        # The input is the output of the previous layer
        input = torch.tensor(omega.reshape(1, self.W[l].shape[1]))
      
      # State propagation
      nu = self.layers[l](input).detach().numpy().reshape(self.W[l].shape[0], 1)

      # Event computation: Psi >= rho * eta
      # Right hand term
      rht = self.rho[l] * self.eta[l]
      # xi = [xtilde, psitilde, nutilde] vector for the LHT computation
      xtilde = self.state.reshape(3,1) - self.xstar.reshape(3, 1)
      psitilde = nu - self.last_w[l]
      nutilde = nu - self.wstar[l]
      xi = np.vstack([xtilde, psitilde, nutilde])
      mat = self.bigX[l]
      lht = (xi.T @ mat @ xi)[0][0]

      # Event trigger if Psi >= rho * eta
      event = lht > rht

      # If an event is triggered, update the hidden layer values with the activation function and store it in the layer's state buffer. Store the lht value as it will be needed by the eta dynamics
      if event:
        omega = self.func(torch.tensor(nu)).detach().numpy()
        # Storing in the layer's state buffer
        self.last_w[l] = omega
        # Flagging the event
        e[l] = 1
        # LHT updated value
        psitilde = nu - omega
        xi = np.vstack([xtilde, psitilde, nutilde])
        lht = (xi.T @ mat @ xi)[0][0]
        # Storing the LHT value for eta dynamics
        val[l] = lht
      
      # If no event is triggered, output the content of the layer's state buffer and store the LHT value for eta dynamics
      else:
        val[l] = lht
        omega = self.last_w[l]
      
    # ETM state (eta) update
    for i in range(self.nlayers):
      self.eta[i] = self.rho[i] * self.eta[i] - val[i]
    
    # Returns the last output of the last layer along with the event flag vector and the stack of the ETM states
    return omega, e, self.eta.tolist()
  
  # Function to compute the state evolution of the system
  def step(self):

    # Input computation
    u, e, eta = self.forward()

    # Non linearity computation
    nonlin = np.sin(self.E @ self.state) - self.E @ self.state

    # State update with system dynamics
    self.state = self.A @ self.state + self.B @ u + self.C * nonlin + self.D * self.constant_reference

    # Retrun the state, the input, the event vector and the ETM states
    return self.state, u, e, eta