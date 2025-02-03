from system import System
import numpy as np
import cvxpy as cp
import warnings
import os

class LMI():
  def __init__(self, W, b):
    
    # Declare system to import values
    self.system = System(W, b, [], 0.0, 0.0)
    self.nx = self.system.nx
    self.nu = self.system.nu
    self.nq = self.system.nq
    self.max_torque = self.system.max_torque
    self.A = self.system.A
    self.B = self.system.B
    self.C = self.system.C
    self.Nux = self.system.Nux
    self.Nuw = self.system.Nuw
    self.Nub = self.system.Nub
    self.Nvx = self.system.Nvx
    self.Nvw = self.system.Nvw
    self.Nvb = self.system.Nvb
    self.R = self.system.R
    self.Rw = self.system.Rw
    self.Rb = self.system.Rb
    self.nphi = self.system.nphi
    self.neurons = self.system.neurons
    self.nlayers = self.system.nlayers
    self.wstar = self.system.wstar
    self.ustar = self.system.ustar
    self.bound = self.system.bound
    self.nbigx1 = self.nx + self.neurons[0] * 2
    self.nbigx2 = self.nx + self.neurons[1] * 2
    self.nbigx3 = self.nx + self.neurons[2] * 2
    self.nbigx4 = self.nx + self.neurons[3] * 2

    # Flag variables to determine which kind of LMI has to be solved
    self.old_trigger = True
    self.dynamic = True
    self.optim_finsler = False
    
    # Sign definition of Delta V parameter
    self.m_thres = 1e-6

    # Parameters definition
    self.alpha = cp.Parameter(nonneg=True)
    self.convergence_rate = cp.Parameter(nonneg=True)

    # Auxiliary matrices
    self.Abar = self.A + self.B @ self.Rw
    self.Bbar = -self.B @ self.Nuw @ self.R

    # Function that handles all Variables declarations
    self.init_variables()

    # Function that handles all Constraints declarations
    self.init_constraints()

    # Function that handles final problem definition
    self.create_problem()
  
  # Function that handles all Variables declarations
  def init_variables(self):

    # P matrix for Lyapunov function
    self.P = cp.Variable((self.nx, self.nx), symmetric=True)

    # ETM Variables
    T_val = cp.Variable(self.nphi)
    self.T = cp.diag(T_val)
    self.T1 = self.T[:self.neurons[0], :self.neurons[0]]
    self.T2 = self.T[self.neurons[0]:self.neurons[0]+self.neurons[1], self.neurons[0]:self.neurons[0]+self.neurons[1]]
    self.T3 = self.T[self.neurons[0]+self.neurons[1]:self.neurons[0]+self.neurons[1]+self.neurons[2], self.neurons[0]+self.neurons[1]:self.neurons[0]+self.neurons[1]+self.neurons[2]]
    self.T_sat = cp.reshape(self.T[-1, -1], (self.nu, self.nu))

    self.Z = cp.Variable((self.nphi, self.nx))
    self.Z1 = self.Z[:self.neurons[0], :]
    self.Z2 = self.Z[self.neurons[0]:self.neurons[0] + self.neurons[1], :]
    self.Z3 = self.Z[self.neurons[0] + self.neurons[1]:self.neurons[0] + self.neurons[1] + self.neurons[2], :]
    self.Z_sat = cp.reshape(self.Z[-1, :], (self.nu, self.nx))

    if not self.old_trigger:
      # Finsler multipliers, structured to reduce computational burden and different for each layer
      self.N11 = cp.Variable((self.nx, self.nphi))
      self.N12 = cp.Variable((self.nphi, self.nphi), symmetric=True)
      N13 = cp.Variable(self.nphi)
      self.N13 = cp.diag(N13)
      self.N1 = cp.vstack([self.N11, self.N12, self.N13])

      self.N21 = cp.Variable((self.nx, self.nphi))
      self.N22 = cp.Variable((self.nphi, self.nphi), symmetric=True)
      N23 = cp.Variable(self.nphi)
      self.N23 = cp.diag(N23)
      self.N2 = cp.vstack([self.N21, self.N22, self.N23])
      
      self.N31 = cp.Variable((self.nx, self.nphi))
      self.N32 = cp.Variable((self.nphi, self.nphi), symmetric=True)
      N33 = cp.Variable(self.nphi)
      self.N33 = cp.diag(N33)
      self.N3 = cp.vstack([self.N31, self.N32, self.N33])

      self.N41 = cp.Variable((self.nx, self.nphi))
      self.N42 = cp.Variable((self.nphi, self.nphi), symmetric=True)
      N43 = cp.Variable(self.nphi)
      self.N43 = cp.diag(N43)
      self.N4 = cp.vstack([self.N41, self.N42, self.N43])

      # New ETM matrices
      self.bigX1 = cp.Variable((self.nbigx1, self.nbigx1))
      self.bigX2 = cp.Variable((self.nbigx2, self.nbigx2))
      self.bigX3 = cp.Variable((self.nbigx3, self.nbigx3))
      self.bigX4 = cp.Variable((self.nbigx4, self.nbigx4))
      self.bigX = [self.bigX1, self.bigX2, self.bigX3, self.bigX4]
    
    eps = cp.Variable(self.nx + self.nphi + self.nq)
    self.eps = cp.diag(eps)
    
    # Eta dynamics variables
    if self.dynamic:
      Rho = cp.Variable(self.nphi)
      self.Rho = cp.diag(Rho)
      
    if self.optim_finsler:
      # ETM minimization variables
      self.alphax = cp.Variable(self.nlayers, nonneg=True)
      id1 = np.eye(self.nbigx1)
      id2 = np.eye(self.nbigx2)
      id3 = np.eye(self.nbigx3)
      id4 = np.eye(self.nbigx4)
      self.eyex = [id1, id2, id3, id4]
  
  # Function that handles all Constraints declarations
  def init_constraints(self):

    # Sin non-linearity sector condition
    self.Sinsec = cp.bmat([
      [0.0, -1.0],
      [-1.0, -2.0]
    ])
    # Transformation matrix to go from [x, phi] = [x, sin(x) - x] to [x, psi, phi]
    self.Rsin = cp.bmat([
      [np.array([[1.0, 0.0, 0.0]]), np.zeros((1, self.nphi)), np.zeros((1, self.nq))],
      [np.zeros((self.nq, self.nx)), np.zeros((1, self.nphi)), np.eye(self.nq)]
    ])

    # Delta V matrix formulation with non-linearity sector condition, beign positive definite in -pi, pi it's added as a positive term
    self.M = cp.bmat([
      [self.Abar.T @ self.P @ self.Abar - self.P,     self.Abar.T @ self.P @ self.Bbar,     self.Abar.T @ self.P @ self.C],
      [self.Bbar.T @ self.P @ self.Abar,              self.Bbar.T @ self.P @ self.Bbar,     self.Bbar.T @ self.P @ self.C],
      [self.C.T @ self.P @ self.Abar,                 self.C.T @ self.P @ self.Bbar,        self.C.T @ self.P @ self.C]
    ]) + self.Rsin.T @ self.Sinsec @ self.Rsin

    # ETM constraints

    # Useful declarations for zeros and identities of the correct shapes
    idx = np.eye(self.nx)
    xzero = np.zeros((self.nx, self.neurons[0]))
    xzeros = np.zeros((self.nx, self.nu))

    id = np.eye(self.neurons[0])
    zero = np.zeros((self.neurons[0], self.neurons[0]))
    zerox = np.zeros((self.neurons[0], self.nx))
    zeros = np.zeros((self.neurons[0], self.nu))

    ids = np.eye(self.nu)
    szerox = np.zeros((self.nu, self.nx))
    szero = np.zeros((self.nu, self.neurons[0]))
    szeros = np.zeros((self.nu, self.nu))

    # Transformation matrix to go from [x, psi_1, nu_1] to [x, psi_1, psi_2, psi_3, psi_4, nu_1, nu_2, nu_3, nu_4] = [x, psi, nu]
    self.R1 = cp.bmat([
      [idx,   xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [zerox, id,    zero,  zero,  zeros, zero,   zero,  zero,  zeros],
      [zerox, zero,  zero,  zero,  zeros, id,     zero,  zero,  zeros],
    ])

    # Transformation matrix to go from [x, psi_2, nu_2] to [x, psi_1, psi_2, psi_3, psi_4, nu_1, nu_2, nu_3, nu_4] = [x, psi, nu]
    self.R2 = cp.bmat([
      [idx,   xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [zerox, zero,  id,    zero,  zeros,  zero,  zero,  zero,  zeros],
      [zerox, zero,  zero,  zero,  zeros,  zero,  id,    zero,  zeros],
    ])

    # Transformation matrix to go from [x, psi_3, nu_3] to [x, psi_1, psi_2, psi_3, psi_4, nu_1, nu_2, nu_3, nu_4] = [x, psi, nu]
    self.R3 = cp.bmat([
      [idx,   xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [zerox, zero,  zero,  id,    zeros,  zero,  zero,  zero,  zeros],
      [zerox, zero,  zero,  zero,  zeros,  zero,  zero,  id,    zeros],
    ])

    # Transformation matrix to go from [x, psi_4, nu_4] to [x, psi_1, psi_2, psi_3, psi_4, nu_1, nu_2, nu_3, nu_4] = [x, psi, nu]
    self.Rsat = cp.bmat([
      [idx,    xzero, xzero, xzero, xzeros, xzero, xzero, xzero, xzeros],
      [szerox, szero, szero, szero, ids,    szero, szero, szero, szeros],
      [szerox, szero, szero, szero, szeros, szero, szero, szero, ids]
    ])

    # Transformation matrix to go from [x, psi, nu] to [x, psi]
    self.Rnu = cp.bmat([
      [np.eye(self.nx),                np.zeros((self.nx, self.nphi)),   np.zeros((self.nx, self.nq))],
      [np.zeros((self.nphi, self.nx)), np.eye(self.nphi),                np.zeros((self.nphi, self.nq))],
      [self.R @ self.Nvx,              np.eye(self.R.shape[0]) - self.R, np.zeros((self.nphi, self.nq))],
    ])

    # Structure of sector condition for layer 1 to add to finsler constraint
    self.Omega1 = cp.bmat([
      [np.zeros((self.nx, self.nx)),         np.zeros((self.nx, self.neurons[0])),         np.zeros((self.nx, self.neurons[0]))],
      [self.Z1, self.T1, -self.T1],
      [np.zeros((self.neurons[0], self.nx)), np.zeros((self.neurons[0], self.neurons[0])), np.zeros((self.neurons[0], self.neurons[0]))]
    ])
    
    # Structure of sector condition for layer 2 to add to finsler constraint
    self.Omega2 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[1])), np.zeros((self.nx, self.neurons[1]))],
      [self.Z2, self.T2, -self.T2],
      [np.zeros((self.neurons[1], self.nx)), np.zeros((self.neurons[1], self.neurons[1])), np.zeros((self.neurons[1], self.neurons[1]))]
    ])
    
    # Structure of sector condition for layer 3 to add to finsler constraint
    self.Omega3 = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[2])), np.zeros((self.nx, self.neurons[2]))],
      [self.Z3, self.T3, -self.T3],
      [np.zeros((self.neurons[2], self.nx)), np.zeros((self.neurons[2], self.neurons[2])), np.zeros((self.neurons[2], self.neurons[2]))]
    ])

    # Structure of sector condition for last saturation to add to finsler constraint
    self.Omegas = cp.bmat([
      [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.nu)), np.zeros((self.nx, self.nu))],
      [self.Z_sat, self.T_sat, -self.T_sat],
      [np.zeros((self.nu, self.nx)), np.zeros((self.nu, self.nu)), np.zeros((self.nu, self.nu))]
    ])

    # Addition of sector conditions to Delta V matrix
    if self.old_trigger:
      self.M += -self.Rnu.T @ (self.R1.T @ (self.Omega1 + self.Omega1.T) @ self.R1 + self.R2.T @ (self.Omega2 + self.Omega2.T) @ self.R2 + self.R3.T @ (self.Omega3 + self.Omega3.T) @ self.R3 + self.Rsat.T @ (self.Omegas + self.Omegas.T) @ self.Rsat) @ self.Rnu
    else:
      self.M += -self.Rnu.T @ (self.R1.T @ (self.bigX1 + self.bigX1.T) @ self.R1 + self.R2.T @ (self.bigX2 + self.bigX2.T) @ self.R2 + self.R3.T @ (self.bigX3 + self.bigX3.T) @ self.R3 + self.Rsat.T @ (self.bigX4 + self.bigX4.T) @ self.Rsat) @ self.Rnu

    # Definition of Ker([x, psi, nu]) to add in the finsler constraints
    if not self.old_trigger:
      self.hconstr = cp.hstack([self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, -np.eye(self.nphi)])

      # Finsler constraints for each layer
      self.finsler1 = self.R1.T @ (self.bigX1 - self.Omega1 + self.bigX1.T - self.Omega1.T) @ self.R1 + self.N1 @ self.hconstr + self.hconstr.T @ self.N1.T

      self.finsler2 = self.R2.T @ (self.bigX2 - self.Omega2 + self.bigX2.T - self.Omega2.T) @ self.R2 + self.N2 @ self.hconstr + self.hconstr.T @ self.N2.T
      
      self.finsler3 = self.R3.T @ (self.bigX3 - self.Omega3 + self.bigX3.T - self.Omega3.T) @ self.R3 + self.N3 @ self.hconstr + self.hconstr.T @ self.N3.T

      self.finsler4 = self.Rsat.T @ (self.bigX4 - self.Omegas + self.bigX4.T - self.Omegas.T) @ self.Rsat + self.N4 @ self.hconstr + self.hconstr.T @ self.N4.T
   
    # Big M matrix definition with components w.r.t. sqrt(eta)
    if self.dynamic:
      self.rho_eps = cp.Variable(nonneg=True)
      self.id = np.eye(self.nphi)
      self.rho_lmi = 2*(self.Rho - self.convergence_rate * self.id)
    
    # Constraint definition 
    self.constraints = [self.P >> 0]
    self.constraints += [self.T >> 0]
    self.constraints += [self.M << -self.m_thres * np.eye(self.M.shape[0])]
    self.constraints += [self.eps >> 0]
    self.constraints += [self.M + self.eps >> 0]
    if self.dynamic:
      self.constraints += [self.Rho >> 0]
      self.constraints += [self.rho_lmi << 0]
      self.constraints += [self.rho_lmi - self.rho_eps * self.id >> 0]
    if not self.old_trigger:
      self.constraints += [self.finsler1 << 0]
      self.constraints += [self.finsler2 << 0]
      self.constraints += [self.finsler3 << 0]
      self.constraints += [self.finsler4 << 0]

    # Minimization constraints of X_i for each layer
    if self.optim_finsler:
      for i in range(self.nlayers):
        mat = cp.bmat([
          [-self.alphax[i] * self.eyex[i], self.bigX[i]],
          [self.bigX[i].T, -self.eyex[i]]
        ])
        self.constraints += [mat << 0]
    
    # Ellipsoid conditions for activation functions
    for i in range(self.nlayers - 1):
      for k in range(self.neurons[i]):
        Z_el = self.Z[i*self.neurons[i] + k]
        T_el = self.T[i*self.neurons[i] + k, i*self.neurons[i] + k]
        vcap = np.min([np.abs(-self.bound - self.wstar[i][k][0]), np.abs(self.bound - self.wstar[i][k][0])], axis=0)
        ellip = cp.bmat([
            [self.P, cp.reshape(Z_el, (self.nx ,1))],
            [cp.reshape(Z_el, (1, self.nx)), cp.reshape(2*self.alpha*T_el - self.alpha**2*vcap**(-2), (1, 1))] 
        ])
        self.constraints += [ellip >> 0]
    
    # Ellipsoid conditions for last saturation
    Z_el = self.Z_sat
    T_el = self.T_sat
    vcap = np.min([np.abs(-self.bound - self.ustar), np.abs(self.bound - self.ustar)], axis=0)
    ellip = cp.bmat([
        [self.P, cp.reshape(Z_el, (self.nx ,1))],
        [cp.reshape(Z_el, (1, self.nx)), cp.reshape(2*self.alpha*T_el - self.alpha**2*vcap**(-2), (1, 1))] 
    ])
    self.constraints += [ellip >> 0]

  # Function that handles final problem definition
  def create_problem(self):

    # Objective function defined as the sum of the trace of P, eps and the sum of all alphax variables
    if self.optim_finsler:
      obj = cp.trace(self.P) + cp.trace(self.eps)
      for i in range(self.nlayers):
        obj += self.alphax[i]
    else:
      if self.old_trigger and not self.dynamic:
        obj = cp.trace(self.P)
      else:
        obj = cp.trace(self.P) + cp.trace(self.eps)
    if self.dynamic:
      obj += self.rho_eps

    self.objective = cp.Minimize(obj)

    # Problem definition
    self.prob = cp.Problem(self.objective, self.constraints)

    # Warnings disabled only for clearness during debug procedures
    # User warnings filter
    warnings.filterwarnings("ignore", category=UserWarning, module='cvxpy')

  # Function that takes parameter values as input and solves the LMI
  def solve(self, alpha_val, convergence_rate, verbose=False): #, search=False):
    # Parameters update
    self.alpha.value = alpha_val
    self.convergence_rate.value = convergence_rate

    try:
      self.prob.solve(solver=cp.MOSEK, verbose=True)
    except cp.error.SolverError:
      return None

    if self.prob.status not in ["optimal", "optimal_inaccurate"]:
      return None
    else:
      if verbose:
        print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(self.P.value))}")
        print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(self.M.value))}") 
        print(f"Size of ROA: {np.pi/np.sqrt(np.linalg.det(self.P.value))}")
        if self.dynamic:
            print(f"Rho value: {self.Rho.value}")
      
      # Returns area of ROA if feasible
      return np.pi/np.sqrt(np.linalg.det(self.P.value))
  
  # Function that searches for the optimal alpha value by performing a golden ratio search until a certain numerical accuracy is reached or the limit of iterations is reached 
  def search_alpha(self, feasible_extreme, infeasible_extreme, threshold, verbose=False):

    golden_ratio = (1 + np.sqrt(5)) / 2
    i = 0
    
    # Loop until the difference between the two extremes is smaller than the threshold or the limit of iterations is reached
    while (feasible_extreme - infeasible_extreme > threshold) and i < 100:

      i += 1
      alpha1 = feasible_extreme - (feasible_extreme - infeasible_extreme) / golden_ratio
      alpha2 = infeasible_extreme + (feasible_extreme - infeasible_extreme) / golden_ratio
      
      # Solve the LMI for the two alpha values
      ROA = self.solve(alpha1, verbose=False) # , search=False)
      if ROA is None:
        val1 = -1
      else:
        val1 = ROA
      
      ROA = self.solve(alpha2, verbose=False) # , search=False)
      if ROA is None:
        val2 = -1
      else:
        val2 = ROA
        
      # Update the feasible and infeasible extremes
      if val1 > val2:
        feasible_extreme = alpha2
      else:
        infeasible_extreme = alpha1
        
      if verbose:
        if val1 > val2:
          ROA = val1
        else:
          ROA = val2
        print(f"\nIteration number: {i}")
        print(f"==================== \nCurrent ROA value: {ROA}")
        print(f"Current alpha value: {feasible_extreme}\n==================== \n")
    return feasible_extreme

  # Function that saves the variables of interest to use in the simulations
  def save_results(self, path_dir: str):
    if not os.path.exists(path_dir):
      os.makedirs(path_dir)
    np.save(f"{path_dir}/P.npy", self.P.value)
    if self.old_trigger:
      np.save(f"{path_dir}/bigX1.npy", self.Omega1.value)
      np.save(f"{path_dir}/bigX2.npy", self.Omega2.value)
      np.save(f"{path_dir}/bigX3.npy", self.Omega3.value)
      np.save(f"{path_dir}/bigX4.npy", self.Omegas.value)
      if self.dynamic:
        np.save(f"{path_dir}/Rho.npy", self.Rho.value)
      else:
        np.save(f"{path_dir}/Rho.npy", np.zeros((self.nphi, self.nphi)))
    else: 
      np.save(f"{path_dir}/Rho.npy", self.Rho.value)
      np.save(f"{path_dir}/bigX1.npy", self.bigX1.value)
      np.save(f"{path_dir}/bigX2.npy", self.bigX2.value)
      np.save(f"{path_dir}/bigX3.npy", self.bigX3.value)
      np.save(f"{path_dir}/bigX4.npy", self.bigX4.value)
      if self.dynamic:
        np.save(f"{path_dir}/Rho.npy", self.Rho.value)
      else:
        np.save(f"{path_dir}/Rho.npy", np.zeros((self.nphi, self.nphi)))
      
# Main loop execution 
if __name__ == "__main__":
  
  # Weights and bias import
  W1_name = os.path.abspath(__file__ + "/../weights/W1.csv")
  W2_name = os.path.abspath(__file__ + "/../weights/W2.csv")
  W3_name = os.path.abspath(__file__ + "/../weights/W3.csv")
  W4_name = os.path.abspath(__file__ + "/../weights/W4.csv")

  b1_name = os.path.abspath(__file__ + "/../weights/b1.csv")
  b2_name = os.path.abspath(__file__ + "/../weights/b2.csv")
  b3_name = os.path.abspath(__file__ + "/../weights/b3.csv")
  b4_name = os.path.abspath(__file__ + "/../weights/b4.csv")
  
  W1 = np.loadtxt(W1_name, delimiter=',')
  W2 = np.loadtxt(W2_name, delimiter=',')
  W3 = np.loadtxt(W3_name, delimiter=',')
  W4 = np.loadtxt(W4_name, delimiter=',')
  W4 = W4.reshape((1, len(W4)))

  W = [W1, W2, W3, W4]
  
  b1 = np.loadtxt(b1_name, delimiter=',')
  b2 = np.loadtxt(b2_name, delimiter=',')
  b3 = np.loadtxt(b3_name, delimiter=',')
  b4 = np.loadtxt(b4_name, delimiter=',')
  
  b = [b1, b2, b3, b4]

  # Lmi object creation
  lmi = LMI(W, b)
  
  # Alpha search 
  # alpha = lmi.search_alpha(1.0, 0.0, 1e-5, 1.0 verbose=True)

  # Good alpha value found in previous simulations
  alpha = np.load('weights/alpha.npy')

  lmi.solve(alpha, 0.8, verbose=True)
