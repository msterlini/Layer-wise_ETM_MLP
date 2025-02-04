from system import System
import numpy as np
import cvxpy as cp
import warnings

class LMI():
  def __init__(self, W, b):
    
    # Declare system to import values
    self.system   = System(W, b, [], 0.0, 0.0)
    self.nx       = self.system.nx
    self.nu       = self.system.nu
    self.nq       = self.system.nq
    self.nphi     = self.system.nphi
    self.neurons  = self.system.neurons
    self.nlayers  = self.system.nlayers
    self.wstar    = self.system.wstar
    self.ustar    = self.system.ustar
    self.bound    = self.system.bound

    # TO DO: PUT IN CONFIGURATION FILE
    # TO DO: PUT IN CONFIGURATION FILE
    # TO DO: PUT IN CONFIGURATION FILE
    # TO DO: PUT IN CONFIGURATION FILE
    # TO DO: PUT IN CONFIGURATION FILE
    # Flag variables to determine which kind of LMI has to be solved
    self.old_trigger = False
    self.optim_finsler = False
    # TO DO: PUT IN CONFIGURATION FILE
    # TO DO: PUT IN CONFIGURATION FILE
    # TO DO: PUT IN CONFIGURATION FILE
    # TO DO: PUT IN CONFIGURATION FILE
    # TO DO: PUT IN CONFIGURATION FILE
    
    # Sign definition of Delta V parameter
    self.m_thres = 1e-6

    # Parameters definition
    self.alpha = cp.Parameter(nonneg=True)

    # Auxiliary matrices
    self.Abar = self.system.Abar
    self.Bbar = self.system.Bbar
    self.C    = self.system.C
    self.R    = self.system.R
    self.Nvx  = self.system.Nvx
    
    # Projection matrices
    self.Pi_nu = self.system.Pi_nu
    self.Pi_s  = self.system.Pi_s

    # Quadratic abstraction for nonlinearity
    self.Phi_abstraction = self.system.Phi_abstraction

    # Projection matrices
    self.projection_matrices = self.system.projection_matrices

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
    T_val         = cp.Variable(self.nphi)
    self.T        = cp.diag(T_val)
    self.T_layers = []
    start = 0
    for i in range(self.nlayers):
      end = start + self.neurons[i]
      self.T_layers.append(self.T[start:end, start:end])
      start = end

    self.Z        = cp.Variable((self.nphi, self.nx))
    self.Z_layers = []
    start = 0
    for i in range(self.nlayers):
      end = start + self.neurons[i]
      self.Z_layers.append(self.Z[start:end, :])
      start = end

    # Finsler multipliers, structured to reduce computational burden and different for each layer
    self.finsler_multipliers = []
    for i in range(self.nlayers):
      N1 = cp.Variable((self.nx, self.nphi))
      N2 = cp.Variable((self.nphi, self.nphi), symmetric=True)
      N3 = cp.diag(cp.Variable(self.nphi))
      N = cp.vstack([N1, N2, N3])
      self.finsler_multipliers.append(N)

    # New ETM matrices
    self.bigX_matrices = []
    for i in range(self.nlayers):
      size = self.nx + 2 * self.neurons[i]
      bigX = cp.Variable((size, size))
      self.bigX_matrices.append(bigX)
    
    # Variable for Sigma
    eps = cp.Variable(self.nx + self.nphi + self.nq)
    self.eps = cp.diag(eps)
    
    # ETM minimization variables
    self.betas = []
    for i in range(self.nlayers):
      beta = cp.Variable(nonneg=True)
      self.betas.append(beta)
  
  # Function that handles all Constraints declarations
  def init_constraints(self):

    # Delta V matrix formulation with non-linearity sector condition, beign positive definite in -pi, pi it's added as a positive term
    self.M = cp.bmat([
      [self.Abar.T],
      [self.Bbar.T],
      [self.C.T]
    ]) @ self.P @ cp.bmat([[self.Abar, self.Bbar, self.C]]) - cp.bmat([
      [self.P,                          np.zeros((self.nx, self.nphi)),   np.zeros((self.nx, self.nq))],
      [np.zeros((self.nphi, self.nx)),  np.zeros((self.nphi, self.nphi)), np.zeros((self.nphi, self.nq))],
      [np.zeros((self.nq, self.nx)),    np.zeros((self.nq, self.nphi)),   np.zeros((self.nq, self.nq))]
    ]) + self.Pi_s.T @ self.Phi_abstraction @ self.Pi_s

    # ETM constraints
    # Structure of sector condition to add to finsler constraint
    self.Omegas = []
    for i in range(self.nlayers):
      Omega = cp.bmat([
        [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[i])), np.zeros((self.nx, self.neurons[i]))],
        [self.Z_layers[i], self.T_layers[i], -self.T_layers[i]],
        [np.zeros((self.neurons[i], self.nx)), np.zeros((self.neurons[i], self.neurons[i])), np.zeros((self.neurons[i], self.neurons[i]))]
      ])
      self.Omegas.append(Omega)

    # Addition of sector conditions to Delta V matrix
    for i in range(self.nlayers):
      if self.old_trigger:
        self.M += -self.Pi_nu.T @ (self.projection_matrices[i].T @ (self.Omegas[i] + self.Omegas[i].T) @ self.projection_matrices[i]) @ self.Pi_nu
      else:
        self.M += -self.Pi_nu.T @ (self.projection_matrices[i].T @ (self.bigX_matrices[i] + self.bigX_matrices[i].T) @ self.projection_matrices[i]) @ self.Pi_nu

    # Definition of Ker([x, psi, nu]) to add in the finsler constraints
    self.hconstr = cp.hstack([self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, -np.eye(self.nphi)])

    # Finsler constraints for each layer
    self.finsler_constraints = []
    for i in range(self.nlayers):
      finsler = self.projection_matrices[i].T @ (self.bigX_matrices[i] - self.Omegas[i] + self.bigX_matrices[i].T - self.Omegas[i].T) @ self.projection_matrices[i] + self.finsler_multipliers[i] @ self.hconstr + self.hconstr.T @ self.finsler_multipliers[i].T
      self.finsler_constraints.append(finsler)
   
    # Constraint definition 
    self.constraints = [self.P >> 0]
    self.constraints += [self.T >> 0]
    self.constraints += [self.M << -self.m_thres * np.eye(self.M.shape[0])]
    self.constraints += [self.eps >> 0]
    self.constraints += [self.M + self.eps >> 0]
    if not self.old_trigger:
      for constraint in self.finsler_constraints:
        self.constraints += [constraint << 0]

    # Minimization constraints of X_i for each layer
    if self.optim_finsler:
      for i in range(self.nlayers):
        id = np.eye(self.nx + 2 * self.neurons[i])
        mat = cp.bmat([
          [-self.betas[i] * id, self.bigX_matrices[i]],
          [self.bigX_matrices[i].T, -id]
        ])
        self.constraints += [mat << 0]
    
    # Ellipsoid conditions for activation functions
    for i in range(self.nlayers):
      for k in range(self.neurons[i]):

        Z_el = self.Z_layers[i][k, :]
        T_el = self.T_layers[i][k, k]

        if i == self.nlayers-1:
          vcap = np.min([np.abs(-self.bound - self.wstar[i][k][0]), np.abs(self.bound - self.wstar[i][k][0])], axis=0)
        else:
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
        obj += self.betas[i]
    else:
      if self.old_trigger:
        obj = cp.trace(self.P)
      else:
        obj = cp.trace(self.P) + cp.trace(self.eps)

    self.objective = cp.Minimize(obj)

    # Problem definition
    self.prob = cp.Problem(self.objective, self.constraints)

    # Warnings disabled only for clearness during debug procedures
    # User warnings filter
    warnings.filterwarnings("ignore", category=UserWarning, module='cvxpy')

  # Function that takes parameter values as input and solves the LMI
  def solve(self, alpha_val, verbose=False): #, search=False):
    # Parameters update
    self.alpha.value = alpha_val

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
        print(f"Size of ROA: {4/3 * np.pi/np.sqrt(np.linalg.det(self.P.value))}")
      
      # Returns area of ROA if feasible
      return 4/3 * np.pi/np.sqrt(np.linalg.det(self.P.value))
  
  # # Function that searches for the optimal alpha value by performing a golden ratio search until a certain numerical accuracy is reached or the limit of iterations is reached 
  # def search_alpha(self, feasible_extreme, infeasible_extreme, threshold, verbose=False):

  #   golden_ratio = (1 + np.sqrt(5)) / 2
  #   i = 0
    
  #   # Loop until the difference between the two extremes is smaller than the threshold or the limit of iterations is reached
  #   while (feasible_extreme - infeasible_extreme > threshold) and i < 100:

  #     i += 1
  #     alpha1 = feasible_extreme - (feasible_extreme - infeasible_extreme) / golden_ratio
  #     alpha2 = infeasible_extreme + (feasible_extreme - infeasible_extreme) / golden_ratio
      
  #     # Solve the LMI for the two alpha values
  #     ROA = self.solve(alpha1, verbose=False) # , search=False)
  #     if ROA is None:
  #       val1 = -1
  #     else:
  #       val1 = ROA
      
  #     ROA = self.solve(alpha2, verbose=False) # , search=False)
  #     if ROA is None:
  #       val2 = -1
  #     else:
  #       val2 = ROA
        
  #     # Update the feasible and infeasible extremes
  #     if val1 > val2:
  #       feasible_extreme = alpha2
  #     else:
  #       infeasible_extreme = alpha1
        
  #     if verbose:
  #       if val1 > val2:
  #         ROA = val1
  #       else:
  #         ROA = val2
  #       print(f"\nIteration number: {i}")
  #       print(f"==================== \nCurrent ROA value: {ROA}")
  #       print(f"Current alpha value: {feasible_extreme}\n==================== \n")
  #   return feasible_extreme

  # Function that saves the variables of interest to use in the simulations
  # def save_results(self, path_dir: str):
  #   if not os.path.exists(path_dir):
  #     os.makedirs(path_dir)
  #   np.save(f"{path_dir}/P.npy", self.P.value)
  #   if self.old_trigger:
  #     np.save(f"{path_dir}/bigX1.npy", self.Omega1.value)
  #     np.save(f"{path_dir}/bigX2.npy", self.Omega2.value)
  #     np.save(f"{path_dir}/bigX3.npy", self.Omega3.value)
  #     np.save(f"{path_dir}/bigX4.npy", self.Omegas.value)
  #   else: 
  #     np.save(f"{path_dir}/Rho.npy", self.Rho.value)
  #     np.save(f"{path_dir}/bigX1.npy", self.bigX1.value)
  #     np.save(f"{path_dir}/bigX2.npy", self.bigX2.value)
  #     np.save(f"{path_dir}/bigX3.npy", self.bigX3.value)
  #     np.save(f"{path_dir}/bigX4.npy", self.bigX4.value)
      
# Main loop execution 
if __name__ == "__main__":
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
  
  # Lmi object creation
  lmi = LMI(W, b)

  alpha = np.load('weights/alpha.npy')

  lmi.solve(alpha, verbose=True)
