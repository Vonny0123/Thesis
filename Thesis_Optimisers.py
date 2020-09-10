# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:44:53 2020

@author: ewand
"""


import numpy as np
from scipy.optimize import line_search
from numpy.linalg import multi_dot
import time
import scipy

class Block_BFGS_Optimisers():
  def __init__(self, 
               opt,
               fun,
               grad,
               d,
               q,
               maxiter=1000,
               tol=1e-8,
               which_make_symm=2,
               weighted_make_symm=False,
               conditions=False):
    opts = {'b-bfgs': self.Block_BFGS_TakeSteps,
            'o-b-bfgs': self.Orthogonalised_Block_BFGS_TakeSteps,
            'r-b-bfgs': self.Rolling_Block_BFGS_TakeSteps,
            'o-r-b-bfgs': self.Orthogonalised_Rolling_Block_BFGS_TakeSteps,
            's-b-bfgs': self.Sampled_Block_BFGS_TakeSteps,
            'sd': None}
    if opt == 'sd':
      self.optimiser = self.sd_optimiser
    else:
      self.optimiser = self.bfgs_optimiser
    self.opt = opt
    self.Take_Steps = opts[opt]
    self.fun = fun
    self.grad = grad
    self.d = int(d)
    self.q = int(q)
    self.tol = tol
    self.weighted_make_symm = weighted_make_symm
    if which_make_symm == 1:
        self.make_symm = self.make_symm1
    if which_make_symm == 2:
        self.make_symm = self.make_symm2
    self.conditions = conditions
    self.X = np.zeros((d,q))
    self.S = np.zeros((d,q))
    self.G = np.zeros((d,q))
    self.already_failed_once = False
    self.maxiter = maxiter
    self.I = np.eye(d)
    self.done = False
    self.grad_norms = list()
    self.fvals = list()
    self.alphas = list()
    self.conditions = list()
    self.condition_time = 0

  def bArmijo(self, fun, grad, x, p, alpha, beta, maxiter):
    '''
    Implements a Backtracking Armijo linesearch.

    Parameters
    ----------
    fun : Function
        The function to be optimised
    grad : Function
        Returns the value of the gradient of fun
    x : numpy.array
        Current estimate of minimiser
    p : numpy.array
        Search direction for Backtracking Armijo linesearch
    alpha : float
        Value of alpha to start at, will use 1
    beta : float
        Multiplier at each failed iteration
    maxiter : int
        Number of iterations to try before declaring failure

    Returns
    -------
    alpha : float
        Multiplier satisfying the Armijo condition

    '''
    g = grad(x)
    f = fun(x)
    k = 0
    armijo = False
    while k <= maxiter:
      armijo = fun(x + alpha*p) < f + 1e-4 * alpha * np.dot(np.transpose(p), g)
      if not armijo:
        alpha = alpha * beta
        k += 1
      else:
        return alpha
    return alpha
      
  def make_symm1(self, S, Y):
    '''
    First method of finding perturbation in Y resulting in (Y^T)S being symmetric.

    Parameters
    ----------
    S : numpy.array
      Matrix of displacements, as in secant equations
    Y : numpy.array
      Matrix of gradient differences, as in secant equations

    Returns
    -------
    DY : numpy.array
      Satisfies [(Y+DY)^T]S symmetric

    '''
    X = np.dot(np.transpose(Y), S) - np.dot(np.transpose(S), Y)  
    L = -np.tril(X)
    if self.weighted_make_symm:
        LU = scipy.linalg.lu_factor(np.dot(np.transpose(S), Y))
        DY = scipy.linalg.lu_solve(LU, np.transpose(L))
        DY = np.dot(Y, DY)
    else:
        LU = scipy.linalg.lu_factor(np.dot(np.transpose(S), S))
        DY = scipy.linalg.lu_solve(LU, np.transpose(L))
        DY = np.dot(S, DY)
    return DY

  def make_symm2(self, S, Y):
    '''
    Second method of finding perturbation in Y resulting in (Y^T)S being symmetric.

    Parameters
    ----------
    S : numpy.array
      Matrix of displacements, as in secant equations
    Y : numpy.array
      Matrix of gradient differences, as in secant equations

    Returns
    -------
    DY : numpy.array
      Satisfies [(Y+DY)^T]S symmetric

    '''
    DY = np.zeros_like(Y, dtype=float)
    for j in range(1, np.shape(Y)[1]):
      Slj = S[:,:j]
      Sj = S[:,j]
      Ylj = (Y + DY)[:,:j]
      Yj = Y[:,j]
      if self.weighted_make_symm:
          X = np.dot(np.transpose(Slj), Ylj)
          L = scipy.linalg.lu_factor(X)
          Z = np.dot(np.transpose(Ylj), Sj) - np.dot(np.transpose(Slj), Yj)
          Lambda = scipy.linalg.lu_solve(L, Z)
          DY[:,j] = np.dot(Ylj, Lambda)/2
      else:
          X = np.dot(np.transpose(Slj), Slj)
          L = scipy.linalg.lu_factor(X)
          Z = np.dot(np.transpose(Ylj), Sj) - np.dot(np.transpose(Slj), Yj)
          Lambda = scipy.linalg.lu_solve(L, Z)
          DY[:,j] = np.dot(Slj, Lambda)/2
    return DY

  def cholesky(self, A):
    '''
    Implements a modified Cholesky factorisation, giving a list of columns that caused the matrix to be indefinite. These will be deleted from S and Y.

    Parameters
    ----------
    A : numpy.array
      Matrix to be factorised

    Returns
    -------
    L : numpy.array
      Lower triangular matrix, maybe rectangular, if A is positive definite then A = LL^T, else this is only approximate
    bad_cols : list
      List of column indeces that led to A being indefinite, these must be deleted from S and Y.

    '''
    bad_cols = list()
    L = [[0.0] * len(A) for _ in range(len(A))]
    for i, (Ai, Li) in enumerate(zip(A, L)):
      for j, Lj in enumerate(L[:i+1]):
        s = sum(Li[k] * Lj[k] for k in range(j))
        if Ai[i] <= s:
          Li = np.zeros_like(Li)
          bad_cols.append(i)
          break
        elif i == j:
          Li[j] = np.sqrt(Ai[i] - s)
        elif Lj[j] == 0:
          Li[j] = 0
        else:
          Li[j] = (1.0 / Lj[j] * (Ai[j] - s))
    L = np.delete(L, bad_cols, 0)
    L = np.delete(L, bad_cols, 1)                      
    return L, bad_cols

    
  def line_search(self, x, p, g, H, i):
    '''
    Attempts a Wolfe line search with Quasi-Newton search direction, if this fails attempts a Wolfe line search with steepest descent direction. If this fails, backtracking Armijo is used.

    Parameters
    ----------
    x : np.array
      Current estimate of minimiser.
    p : np.array
      Search direction.
    g : np.array
      Gradient at x.
    H : np.array
      Hessian estimate.
    i : int
      Iteration counter, only used to display error.

    Returns
    -------
    alpha : float
      Satisfying one of the conditions, whichever is the first met.
    fval : np.array
      Value of fun at the new iterate.
    gkp1 : np.array
      Value of grad at new iterate.
    H : np.array
      Hessian estimate, may have been reset when linesearch fails.
    p : np.array
      Search direction used, may have been changed to steepest descent direction.

    '''
    alpha, ls_f_evals, ls_g_evals, fval, old_fval, gkp1 = line_search(self.fun, 
                                                                      self.grad,
                                                                      x,
                                                                      p, 
                                                                      g, 
                                                                      maxiter=1000)

    if alpha is None:
      H = self.I
      p = -g
      alpha, ls_f_evals, ls_g_evals, fval, old_fval, gkp1 = line_search(self.fun, 
                                                                      self.grad,
                                                                      x,
                                                                      p, 
                                                                      g, 
                                                                      maxiter=1000)
      print(f'Linesearch failed, resetting H and try again, iteration {self.q - i}')
      
    if alpha is None:
      alpha = self.bArmijo(self.fun, self.grad, x, p, 1, 0.1, 100)
      ls_f_evals += 1
      ls_g_evals += 1
      fval = self.fun(x + alpha*p)
      gkp1 = self.grad(x + alpha*p)
      print(f'Linesearch failed again, bArmijo gave alpha = {alpha}')
    return alpha, fval, gkp1, H, p

  def sd_optimiser(self, x0):
    '''
    Implements the steepest descent algorithm.

    Parameters
    ----------
    x0 : np.array
      Initial estimate of minimiser.

    Returns
    -------
    results : dict
      Dictionary of recorded data used to analyse performance.

    '''
    start_optimiser = time.time()
    print('Inverse-Hessian Updates: ', end='', flush=True)
    self.x0 = x0
    x = x0
    g = self.grad(x)
    i = 1
    total_steps = 0
    while i < self.maxiter:
      
      fval = self.fun(x)
      grad_norm = np.linalg.norm(g, ord=2)
      self.grad_norms.append(grad_norm)
      self.fvals.append(fval)
      if self.grad_norms[-1] < self.tol:
        self.done = True
        break
      alpha, fval, gp1, H, p = self.line_search(x, -g, g, self.I, i)
      
      self.alphas.append(alpha)
      x = x - alpha*g
      g = gp1
      if i % 100 == 0:
        print(f'{i}, {self.grad_norms[-1]}, {self.fvals[-1]}', end=' ', flush=True)
                
      i += 1
      total_steps += 1
      
    results = {'Opt': self.opt,
               'Init': np.array(self.x0),
               'Solution': x,
               'Updates': i,
               'Iterations': total_steps,
               'Gradient Norms': np.array(self.grad_norms),
               'Time Elapsed': time.time() - start_optimiser,
               'Dimension': self.d,
               'q': self.q,
               'Step Lengths': np.array(self.alphas),
               'Function Values': np.array(self.fvals)}
    if self.done:
      return results
    else:
      print('Failed to converge in maxiter')
      return results

  
  def bfgs_optimiser(self, x0):
    '''
    Implements the BFGS algorithm.

    Parameters
    ----------
    x0 : np.array
      Initial estimate of minimiser.

    Returns
    -------
    results : dict
      Dictionary of recorded data used to analyse performance.

    '''
    start_optimiser = time.time()
    print('Inverse-Hessian Updates: ', end='', flush=True)
    self.x0 = x0
    x = x0
    H = self.I
    i = 1
    total_steps = 0
    steps = 0
    while i < self.maxiter:
      
      try:
        x, H, steps = self.updater(x, H, iteration=i)
        print(f'{i}, {self.grad_norms[-1]}, {self.fvals[-1]}', end=' ', flush=True)#, {[self.fun(x), self.grad_norms[-1]]}, ', end='', flush=True)
      except Exception as e:
        H = self.I
        print(e)
        print('iteration '+str(i))
        
      if self.conditions:
        start_conditions = time.time()
        condition = np.linalg.cond(H)
        self.condition_time += (time.time() - start_conditions)
        self.conditions.append(condition)
        
      i += 1
      total_steps += steps
      if self.done:
        break
      
    results = {'Opt': self.opt,
               'Init': np.array(self.x0),
               'Solution': x,
               'Updates': i,
               'Iterations': total_steps,
               'Gradient Norms': np.array(self.grad_norms),
               'Time Elapsed': time.time() - start_optimiser,
               'Dimension': self.d,
               'q': self.q,
               'Step Lengths': np.array(self.alphas),
               'Function Values': np.array(self.fvals)}
    if self.conditions:
      results.update({'Condition Number of Inverse Hessian': np.array(self.conditions),
                      'Time Spent on Condition Checks': np.sum(self.condition_times)})
    if self.done:
      return results
    else:
      print('Failed to converge in maxiter')
      return results

  
  def updater(self, x, H, iteration):
    '''
    Updates the inverse-Hessian estimate based on information obtained in Take_Steps, common to all block BFGS functions algorithms.

    Parameters
    ----------
    x : np.array
      Current estimate of minimiser.
    H : np.array
      Inverse-Hessian estimate.
    iteration : int
      Current iteration, gets passed to other functions for error analysis.

    Returns
    -------
    x : np.array
      New estimate of minimiser since Take_Steps was implemented.
    H : np.array
      New inverse-Hessian estimate.
    steps : Number of steps taken in Take_Steps, used for analysing results.
      DESCRIPTION.

    '''
    x, S, Y, steps = self.Take_Steps(x, H, iteration)
    if S is None:
      return x, self.I, steps
    if self.done:
      return x, None, steps
    try:
      DY = self.make_symm(S, Y)
      Y = Y + DY
      L, bad_cols = self.cholesky(np.dot(np.transpose(Y), S))
      print(bad_cols)
      S_del = np.delete(S, bad_cols, 1)
      if np.size(S_del) == 0:
        return x, self.I, steps
      Y_del = np.delete((Y+DY), bad_cols, 1)
      L = (L, True)
      DeltaST = scipy.linalg.cho_solve(L, np.transpose(S_del)) 
      YDeltaST = np.dot(Y_del, DeltaST)
      Z = np.dot(S_del, DeltaST)  
      H = Z + multi_dot([self.I-np.transpose(YDeltaST), H, self.I-YDeltaST])
    except Exception as e:
      print(f'Update {iteration} failed due to: \n {e}. \n Resetting Hessian.')
      return x, self.I, steps
    
    return x, H, steps
  
  def Block_BFGS_TakeSteps(self, x, H, iteration):
    '''
    TakeSteps funtion for Block BFGS method. 

    Parameters
    ----------
    x : np.array
      Current estimate of minimiser.
    H : np.array
      Current estimate of inverse-Hessian.
    iteration : int
      Current iteration.

    Returns
    -------
    x : np.array
      New estimate of minimiser.
    np.array or None
      Matrix of displacements, or None if optimisation is done.
    np.array or None
      Matrix of gradient differences, or None if optimisation is done.
    steps : int
      Number of steps taken in Teak Steps.

    '''
    steps = 0
    S = np.zeros((self.d, self.q))
    Y = np.zeros((self.d, self.q))
    i = self.q-1
    g = self.grad(x)
    while i >= 0:
      
      self.X[:, i] = x
      if self.done:
        break
      self.G[:, i] = g
      p = -np.dot(H,g)

      alpha, fval, gp1, H, p = self.line_search(x, p, g, H, i)
      if alpha is None:
        H = self.I
        continue
      self.alphas.append(alpha)
      grad_norm = np.linalg.norm(gp1, ord=2)
      self.grad_norms.append(grad_norm)
      self.done = grad_norm < self.tol
      self.fvals.append(fval)
      x = x + alpha*p
      g = gp1
      steps += 1
      i -= 1
      
    if not self.done:
      for i in range(self.q):
        S[:, i] = x - self.X[:, i]
        Y[:, i] = gp1 - self.G[:, i]
      return x, S, Y, steps
    else:
      return x, None, None, steps
    
  def Orthogonalised_Block_BFGS_TakeSteps(self, x, H, iteration):
    '''
    TakeSteps funtion for Orthogonalised Block BFGS method. 

    Parameters
    ----------
    x : np.array
      Current estimate of minimiser.
    H : np.array
      Current estimate of inverse-Hessian.
    iteration : int
      Current iteration.

    Returns
    -------
    x : np.array
      New estimate of minimiser.
    np.array or None
      Matrix of displacements, or None if optimisation is done.
    np.array or None
      Matrix of gradient differences, or None if optimisation is done.
    steps : int
      Number of steps taken in Teak Steps.

    '''
    steps = 0
    S = np.zeros((self.d, self.q))
    Y = np.zeros((self.d, self.q))
    i = self.q-1
    g = self.grad(x)
    while i >=0:
      steps += 1
      p = -np.dot(H,g)
      alpha, fval, gp1, H, p = self.line_search(x, p, g, H, i)
      self.alphas.append(alpha)
      grad_norm = np.linalg.norm(gp1, ord=2)
      self.grad_norms.append(grad_norm)
      self.done = grad_norm < self.tol
      self.fvals.append(fval)
      x = x + alpha*p
      S[:, i] = alpha*p
      g = gp1
      if self.done:
        print('Done')
        break
      i -= 1
      
    if not self.done:
      S = scipy.linalg.orth(S)*np.linalg.norm(S, ord='fro')/self.q
      ncols = np.shape(S)[1]
      for i in range(ncols):
        S[:, i] = -np.sign(np.dot(g, S[:, i]))*S[:, i]
      Y = np.transpose([self.grad(x + S[:, i]) - g for i in range(ncols)])
      #with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
      #  xs = [x + S[:, i] for i in range(np.shape(S)[1])]
      #  results = executor.map(self.grad, xs)
      #Y = np.transpose([result - g for result in results])
      return x, S, Y, steps
    else: return x, None, None, steps

  def Rolling_Block_BFGS_TakeSteps(self, x, H, iteration):
    '''
    TakeSteps funtion for Rolling Block BFGS method. 

    Parameters
    ----------
    x : np.array
      Current estimate of minimiser.
    H : np.array
      Current estimate of inverse-Hessian.
    iteration : int
      Current iteration.

    Returns
    -------
    x : np.array
      New estimate of minimiser.
    np.array or None
      Matrix of displacements, or None if optimisation is done.
    np.array or None
      Matrix of gradient differences, or None if optimisation is done.
    steps : int
      Number of steps taken in Teak Steps.

    '''
    steps = 0
    ncols = np.min([self.q, iteration])

    S = np.zeros((self.d, ncols))
    Y = np.zeros((self.d, ncols))
    g = self.grad(x)
    steps += 1
    
    p = -np.dot(H,g)
    alpha, fval, gp1, H, p = self.line_search(x, p, g, H, self.q)
    self.alphas.append(alpha)
    if alpha is None:
      return x, None, None, steps
    
    self.X[:, 1:] = self.X[:, :-1]
    self.X[:, 0] = x
    self.G[:, 1:] = self.G[:, :-1]
    self.G[:, 0] = g
    grad_norm = np.linalg.norm(gp1, ord=2)
    self.grad_norms.append(grad_norm)
    self.done = grad_norm < self.tol
    self.fvals.append(fval)
    x = x + alpha*p
    g = gp1
    X = self.X[:, :ncols]
    G = self.G[:, :ncols]
    if not self.done:
      for i in range(ncols):
        S[:, i] = x - X[:, i]
        Y[:, i] = gp1 - G[:, i]
      return x, S, Y, steps
    else:
      return x, None, None, steps

  def Orthogonalised_Rolling_Block_BFGS_TakeSteps(self, x, H, iteration):
    '''
    TakeSteps funtion for Orthogonalised Rolling Block BFGS method. 

    Parameters
    ----------
    x : np.array
      Current estimate of minimiser.
    H : np.array
      Current estimate of inverse-Hessian.
    iteration : int
      Current iteration.

    Returns
    -------
    x : np.array
      New estimate of minimiser.
    np.array or None
      Matrix of displacements, or None if optimisation is done.
    np.array or None
      Matrix of gradient differences, or None if optimisation is done.
    steps : int
      Number of steps taken in Teak Steps.

    '''
    steps = 0
    ncols = np.min([self.q, iteration])
    S = np.zeros((self.d, ncols))
    Y = np.zeros((self.d, ncols))
    g = self.grad(x)
    steps += 1
    p = -np.dot(H,g)
    alpha, fval, gp1, H, p = self.line_search(x, p, g, H, self.q)
    self.alphas.append(alpha)
    if alpha is None:
      return x, None, None, steps
    self.S[:, 1:] = self.S[:, :-1]
    self.S[:, 0] = alpha*p
    self.G[:, 1:] = self.G[:, :-1]
    self.G[:, 0] = g
    grad_norm = np.linalg.norm(gp1, ord=2)
    self.grad_norms.append(grad_norm)
    self.done = grad_norm < self.tol
    self.fvals.append(fval)
    x = x + alpha*p
    g = gp1
    S = self.S[:, :ncols]
    if not self.done:
      S = scipy.linalg.orth(S)*np.linalg.norm(S, ord='fro')/ncols
      for i in range(ncols):
        S[:, i] = -np.sign(np.dot(g, S[:, i]))*S[:, i]
      Y = np.transpose([self.grad(x + S[:, i]) - g for i in range(ncols)])
      return x, S, Y, steps
    else:
      return x, None, None, steps

  def Sampled_Block_BFGS_TakeSteps(self, x, H, iteration):
    '''
    TakeSteps funtion for Sampled Block BFGS method. 

    Parameters
    ----------
    x : np.array
      Current estimate of minimiser.
    H : np.array
      Current estimate of inverse-Hessian.
    iteration : int
      Current iteration.

    Returns
    -------
    x : np.array
      New estimate of minimiser.
    np.array or None
      Matrix of displacements, or None if optimisation is done.
    np.array or None
      Matrix of gradient differences, or None if optimisation is done.
    steps : int
      Number of steps taken in Teak Steps.

    '''
    steps = 0
    S = np.random.rand(self.d, self.q)
    S_prev = np.zeros((self.d, self.q))
    Y = np.zeros((self.d, self.q))
    i = self.q-1
    g = self.grad(x)
    while i >= 0:
      steps += 1
      self.X[:, i] = x
      if self.done:
        break
      p = -np.dot(H,g)
      alpha, fval, gp1, H, p = self.line_search(x, p, g, H, i)
      if alpha is None:
        H = self.I
        continue
      self.alphas.append(alpha)
      grad_norm = np.linalg.norm(gp1, ord=2)
      self.grad_norms.append(grad_norm)
      self.done = grad_norm < self.tol
      self.fvals.append(fval)
      x = x + alpha*p
      g = gp1
      if self.done:
        print('Done')
        break
      i -= 1
    ncols = np.min([self.q, iteration])
    if not self.done:
      for i in range(self.q - 1):
        S_prev[:, i] = self.X[:, i+1] - self.X[:, i]
      S_prev[:, ncols] = x - self.X[:, ncols]
      S = scipy.linalg.orth(S)*np.linalg.norm(S_prev, ord='fro')/self.q
      for i in range(ncols):
        S[:, i] = -np.sign(np.dot(g, S[:, i]))*S[:, i]
      Y = np.transpose([self.grad(x + S[:, i]) - g for i in range(ncols)])
      return x, S, Y, steps
    else: 
      return x, None, None, steps
