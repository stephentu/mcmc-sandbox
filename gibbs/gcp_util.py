### first part taken from https://code.google.com/p/haines/source/browse/gcp/wishart.py

# Copyright 2011 Tom SF Haines
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import math
import random
import numpy
import numpy.linalg
import numpy.random
import scipy.special

class Wishart:
  """Simple Wishart distribution class, quite basic really, but has caching to avoid duplicate computation."""
  def __init__(self, dims):
    """dims is the number of dimensions - it initialises with the dof set to 1 and the scale set to the identity matrix. Has copy constructor support."""
    if isinstance(dims, Wishart):
      self.dof = dims.dof
      self.scale = dims.scale.copy()
      self.invScale = dims.invScale.copy() if dims.invScale!=None else None
      self.norm = dims.norm
      self.cholesky = dims.cholesky.copy() if dims.cholesky!=None else None
    else:
      self.dof = 1.0
      self.scale = numpy.identity(dims, dtype=numpy.float32)
      self.invScale = None
      self.norm = None
      self.cholesky = None

  def setDof(self, dof):
    """Sets the degrees of freedom of the distribution."""
    self.dof = dof
    self.norm = None

  def setScale(self, scale):
    """Sets the scale matrix, must be symmetric positive definite"""
    ns = numpy.array(scale, dtype=numpy.float32)
    assert(ns.shape==self.scale.shape)
    self.scale = ns
    self.invScale = None
    self.norm = None
    self.cholesky = None

  def getDof(self):
    """Returns the degrees of freedom."""
    return self.dof

  def getScale(self):
    """Returns the scale matrix."""
    return self.scale

  def getInvScale(self):
    """Returns the inverse of the scale matrix."""
    if self.invScale==None:
      self.invScale = numpy.linalg.inv(self.scale)
    return self.invScale


  def getNorm(self):
    """Returns the normalising constant of the distribution, typically not used by users."""
    if self.norm==None:
      d = self.scale.shape[0]
      self.norm  = math.pow(2.0,-0.5*self.dof*d)
      self.norm *= math.pow(numpy.linalg.det(self.scale),-0.5*self.dof)
      self.norm *= math.pow(math.pi,-0.25*d*(d-1))
      for i in xrange(d):
        self.norm /= scipy.special.gamma(0.5*(n-i))
    return self.norm

  def prob(self, mat):
    """Returns the probability of the provided matrix, which must be the same shape as the scale matrix and also symmetric and positive definite."""
    d = self.scale.shape[0]
    val  = math.pow(numpy.linalg.det(mat),0.5*(n-1-d))
    val *= math.exp(-0.5 * numpy.linalg.trace(numpy.dot(mat,self.getInvScale())))
    return self.getNorm() * val


  def sample(self):
    """Returns a draw from the distribution - will be a symmetric positive definite matrix."""
    if self.cholesky==None:
      self.cholesky = numpy.linalg.cholesky(self.scale)
    d = self.scale.shape[0]
    a = numpy.zeros((d,d),dtype=numpy.float32)
    for r in xrange(d):
      if r!=0: a[r,:r] = numpy.random.normal(size=(r,))
      a[r,r] = math.sqrt(random.gammavariate(0.5*(self.dof-d+1),2.0))
    return numpy.dot(numpy.dot(numpy.dot(self.cholesky,a),a.T),self.cholesky.T)


  def __str__(self):
    return '{dof:%f, scale:%s}'%(self.dof, str(self.scale))

# The rest is our stuff
import numpy as np

def almost_eq(a, b):
    return (np.fabs(a - b) <= 1e-5).all()

def rank(A):
    _, S, _ = np.linalg.svd(A)
    return int(np.sum(S >= 1e-10))

def random_orthogonal_matrix(m, n):
    A, _ = np.linalg.qr(np.random.random((m, n)))
    assert rank(A) == min(m, n)
    return A

def random_orthonormal_matrix(n):
    A = random_orthogonal_matrix(n, n)
    assert almost_eq(np.dot(A.T, A), np.eye(n))
    return A

def sample_niw(mu0, lambda0, psi0, nu0):
    # sample Sigma^{-1} from Wishart(psi0^{-1}, nu0)
    D, = mu0.shape
    assert psi0.shape == (D,D)
    assert lambda0 > 0.0
    assert nu0 > D - 1
    w = Wishart(D)
    w.setDof(nu0)
    w.setScale(np.linalg.inv(psi0))
    covinv = w.sample()
    cov = np.linalg.inv(covinv)
    mu = np.random.multivariate_normal(mean=mu0, cov=1./lambda0 * cov)
    return mu, cov
