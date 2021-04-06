#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:51:35 2017

@author: jiachengzhang
"""

import numpy as np
import sys
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg


def Density_integration_Poisson_uncertainty(Xn,Yn,fluid_mask,grad_x,grad_y,dirichlet_label,dirichlet_value,
    uncertainty_quantification=True, sigma_grad_x=None, sigma_grad_y=None, sigma_dirichlet=None):
  # Evaluate the density field from the gradient fields by solving the Poisson equation.
  # The uncertainty of the density field can be quantified.
  """
  Inputs:
    Xn,Yn: 2d array of mesh grid. 
    fluid_mask: 2d array of binary mask of flow field. Boundary points are considered in the flow (mask should be True)
    grad_x, grad_y: 2d array of gradient field.
    dirichlet_label: 2d array of binary mask indicating the Dirichlet BC locations. Ohterwise Neumann. At least one point should be dirichlet.
    dirichlet_value: 2d array of diriclet BC values.
    uncertainty_quantification: True to perform the uncertainty quantification, False only perform integration.
    sigma_grad_x, sigma_grad_y, sigma_dirichlet: the uncertainty given as std of the input fields. 2d array of fields.
  Returns: 
    Pn: the integrated density field.
    sigma_Pn: the uncertainty (std) of the integrated density field. 
  """

  Ny, Nx = np.shape(Xn)
  dx = Xn[1,1] - Xn[0,0]
  dy = Yn[1,1] - Yn[0,0]
  invdx = 1.0/dx
  invdy = 1.0/dy
  invdx2 = invdx**2
  invdy2 = invdy**2

  fluid_mask_ex = np.zeros((Ny+2,Nx+2)).astype('bool')
  fluid_mask_ex[1:-1,1:-1] = fluid_mask

  grad_x_ex = np.zeros((Ny+2,Nx+2))
  grad_x_ex[1:-1,1:-1] = grad_x
  grad_y_ex = np.zeros((Ny+2,Nx+2))
  grad_y_ex[1:-1,1:-1] = grad_y
  dirichlet_label_ex = np.zeros((Ny+2,Nx+2)).astype('bool')
  dirichlet_label_ex[1:-1,1:-1] = dirichlet_label
  dirichlet_value_ex = np.zeros((Ny+2,Nx+2))
  dirichlet_value_ex[1:-1,1:-1] = dirichlet_value
  
  sigma_grad_x_ex = np.zeros((Ny+2,Nx+2))
  sigma_grad_y_ex = np.zeros((Ny+2,Nx+2))
  sigma_dirichlet_ex = np.zeros((Ny+2,Nx+2))
  if uncertainty_quantification==True:
    sigma_grad_x_ex[1:-1,1:-1] = sigma_grad_x
    sigma_grad_y_ex[1:-1,1:-1] = sigma_grad_y
    sigma_dirichlet_ex[1:-1,1:-1] = sigma_dirichlet

  # Generate the linear operator.
  j,i = np.where(fluid_mask_ex==True)
  Npts = len(j)
  fluid_index = -np.ones(fluid_mask_ex.shape).astype('int')
  fluid_index[j,i] = range(Npts)
  iC = fluid_index[j,i]
  iC_label = dirichlet_label_ex[j,i]
  iE = fluid_index[j,i+1]
  iW = fluid_index[j,i-1]
  iN = fluid_index[j+1,i]
  iS = fluid_index[j-1,i]

  LaplacianOperator = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
  # RHS = np.zeros(Npts)
  # Var_RHS = np.zeros(Npts) # variance

  # Also generate the linear operator which maps from the grad_x, grad_y, dirichlet val to the rhs.
  Map_grad_x = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
  Map_grad_y = scysparse.csr_matrix((Npts,Npts),dtype=np.float)
  Map_dirichlet_val = scysparse.csr_matrix((Npts,Npts),dtype=np.float)

  # First, construct the linear operator and RHS as if all they are all Nuemanan Bc

  # if the east and west nodes are inside domain
  loc = (iE!=-1)*(iW!=-1)
  LaplacianOperator[iC[loc],iC[loc]] += -2.0*invdx2
  LaplacianOperator[iC[loc],iE[loc]] += +1.0*invdx2
  LaplacianOperator[iC[loc],iW[loc]] += +1.0*invdx2
  # RHS[iC[loc]] += (grad_x_ex[j[loc],i[loc]+1] - grad_x_ex[j[loc],i[loc]-1])*(invdx*0.5)
  # Var_RHS[iC[loc]] += (invdx*0.5)**2 * (sigma_grad_x_ex[j[loc],i[loc]+1]**2 + sigma_grad_x_ex[j[loc],i[loc]-1]**2)
  Map_grad_x[iC[loc],iE[loc]] += invdx*0.5
  Map_grad_x[iC[loc],iW[loc]] += -invdx*0.5

  # if the east node is ouside domian
  loc = (iE==-1)
  LaplacianOperator[iC[loc],iC[loc]] += -2.0*invdx2
  LaplacianOperator[iC[loc],iW[loc]] += +2.0*invdx2
  # RHS[iC[loc]] += -(grad_x_ex[j[loc],i[loc]] + grad_x_ex[j[loc],i[loc]-1])*invdx
  # Var_RHS[iC[loc]] += invdx**2 * (sigma_grad_x_ex[j[loc],i[loc]]**2 + sigma_grad_x_ex[j[loc],i[loc]-1]**2)
  Map_grad_x[iC[loc],iC[loc]] += -invdx
  Map_grad_x[iC[loc],iW[loc]] += -invdx

  # if the west node is ouside domian
  loc = (iW==-1)
  LaplacianOperator[iC[loc],iC[loc]] += -2.0*invdx2
  LaplacianOperator[iC[loc],iE[loc]] += +2.0*invdx2
  # RHS[iC[loc]] += (grad_x_ex[j[loc],i[loc]] + grad_x_ex[j[loc],i[loc]+1])*invdx
  # Var_RHS[iC[loc]] += invdx**2 * (sigma_grad_x_ex[j[loc],i[loc]]**2 + sigma_grad_x_ex[j[loc],i[loc]+1]**2)
  Map_grad_x[iC[loc],iC[loc]] += invdx
  Map_grad_x[iC[loc],iE[loc]] += invdx

  # if the north and south nodes are inside domain
  loc = (iN!=-1)*(iS!=-1)
  LaplacianOperator[iC[loc],iC[loc]] += -2.0*invdy2
  LaplacianOperator[iC[loc],iN[loc]] += +1.0*invdy2
  LaplacianOperator[iC[loc],iS[loc]] += +1.0*invdy2
  # RHS[iC[loc]] += (grad_y_ex[j[loc]+1,i[loc]] - grad_y_ex[j[loc]-1,i[loc]])*(invdy*0.5)
  # Var_RHS[iC[loc]] += (invdy*0.5)**2 * (sigma_grad_y_ex[j[loc]+1,i[loc]]**2 + sigma_grad_y_ex[j[loc]-1,i[loc]]**2)
  Map_grad_y[iC[loc],iN[loc]] += invdy*0.5
  Map_grad_y[iC[loc],iS[loc]] += -invdy*0.5

  # if the north node is ouside domian
  loc = (iN==-1)
  LaplacianOperator[iC[loc],iC[loc]] += -2.0*invdy2
  LaplacianOperator[iC[loc],iS[loc]] += +2.0*invdy2
  # RHS[iC[loc]] += -(grad_y_ex[j[loc],i[loc]] + grad_y_ex[j[loc]-1,i[loc]])*invdy
  # Var_RHS[iC[loc]] += invdy**2 * (sigma_grad_y_ex[j[loc],i[loc]]**2 + sigma_grad_y_ex[j[loc]-1,i[loc]]**2)
  Map_grad_y[iC[loc],iC[loc]] += -invdy
  Map_grad_y[iC[loc],iS[loc]] += -invdy

  # if the south node is ouside domian
  loc = (iS==-1)
  LaplacianOperator[iC[loc],iC[loc]] += -2.0*invdy2
  LaplacianOperator[iC[loc],iN[loc]] += +2.0*invdy2
  # RHS[iC[loc]] += (grad_y_ex[j[loc],i[loc]] + grad_y_ex[j[loc]+1,i[loc]])*invdy
  # Var_RHS[iC[loc]] += invdy**2 * (sigma_grad_y_ex[j[loc],i[loc]]**2 + sigma_grad_y_ex[j[loc]+1,i[loc]]**2)
  Map_grad_y[iC[loc],iC[loc]] += invdy
  Map_grad_y[iC[loc],iN[loc]] += invdy

  # Then change the boundary conidtion at locatiosn of Dirichlet.
  loc = (iC_label==True)
  LaplacianOperator[iC[loc],:] = 0.0
  LaplacianOperator[iC[loc],iC[loc]] = 1.0*invdx2
  # RHS[iC[loc]] = dirichlet_value_ex[j[loc],i[loc]] * invdx2
  # Var_RHS[iC[loc]] = sigma_dirichlet_ex[j[loc],i[loc]]**2 * invdx2**2
  Map_grad_x[iC[loc],:] = 0.0
  Map_grad_y[iC[loc],:] = 0.0
  Map_dirichlet_val[iC[loc],iC[loc]] = 1.0*invdx2

  LaplacianOperator.eliminate_zeros()
  Map_grad_x.eliminate_zeros()
  Map_grad_y.eliminate_zeros()
  Map_dirichlet_val.eliminate_zeros()

  # Solve for the field.
  grad_x_vect = grad_x_ex[j,i]
  grad_y_vect = grad_y_ex[j,i]
  dirichlet_val_vect = dirichlet_value_ex[j,i]
  RHS = Map_grad_x.dot(grad_x_vect) + Map_grad_y.dot(grad_y_vect) + Map_dirichlet_val.dot(dirichlet_val_vect)
  # Solve the linear system
  Pn_ex = np.zeros((Ny+2,Nx+2))
  p_vect = splinalg.spsolve(LaplacianOperator, RHS)
  Pn_ex[j,i] = p_vect
  
  # Uncertainty propagation
  sigma_Pn_ex = np.zeros((Ny+2,Nx+2))
  if uncertainty_quantification==True:
    # Propagate to get the covariance matrix for RHS
    Cov_grad_x = scysparse.diags(sigma_grad_x_ex[j,i]**2,shape=(Npts,Npts),format='csr')
    Cov_grad_y = scysparse.diags(sigma_grad_y_ex[j,i]**2,shape=(Npts,Npts),format='csr')
    Cov_dirichlet_val = scysparse.diags(sigma_dirichlet_ex[j,i]**2,shape=(Npts,Npts),format='csr')
    Cov_RHS = Map_grad_x*Cov_grad_x*Map_grad_x.transpose() + Map_grad_y*Cov_grad_y*Map_grad_y.transpose() + \
              Map_dirichlet_val*Cov_dirichlet_val*Map_dirichlet_val.transpose()

    Laplacian_inv = linalg.inv(LaplacianOperator.A)
    Cov_p = np.matmul(np.matmul(Laplacian_inv, Cov_RHS.A), Laplacian_inv.T)
    Var_p_vect = np.diag(Cov_p)

    sigma_Pn_ex[j,i] = Var_p_vect**0.5

  return Pn_ex[1:-1,1:-1], sigma_Pn_ex[1:-1,1:-1]


def Density_integration_WLS_uncertainty(Xn,Yn,fluid_mask,grad_x,grad_y,dirichlet_label,dirichlet_value,
    uncertainty_quantification=True, sigma_grad_x=None, sigma_grad_y=None, sigma_dirichlet=None):
  # Evaluate the density field from the gradient fields by solving the WLS system.
  # The uncertainty of the density field is also quantified and returned.
  """
  Inputs:
    Xn,Yn: 2d array of mesh grid. 
    fluid_mask: 2d array of binary mask of flow field. Boundary points are considered in the flow (mask should be True)
    grad_x, grad_y: 2d array of gradient field.
    dirichlet_label: 2d array of binary mask indicating the Dirichlet BC locations. Ohterwise Neumann. At least one point should be dirichlet.
    dirichlet_value: 2d array of diriclet BC values.
    uncertainty_quantification: True to perform the uncertainty quantification, False only perform integration.
    sigma_grad_x, sigma_grad_y, sigma_dirichlet: the uncertainty given as std of the input fields. 2d array of fields.
  Returns: 
    Pn: the integrated density field.
    sigma_Pn: the uncertainty (std) of the integrated density field. 
  """

  Ny, Nx = np.shape(Xn)
  dx = Xn[1,1] - Xn[0,0]
  dy = Yn[1,1] - Yn[0,0]
  invdx = 1.0/dx
  invdy = 1.0/dy
  invdx2 = invdx**2
  invdy2 = invdy**2

  fluid_mask_ex = np.zeros((Ny+2,Nx+2)).astype('bool')
  fluid_mask_ex[1:-1,1:-1] = fluid_mask

  dirichlet_label_ex = np.zeros((Ny+2,Nx+2)).astype('bool')
  dirichlet_label_ex[1:-1,1:-1] = dirichlet_label
  dirichlet_value_ex = np.zeros((Ny+2,Nx+2))
  dirichlet_value_ex[1:-1,1:-1] = dirichlet_value
  
  grad_x_ex = np.zeros((Ny+2,Nx+2))
  grad_x_ex[1:-1,1:-1] = grad_x
  grad_y_ex = np.zeros((Ny+2,Nx+2))
  grad_y_ex[1:-1,1:-1] = grad_y

  sigma_grad_x_ex = np.zeros((Ny+2,Nx+2))
  sigma_grad_y_ex = np.zeros((Ny+2,Nx+2))
  sigma_dirichlet_ex = np.zeros((Ny+2,Nx+2))
  sigma_grad_x_ex[1:-1,1:-1] = sigma_grad_x
  sigma_grad_y_ex[1:-1,1:-1] = sigma_grad_y
  sigma_dirichlet_ex[1:-1,1:-1] = sigma_dirichlet

  # Generate the index for mapping the pressure and pressure gradients.
  j,i = np.where(fluid_mask_ex==True)
  Npts = len(j)
  fluid_index = -np.ones(fluid_mask_ex.shape).astype('int')
  fluid_index[j,i] = range(Npts)
  
  # Generate the mask for the gradients
  fluid_mask_Gx_ex = np.logical_and(fluid_mask_ex[:,1:],fluid_mask_ex[:,:-1])
  fluid_mask_Gy_ex = np.logical_and(fluid_mask_ex[1:,:],fluid_mask_ex[:-1,:])

  # Generate the linear operator and the mapping matrix for generating the rhs.
  # For Gx
  jx,ix = np.where(fluid_mask_Gx_ex==True)
  Npts_x = len(jx)
  iC = fluid_index[jx,ix]
  iE = fluid_index[jx,ix+1]
  Operator_Gx = scysparse.csr_matrix((Npts_x,Npts),dtype=np.float)
  Map_Gx = scysparse.csr_matrix((Npts_x,Npts),dtype=np.float)
  Operator_Gx[range(Npts_x),iC] += -invdx
  Operator_Gx[range(Npts_x),iE] += invdx
  Map_Gx[range(Npts_x),iC] += 0.5
  Map_Gx[range(Npts_x),iE] += 0.5

  # For Gy
  jy,iy = np.where(fluid_mask_Gy_ex==True)
  Npts_y = len(jy)
  iC = fluid_index[jy,iy]
  iN = fluid_index[jy+1,iy]
  Operator_Gy = scysparse.csr_matrix((Npts_y,Npts),dtype=np.float)
  Map_Gy = scysparse.csr_matrix((Npts_y,Npts),dtype=np.float)
  Operator_Gy[range(Npts_y),iC] += -invdy
  Operator_Gy[range(Npts_y),iN] += invdy
  Map_Gy[range(Npts_y),iC] += 0.5
  Map_Gy[range(Npts_y),iN] += 0.5

  # For Dirichlet BC
  j_d, i_d = np.where(dirichlet_label_ex==True)
  Npts_d = len(j_d)
  iC = fluid_index[j_d,i_d]
  Operator_d = scysparse.csr_matrix((Npts_d,Npts),dtype=np.float)
  Map_d = scysparse.eye(Npts_d,Npts_d,format='csr') * invdx
  Operator_d[range(Npts_d),iC] += 1.0*invdx
  # dirichlet value vector and cov
  dirichlet_vect = dirichlet_value_ex[j_d,i_d]
  dirichlet_sigma_vect = sigma_dirichlet_ex[j_d,i_d]
  cov_dirichlet = scysparse.diags(dirichlet_sigma_vect**2, format='csr')
  # Generate the vector and cov for pgrad.
  pgrad_x_vect = grad_x_ex[j,i]
  pgrad_y_vect = grad_y_ex[j,i]
  pgrad_vect = np.concatenate((pgrad_x_vect,pgrad_y_vect))
  cov_pgrad_x = sigma_grad_x_ex[j,i]**2
  cov_pgrad_y = sigma_grad_y_ex[j,i]**2
  cov_pgrad_vect = np.concatenate((cov_pgrad_x, cov_pgrad_y))
  cov_pgrad = scysparse.diags(cov_pgrad_vect, format='csr')
  
  # Construct the full operator.
  Operator_GLS = scysparse.bmat([[Operator_Gx],[Operator_Gy],[Operator_d]])

  # Construct the full mapping matrics and get the rhs.
  Map_pgrad = scysparse.bmat([[Map_Gx, None],[None, Map_Gy],[scysparse.csr_matrix((Npts_d,Npts),dtype=np.float), None]])
  Map_dirichlet = scysparse.bmat([[scysparse.csr_matrix((Npts_x+Npts_y,Npts_d),dtype=np.float)],[Map_d]])
  rhs = Map_pgrad.dot(pgrad_vect) + Map_dirichlet.dot(dirichlet_vect)

  # Evaluate the covriance matrix for the rhs
  cov_rhs = Map_pgrad * cov_pgrad * Map_pgrad.transpose() + Map_dirichlet * cov_dirichlet * Map_dirichlet.transpose()

  # Solve for the WLS solution
  weights_vect = cov_rhs.diagonal()**(-1)
  weights_matrix = scysparse.diags(weights_vect,format='csr')
  # Operator_WLS = weights_matrix * Operator_GLS
  # rhs_WLS = weights_matrix.dot(rhs)
  sys_LHS = Operator_GLS.transpose() * weights_matrix * Operator_GLS
  sys_rhs = (Operator_GLS.transpose() * weights_matrix).dot(rhs)

  # Get the solution from lsqr
  # p_vect_wls = splinalg.lsqr(Operator_WLS,rhs_WLS)[0]
  # Pn_WLS = np.zeros(fluid_mask_ex.shape)
  # Pn_WLS[j,i] = p_vect_wls

  # Solve for the WLS solution
  p_vect_wls = splinalg.spsolve(sys_LHS, sys_rhs)
  Pn_WLS_ex = np.zeros(fluid_mask_ex.shape)
  Pn_WLS_ex[j,i] = p_vect_wls

  # Perform the uncertainty propagation
  sigma_Pn_ex = np.zeros((Ny+2,Nx+2))
  if uncertainty_quantification == True:
    cov_sys_rhs = (Operator_GLS.transpose() * weights_matrix) * cov_rhs * (Operator_GLS.transpose() * weights_matrix).transpose()
    sys_LHS_inv = linalg.inv(sys_LHS.A)
    Cov_p = np.matmul(np.matmul(sys_LHS_inv, cov_sys_rhs.A), sys_LHS_inv.T)
    Var_p_vect = np.diag(Cov_p)
    sigma_Pn_ex[j,i] = Var_p_vect**0.5

  return Pn_WLS_ex[1:-1,1:-1], sigma_Pn_ex[1:-1,1:-1]
  

def Density_integration_WLS_uncertainty_weighted_average(Xn,Yn,fluid_mask,grad_x,grad_y,dirichlet_label,dirichlet_value,
    uncertainty_quantification=True, sigma_grad_x=None, sigma_grad_y=None, sigma_dirichlet=None):
  # Evaluate the density field from the gradient fields by solving the WLS system.
  # The uncertainty of the density field is also quantified and returned.
  # The gradient interpolation (from grid points to staggered location) is done by a weighted average approach
  # which minimizes the sum of squared bias error and random error.
  """
  Inputs:
    Xn,Yn: 2d array of mesh grid. 
    fluid_mask: 2d array of binary mask of flow field. Boundary points are considered in the flow (mask should be True)
    grad_x, grad_y: 2d array of gradient field.
    dirichlet_label: 2d array of binary mask indicating the Dirichlet BC locations. Ohterwise Neumann. At least one point should be dirichlet.
    dirichlet_value: 2d array of diriclet BC values.
    uncertainty_quantification: True to perform the uncertainty quantification, False only perform integration.
    sigma_grad_x, sigma_grad_y, sigma_dirichlet: the uncertainty given as std of the input fields. 2d array of fields.
  Returns: 
    Pn: the integrated density field.
    sigma_Pn: the uncertainty (std) of the integrated density field. 
  """

  Ny, Nx = np.shape(Xn)
  dx = Xn[1,1] - Xn[0,0]
  dy = Yn[1,1] - Yn[0,0]
  invdx = 1.0/dx
  invdy = 1.0/dy
  invdx2 = invdx**2
  invdy2 = invdy**2

  fluid_mask_ex = np.zeros((Ny+2,Nx+2)).astype('bool')
  fluid_mask_ex[1:-1,1:-1] = fluid_mask

  dirichlet_label_ex = np.zeros((Ny+2,Nx+2)).astype('bool')
  dirichlet_label_ex[1:-1,1:-1] = dirichlet_label
  dirichlet_value_ex = np.zeros((Ny+2,Nx+2))
  dirichlet_value_ex[1:-1,1:-1] = dirichlet_value
  
  grad_x_ex = np.zeros((Ny+2,Nx+2))
  grad_x_ex[1:-1,1:-1] = grad_x
  grad_y_ex = np.zeros((Ny+2,Nx+2))
  grad_y_ex[1:-1,1:-1] = grad_y

  sigma_grad_x_ex = np.zeros((Ny+2,Nx+2))
  sigma_grad_y_ex = np.zeros((Ny+2,Nx+2))
  sigma_dirichlet_ex = np.zeros((Ny+2,Nx+2))
  sigma_grad_x_ex[1:-1,1:-1] = sigma_grad_x
  sigma_grad_y_ex[1:-1,1:-1] = sigma_grad_y
  sigma_dirichlet_ex[1:-1,1:-1] = sigma_dirichlet

  # Generate the index for mapping the pressure and pressure gradients.
  j,i = np.where(fluid_mask_ex==True)
  Npts = len(j)
  fluid_index = -np.ones(fluid_mask_ex.shape).astype('int')
  fluid_index[j,i] = range(Npts)
  
  # Generate the mask for the gradients
  fluid_mask_Gx_ex = np.logical_and(fluid_mask_ex[:,1:],fluid_mask_ex[:,:-1])
  fluid_mask_Gy_ex = np.logical_and(fluid_mask_ex[1:,:],fluid_mask_ex[:-1,:])

  # Generate the linear operator and the mapping matrix for generating the rhs.
  # For Gx
  jx,ix = np.where(fluid_mask_Gx_ex==True)
  Npts_x = len(jx)
  iC = fluid_index[jx,ix]
  iE = fluid_index[jx,ix+1]
  Operator_Gx = scysparse.csr_matrix((Npts_x,Npts),dtype=np.float)
  Operator_Gx[range(Npts_x),iC] += -invdx
  Operator_Gx[range(Npts_x),iE] += invdx
  # The Mapping Gx that maps grided gradients to staggered location is by weighted average.
  Map_Gx = scysparse.csr_matrix((Npts_x,Npts),dtype=np.float)
  V_C = grad_x_ex[jx,ix]
  V_E = grad_x_ex[jx,ix+1]
  sigma_C = sigma_grad_x_ex[jx,ix]
  sigma_E = sigma_grad_x_ex[jx,ix+1]
  weight_C = ((V_C-V_E)**2 + 2*sigma_E**2) / (2*(V_C-V_E)**2 + 2*sigma_C**2 + 2*sigma_E**2)
  weight_E = 1.0 - weight_C
  Map_Gx[range(Npts_x),iC] += weight_C
  Map_Gx[range(Npts_x),iE] += weight_E

  # For Gy
  jy,iy = np.where(fluid_mask_Gy_ex==True)
  Npts_y = len(jy)
  iC = fluid_index[jy,iy]
  iN = fluid_index[jy+1,iy]
  Operator_Gy = scysparse.csr_matrix((Npts_y,Npts),dtype=np.float)
  Operator_Gy[range(Npts_y),iC] += -invdy
  Operator_Gy[range(Npts_y),iN] += invdy
  # The Mapping Gy that maps grided gradients to staggered location is by weighted average.
  Map_Gy = scysparse.csr_matrix((Npts_y,Npts),dtype=np.float)
  V_C = grad_y_ex[jy,iy]
  V_N = grad_y_ex[jy+1,iy]
  sigma_C = sigma_grad_y_ex[jy,iy]
  sigma_N = sigma_grad_y_ex[jy+1,iy]
  weight_C = ((V_C-V_N)**2 + 2*sigma_N**2) / (2*(V_C-V_N)**2 + 2*sigma_C**2 + 2*sigma_N**2)
  weight_N = 1.0 - weight_C
  Map_Gy[range(Npts_y),iC] += weight_C
  Map_Gy[range(Npts_y),iN] += weight_N 

  # For Dirichlet BC
  j_d, i_d = np.where(dirichlet_label_ex==True)
  Npts_d = len(j_d)
  iC = fluid_index[j_d,i_d]
  Operator_d = scysparse.csr_matrix((Npts_d,Npts),dtype=np.float)
  Map_d = scysparse.eye(Npts_d,Npts_d,format='csr') * invdx
  Operator_d[range(Npts_d),iC] += 1.0*invdx
  # dirichlet value vector and cov
  dirichlet_vect = dirichlet_value_ex[j_d,i_d]
  dirichlet_sigma_vect = sigma_dirichlet_ex[j_d,i_d]
  cov_dirichlet = scysparse.diags(dirichlet_sigma_vect**2, format='csr')
  # Generate the vector and cov for pgrad.
  pgrad_x_vect = grad_x_ex[j,i]
  pgrad_y_vect = grad_y_ex[j,i]
  pgrad_vect = np.concatenate((pgrad_x_vect,pgrad_y_vect))
  cov_pgrad_x = sigma_grad_x_ex[j,i]**2
  cov_pgrad_y = sigma_grad_y_ex[j,i]**2
  cov_pgrad_vect = np.concatenate((cov_pgrad_x, cov_pgrad_y))
  cov_pgrad = scysparse.diags(cov_pgrad_vect, format='csr')
  
  # Construct the full operator.
  Operator_GLS = scysparse.bmat([[Operator_Gx],[Operator_Gy],[Operator_d]])

  # Construct the full mapping matrics and get the rhs.
  Map_pgrad = scysparse.bmat([[Map_Gx, None],[None, Map_Gy],[scysparse.csr_matrix((Npts_d,Npts),dtype=np.float), None]])
  Map_dirichlet = scysparse.bmat([[scysparse.csr_matrix((Npts_x+Npts_y,Npts_d),dtype=np.float)],[Map_d]])
  rhs = Map_pgrad.dot(pgrad_vect) + Map_dirichlet.dot(dirichlet_vect)

  # Evaluate the covriance matrix for the rhs
  cov_rhs = Map_pgrad * cov_pgrad * Map_pgrad.transpose() + Map_dirichlet * cov_dirichlet * Map_dirichlet.transpose()

  # Solve for the WLS solution
  weights_vect = cov_rhs.diagonal()**(-1)
  weights_matrix = scysparse.diags(weights_vect,format='csr')
  # Operator_WLS = weights_matrix * Operator_GLS
  # rhs_WLS = weights_matrix.dot(rhs)
  sys_LHS = Operator_GLS.transpose() * weights_matrix * Operator_GLS
  sys_rhs = (Operator_GLS.transpose() * weights_matrix).dot(rhs)

  # Get the solution from lsqr
  # p_vect_wls = splinalg.lsqr(Operator_WLS,rhs_WLS)[0]
  # Pn_WLS = np.zeros(fluid_mask_ex.shape)
  # Pn_WLS[j,i] = p_vect_wls

  # Solve for the WLS solution
  p_vect_wls = splinalg.spsolve(sys_LHS, sys_rhs)
  Pn_WLS_ex = np.zeros(fluid_mask_ex.shape)
  Pn_WLS_ex[j,i] = p_vect_wls

  # Perform the uncertainty propagation
  sigma_Pn_ex = np.zeros((Ny+2,Nx+2))
  if uncertainty_quantification == True:
    cov_sys_rhs = (Operator_GLS.transpose() * weights_matrix) * cov_rhs * (Operator_GLS.transpose() * weights_matrix).transpose()
    sys_LHS_inv = linalg.inv(sys_LHS.A)
    Cov_p = np.matmul(np.matmul(sys_LHS_inv, cov_sys_rhs.A), sys_LHS_inv.T)
    Var_p_vect = np.diag(Cov_p)
    sigma_Pn_ex[j,i] = Var_p_vect**0.5

  return Pn_WLS_ex[1:-1,1:-1], sigma_Pn_ex[1:-1,1:-1]

      















  





