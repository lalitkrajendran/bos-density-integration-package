#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.io import loadmat
import timeit

# import density integration functions
from DensityIntegrationUncertaintyQuantification import Density_integration_Poisson_uncertainty
from DensityIntegrationUncertaintyQuantification import Density_integration_WLS_uncertainty

import loadmat_functions

def plot_figures(X, Y, rho_x, rho_y, rho, sigma_rho):

    # --------------------------
    # Plot the results
    # --------------------------
    fig = plt.figure(1, figsize=(12,8))
    plt.figure(1)
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    # plot x gradient
    plt.axes(ax1)
    plt.pcolormesh(X, Y, rho_x)
    plt.colorbar()
    ax1.set_aspect('equal')
    plt.title('rho_x')

    # plot y gradient
    plt.axes(ax2)
    plt.pcolormesh(X, Y, rho_y)
    plt.colorbar()
    ax2.set_aspect('equal')
    plt.title('rho_y')

    # plot density
    plt.axes(ax3)
    plt.pcolormesh(X, Y, rho)
    plt.colorbar()
    ax3.set_aspect('equal')
    plt.title('rho')

    # plot density uncertainty7
    plt.axes(ax4)
    plt.pcolormesh(X, Y, sigma_rho)
    plt.colorbar()
    ax4.set_aspect('equal')
    plt.title('sigma rho')

    # plt.tight_layout()

    return fig


def create_border_array(nr, nc, fill_val=1):
    """
    Function to create a binary array with non-zero elements along the four boundaries

    Args:
        nr, nc (int): number of rows and columns of the binary array to be created
        fill_val (int, optional): non-zero value to be assigned to the border elements. Defaults to 1.

    Returns:
        border_array: the 2d array with non-zero elements along the borders        
    """

    # create array with fill val
    border_array = np.ones((nr, nc)) * fill_val
    border_array[1:-1, 1:-1] = 0

    return border_array


def set_bc(nr, nc, rho_0, sigma_rho_0):
    """
    Function to set matrices to enforce dirichlet boundary conditions

    Args:
        nr, nc (int): number of rows and columns
        rho_0 (float, scalar): reference density (kg/m^3)
        sigma_rho_0 (float, scalar): uncertainty in reference density (kg/m^3) 

    Returns:
        dirichlet_label: binary array specifying regions where reference density will be imposed
        rho_dirichlet (float 2D): array containing reference density values at the non-zero dirichlet label points
        sigma_rho_dirichlet (float 2D): array containing reference density UNCERTAINTY values at the non-zero dirichlet label points
    """
    
    # define dirichlet boundary points (minimum one point) - here defined to be all boundaries
    # THIS NEEDS TO BE MODIFIED FOR EACH EXPERIMENT
    dirichlet_label = create_border_array(nr, nc, fill_val=1)

    # # sample to set left boundary to be 1s
    # dirichlet_label = np.zeros((nr, nc))
    # dirichlet_label[:, 0] = 1
    # dirichlet_label[10, 0] = 1

    # set density and density uncertainty at these boundary points
    # HERE SET TO BE THE SAME VALUE FOR ALL BOUNDARY POINTS
    rho_dirichlet = rho_0 * dirichlet_label
    sigma_rho_dirichlet = sigma_rho_0 * dirichlet_label

    # convert to boolean array
    dirichlet_label = dirichlet_label.astype('bool')

    return dirichlet_label, rho_dirichlet, sigma_rho_dirichlet


def calculate_gradients_from_displacements(U, experimental_parameters):
    """
    Function to calculate density gradients from displacements using the BOS equation

    Inputs:
        U: pixel displacements
        experimental_parameters: dictionary of optical layout and other parameters
    
    Returns:
        rho_x: density gradient (kg/m^4)
    """
    # extract experimental parameters
    M, Z_D, l_p, delta_z, gladstone_dale, n_0 = experimental_parameters['magnification'], experimental_parameters['Z_D'], experimental_parameters['pixel_pitch'], experimental_parameters['delta_z'], experimental_parameters['gladstone_dale'], experimental_parameters['n_0']

    # gradient calculation from BOS equation
    rho_x = U * l_p * 1 / (Z_D * M * delta_z) * 1 / gladstone_dale * n_0
    
    return rho_x


def convert_displacements_to_physical_units(X_pix, Y_pix, U, V, sigma_U, sigma_V, experimental_parameters, mask):
    """
    Function to convert displacements to density gradients and co-ordinates to physical units.

    Args:
        X_pix, Y_pix (float): co-ordinate grid in pixels        
        U, V (float): X, Y displacements in pixels
        sigma_U, sigma_V (float): X, Y displacements uncertainty in pixels
        experimental_parameters (dict): structure of all the experimental parameters
        mask (float): binary array of the integration mask

    Returns:
        X, Y: co-ordinate grid in meters
        rho_x, rho_y (float): X, Y density gradients (kg/m^4)
        sigma_rho_x, sigma)rho_y (float): X, Y density gradient uncertainties (kg/m^4)        
    """

    # calculate co-ordinates in the density gradient plane (m)
    X = (X_pix - experimental_parameters['x0']) * experimental_parameters['pixel_pitch'] * 1 / experimental_parameters['magnification_grad']
    Y = (Y_pix - experimental_parameters['y0']) * experimental_parameters['pixel_pitch'] * 1 / experimental_parameters['magnification_grad']
    
    # calculate density gradients from displacements (kg/m^4)
    rho_x = calculate_gradients_from_displacements(U, experimental_parameters)
    rho_y = calculate_gradients_from_displacements(V, experimental_parameters)

    # calculate density gradient uncertainty from displacement uncertainty (kg/m^4)
    sigma_rho_x = calculate_gradients_from_displacements(sigma_U, experimental_parameters)
    sigma_rho_y = calculate_gradients_from_displacements(sigma_V, experimental_parameters)

    # --------------------------
    # enforce mask and ensure valid values for uncertainties
    # --------------------------
    # set uncertainties in masked regions to be zero (e.g. for image matching)
    sigma_rho_x[mask == False] = 0.0
    sigma_rho_y[mask == False] = 0.0

    # set nan values to be zero
    sigma_rho_x[~np.isfinite(sigma_rho_x)] = 0.0
    sigma_rho_y[~np.isfinite(sigma_rho_y)] = 0.0

    # set zero values to a small non-zero number (OR WLS will blow up)
    sigma_rho_x[sigma_rho_x == 0] = 1e-12 
    sigma_rho_y[sigma_rho_y == 0] = 1e-12

    return X, Y, rho_x, rho_y, sigma_rho_x, sigma_rho_y


def create_mask(nr, nc, Eval):
    """
    Function to create the integration mask

    Args:
        nr (int): number of rows
        nc (int): number of columns
        Eval (float): binary array of the mask

    Returns:
        mask (bool): integration mask as a boolean array (True for integration, False otherwise)
    """
    
    # --------------------------
    # set mask
    # --------------------------
    # create binary mask. True for flow, False for blocked region.
    if len(Eval.shape) == 1:
        # mask = np.reshape(a=Eval, newshape=(X.shape[1], X.shape[0]))
        mask = np.reshape(a=Eval, newshape=(nc, nr))
        mask = np.transpose(mask)
    else:
        mask = Eval

    return mask


def load_displacements_correlation(filename, displacement_uncertainty_method):
    """
    Function to load the displacements from Prana processing

    Args:
        filename (str): file name
        displacement_uncertainty_method (str): name of the displacement uncertainty method

    Returns:
        X_pix, Y_pix: pixel co-ordinate grid
        U, V: X, Y displacements
        sigma_U, sigma_V: X, Y, displacement uncertainties
        Eval: Processing mask (binary array)
    """
    
    # load the data:
    data = loadmat_functions.loadmat(filename=filename)

    # extract co-ordinates
    X_pix = data['X'].astype('float')
    Y_pix = data['Y'].astype('float')

    # extract displacements (sign conversion for the SAMPLE DATASET ONLY)
    U = -data['U']
    V = -data['V']

    # extract displacement uncertainties
    sigma_U = data['uncertainty2D'][displacement_uncertainty_method + 'x']
    sigma_V = data['uncertainty2D'][displacement_uncertainty_method + 'y']

    # bias correction for MC
    if displacement_uncertainty_method == 'MC':
        sigma_U = np.sqrt(sigma_U ** 2 - data['uncertainty2D']['biasx'] ** 2)
        sigma_V = np.sqrt(sigma_V ** 2 - data['uncertainty2D']['biasy'] ** 2)

    # extract mask
    Eval = data['Eval'] + 1

    return X_pix, Y_pix, U, V, sigma_U, sigma_V, Eval


def grid_PTV_velocity_uncertainty_2D(Xn, Yn, fluid_mask, xp, yp, up, vp, up_unc, vp_unc, N_avg=3, dist_epsilon=1e-6):
  # This function maps the velocity and its uncertainty from PTV data (on unstructured particle locations) to
  # a predefined Cartesian grid using inverse-distance based weighted average.
  # Inputs:
  # Xn, Yn: 2d array specifying the Cartesian grid.
  # fluid_mask: 2d array of a binary mask specifying the flow region. The particle data will be interpolated only onto the grid points with fluid_mask==True
  # xp, yp: 1d array of particle locations.
  # up, vp: 1d array of particle velocity values.
  # up_unc, vp_unc: 1d array of paticle velocity uncertainty (std).
  # N_avg: the number of particle data points used to determine the velocity for each grid point.
  # dist_epsilon: the minimum distance allowed for setting up the weights (to avoid signularity).
  # Outputs: 
  # Un, Vn: the 3d arrays of velocity on grid.
  # Un_unc, Vn_unc: 3d arrays of velocity uncertainty on grid.
  # Cov_u, Cov_v: the sparse matrices of covariance matrices of interpolated velocity.

  Ny,Nx = np.shape(Xn)

  j,i = np.where(fluid_mask==True)
  Npts = len(j)
  Np = len(xp)

  # Setup the linear operator to map the p velocity to grid using inverse distance based average
  Map_M = scysparse.csr_matrix((Npts,Np),dtype=np.float)
  for ct_pt in range(Npts):
    x_grid = Xn[j[ct_pt],i[ct_pt]]
    y_grid = Yn[j[ct_pt],i[ct_pt]]    
    # find closest particles for linear interpolation.
    dist = ((xp - x_grid)**2 + (yp - y_grid)**2)**0.5
    dist[dist<dist_epsilon] = dist_epsilon
    close_arg = np.argsort(dist)[:N_avg]
    inv_dist = 1.0 / dist[close_arg]
    inv_dist_weight = inv_dist / np.sum(inv_dist)
    for col in range(N_avg):
      Map_M[ct_pt,close_arg[col]] = inv_dist_weight[col]
  
  # Peform the interpolation.
  Un = np.zeros((Ny,Nx))
  Vn = np.zeros((Ny,Nx))
  
  Un[j,i] = Map_M.dot(up)
  Vn[j,i] = Map_M.dot(vp)
  
  # Perform the ucnertainty propagation.
  Un_unc = np.zeros((Ny,Nx))
  Cov_up = scysparse.diags(up_unc**2,format='csr',dtype=np.float)
  Cov_u = Map_M * Cov_up * Map_M.transpose()
  Un_unc[j,i] = Cov_u.diagonal()**0.5

  Vn_unc = np.zeros((Ny,Nx))
  Cov_vp = scysparse.diags(vp_unc**2,format='csr',dtype=np.float)
  Cov_v = Map_M * Cov_vp * Map_M.transpose()
  Vn_unc[j,i] = Cov_v.diagonal()**0.5

  return Un, Vn, Un_unc, Vn_unc, Cov_u, Cov_v


def load_displacements_tracking(filename, displacement_uncertainty_method='crlb', experimental_parameters):
    """
    Function to load displacments from tracking processing and interpolate them to a grid

    Args:
        filename (str): name of the file containing the displacements
        displacement_uncertainty_method (str): uncertainty quantification method. defaults to crlb
        experimental_parameters (dict): dictionary of all experimental parameters

    Returns:
        X_grid, Y_grid: co-ordinate grid (pix)
        U_grid, V_grid: X, Y displacements (pix)
        sigma_U_grid, sigma_V_grid: X, Y displacement uncertainties (pix)
    """
    # ----------------------------
    # load and extract data
    # ----------------------------

    # load the data:
    data = loadmat_functions.loadmat(filename)

    # extract track info
    X_track = data['tracks'][:, 0]
    Y_track = data['tracks'][:, 2]
    if validation:
        U_track = data['tracks'][:, 15]
        V_track = data['tracks'][:, 16]
    else:
        U_track = data['tracks'][:, 13]
        V_track = data['tracks'][:, 14]

    # extract uncertainties
    if displacement_uncertainty_method == 'crlb':
        sigma_U_track = data['uncertainty2D']['U_std']
        sigma_V_track = data['uncertainty2D']['V_std']
    else:
        sigma_U_track = np.zeros_like(a=X_track)
        sigma_V_track = np.zeros_like(a=X_track)

    # ----------------------------
    # identify extents
    # ----------------------------
    xmin = X_track.min()
    xmax = X_track.max()
    ymin = Y_track.min()
    ymax = Y_track.max()

    # ----------------------------
    # interpolate tracks onto a regular grid
    # ----------------------------
    # calculate expected number of dots along x and y
    num_dots_x = np.round((xmax - xmin + 1) / experimental_parameters['dot_spacing'])
    num_dots_y = np.round((ymax - ymin + 1) / experimental_parameters['dot_spacing'])

    # generaet 1D co-ordinate arrays
    x_grid = np.linspace(start=np.ceil(xmin), stop=np.floor(xmax), num=num_dots_x)
    y_grid = np.linspace(start=np.ceil(ymin), stop=np.floor(ymax), num=num_dots_y)

    # generate 2D co-ordinate grid
    [X_grid, Y_grid] = np.meshgrid(x_grid, y_grid)

    # interpolate results onto a regular grid
    U_grid, V_grid, sigma_U_grid, sigma_V_grid, Cov_u, Cov_v = grid_PTV_velocity_uncertainty_2D(X_grid, Y_grid,
                    np.ones_like(X_grid), X_track, Y_track, U_track, V_track, sigma_U_track, sigma_V_track, N_avg=8)

    # set nan elements to zero
    U_grid[np.logical_not(np.isfinite(U_grid))] = 0.0
    V_grid[np.logical_not(np.isfinite(V_grid))] = 0.0
    sigma_U_grid[np.logical_not(np.isfinite(sigma_U_grid))] = 0.0
    sigma_V_grid[np.logical_not(np.isfinite(sigma_V_grid))] = 0.0

    return X_grid, Y_grid, U_grid, V_grid, sigma_U_grid, sigma_V_grid


def main():
    
    # file containing displacements and uncertainties
    filename = 'sample-displacements.mat'
    
    # displacement estimation method ('c' for correlation and 't' for tracking)
    displacement_estimation_method = 'c'
    
    # displacement uncertainty method ('MC' for correlation and 'crlb' for tracking)
    displacement_uncertainty_method = 'MC'

    # set integration method ('p' for poisson or 'w' for wls)
    density_integration_method = 'w'
    
    # -------------------------------------------------
    # experimental parameters for density integration
    # -------------------------------------------------
    experimental_parameters = dict()

    # ambient/reference density (kg/m^3)
    experimental_parameters['rho_0'] = 1.225

    # uncertainty in the reference density (kg/m^3) (MUST BE GREATER THAN 0)
    experimental_parameters['sigma_rho_0'] = 1e-10

    # gladstone dale constant (m^3/kg)
    experimental_parameters['gladstone_dale'] = 0.225e-3

    # ambient refractive index
    experimental_parameters['n_0'] = 1.0 + experimental_parameters['gladstone_dale'] * experimental_parameters['rho_0']

    # thickness of the density gradient field (m)
    experimental_parameters['delta_z'] = 0.01

    # distance between lens and dot target (object / working distance) (m)
    experimental_parameters['object_distance'] = 1.0

    # distance between the mid-point of the density gradient field and the dot pattern (m)
    experimental_parameters['Z_D'] = 0.25

    # distance between the mid-point of the density gradient field and the camera lens (m)
    experimental_parameters['Z_A'] = experimental_parameters['object_distance'] - experimental_parameters['Z_D']
    
    # distance between the dot pattern and the camera lens (m)
    experimental_parameters['Z_B'] = experimental_parameters['object_distance']

    # origin (pixels)
    experimental_parameters['x0'] = 256 
    experimental_parameters['y0'] = 256 

    # size of a pixel on the camera sensor (m)
    experimental_parameters['pixel_pitch'] = 10e-6

    # focal length of camera lens (m)
    experimental_parameters['focal_length'] = 105e-3

    # non-dimensional magnification of the dot pattern (can also set it directly)
    experimental_parameters['magnification'] = experimental_parameters['focal_length'] / (
    experimental_parameters['object_distance'] - experimental_parameters['focal_length'])

    # non-dimensional magnification of the mid-z-PLANE of the density gradient field
    experimental_parameters['magnification_grad'] = experimental_parameters['magnification'] \
                                                    * experimental_parameters['Z_B'] / experimental_parameters['Z_A']
    
    # --------------------------
    # processing
    # --------------------------
    # load displacements and uncertainties from file        
    if displacement_estimation_method == 'c':
        # correlation
        X_pix, Y_pix, U, V, sigma_U, sigma_V, Eval = load_displacements_correlation(filename, displacement_uncertainty_method)    
    elif displacement_estimation_method == 't':
        # tracking
        X_pix, Y_pix, U, V, sigma_U, sigma_V = load_displacements_tracking(filename, displacement_uncertainty_method, experimental_parameters)    

    # create mask array (1 for flow, 0 elsewhere) - only implemented for Correlation at the moment
    if displacement_estimation_method == 'c':
        mask = create_mask(X_pix.shape[0], X_pix.shape[1], Eval)
    elif displacement_estimation_method == 't':        
        mask = np.ones_like(a=U)

    # convert displacements to density gradients and co-ordinates to physical units 
    X, Y, rho_x, rho_y, sigma_rho_x, sigma_rho_y = convert_displacements_to_physical_units(X_pix, Y_pix, U, V, sigma_U, sigma_V, experimental_parameters, mask)

    # define dirichlet boundary points (minimum one point) - here defined to be all boundaries
    # This is specific to the current dataset
    dirichlet_label, rho_dirichlet, sigma_rho_dirichlet = set_bc(X_pix.shape[0], X_pix.shape[1], experimental_parameters['rho_0'], experimental_parameters['sigma_rho_0'])
        
    # calculate density and uncertainty
    if density_integration_method == 'p':
        # Poisson
        rho, sigma_rho = Density_integration_Poisson_uncertainty(X, Y, mask, rho_x, rho_y,
                                                            dirichlet_label, rho_dirichlet,
                                                            uncertainty_quantification=True,
                                                            sigma_grad_x=sigma_rho_x, sigma_grad_y=sigma_rho_y,
                                                            sigma_dirichlet=sigma_rho_dirichlet)
    elif density_integration_method == 'w':
        # Weighted Least Squares
        rho, sigma_rho = Density_integration_WLS_uncertainty(X, Y, mask,rho_x, rho_y,
                                                            dirichlet_label, rho_dirichlet,
                                                            uncertainty_quantification=True,
                                                            sigma_grad_x=sigma_rho_x, sigma_grad_y=sigma_rho_y,
                                                            sigma_dirichlet=sigma_rho_dirichlet)

    # save the results to file
    savemat(filename='sample-result.mat', mdict={'X': X, 'Y': Y, 'rho': rho, 'sigma_rho': sigma_rho,
                        'dirichlet_label': dirichlet_label, 'rho_dirichlet':rho_dirichlet, 'sigma_rho_dirichlet':sigma_rho_dirichlet
                        }, long_field_names=True)

    # plot results
    fig = plot_figures(X, Y, rho_x, rho_y, rho, sigma_rho)
    
    # save plot to file
    fig.savefig('sample-result.png')
    plt.close()


if __name__ == '__main__':
    main()
















