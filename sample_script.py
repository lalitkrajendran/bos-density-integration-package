#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from scipy.io import savemat
from scipy.io import loadmat
import timeit

# import density integration functions
from DensityIntegrationUncertaintyQuantification import Density_integration_Poisson_uncertainty
from DensityIntegrationUncertaintyQuantification import Density_integration_WLS_uncertainty

# import helper functions
import loadmat_functions

def set_bc(X, experimental_parameters):
    
    # define dirichlet boundary points (minimum one point) - here defined to be all boundaries
    dirichlet_label = create_border_array(X.shape[0], X.shape[1], fill_val=1)

    # set density and density uncertainty at these boundary points
    rho_dirichlet = experimental_parameters['rho_0'] * dirichlet_label
    sigma_rho_dirichlet = experimental_parameters['sigma_rho_0'] * dirichlet_label

    # convert to boolean array
    dirichlet_label = dirichlet_label.astype('bool')

    return dirichlet_label, rho_dirichlet, sigma_rho_dirichlet


def convert_displacements_to_physical_units(X_pix, Y_pix, U, V, sigma_U, sigma_V, experimental_parameters, mask):

    # calculate co-ordinates in the density gradient plane (m)
    X = (X_pix - experimental_parameters['x0']) * experimental_parameters['pixel_pitch'] * 1 / experimental_parameters['magnification_grad']
    Y = (Y_pix - experimental_parameters['y0']) * experimental_parameters['pixel_pitch'] * 1 / experimental_parameters['magnification_grad']
    
    # calculate density gradients from displacements (kg/m^4)
    rho_x = U * experimental_parameters['pixel_pitch'] * 1 / (
        experimental_parameters['Z_D'] * experimental_parameters['magnification'] *
        experimental_parameters[
            'delta_z']) * 1 / experimental_parameters['gladstone_dale'] * experimental_parameters['n_0']
    rho_y = V * experimental_parameters['pixel_pitch'] * 1 / (
        experimental_parameters['Z_D'] * experimental_parameters['magnification'] *
        experimental_parameters[
            'delta_z']) * 1 / experimental_parameters['gladstone_dale'] * experimental_parameters['n_0']

    # calculate density gradient uncertainty from displacement uncertainty (kg/m^4)
    sigma_rho_x = sigma_U * experimental_parameters['pixel_pitch'] * \
                  1 / (
                  experimental_parameters['Z_D'] * experimental_parameters['magnification'] * experimental_parameters[
                      'delta_z']) * \
                  1 / experimental_parameters['gladstone_dale'] * experimental_parameters['n_0']
    sigma_rho_y = sigma_V * experimental_parameters['pixel_pitch'] * \
                  1 / (
                  experimental_parameters['Z_D'] * experimental_parameters['magnification'] * experimental_parameters[
                      'delta_z']) * \
                  1 / experimental_parameters['gladstone_dale'] * experimental_parameters['n_0']

    # --------------------------
    # enforce mask and ensure valid values
    # --------------------------
    # set uncertainties in masked regions to be zero (e.g. for image matching)
    sigma_rho_x[mask == False] = 0.0
    sigma_rho_y[mask == False] = 0.0

    # set nan values to be zero
    sigma_rho_x[~np.isfinite(sigma_rho_x)] = 0.0
    sigma_rho_y[~np.isfinite(sigma_rho_y)] = 0.0

    # set zero values to a small non-zero number
    sigma_rho_x[sigma_rho_x == 0] = 1e-8 
    sigma_rho_y[sigma_rho_y == 0] = 1e-8 

    return X, Y, rho_x, rho_y, sigma_rho_x, sigma_rho_y


def create_border_array(nr, nc, fill_val=1):
    # create array with fill val
    border_array = np.ones((nr, nc)) * fill_val
    border_array[1:-1, 1:-1] = 0

    return border_array


def create_mask(X, Eval):
    # --------------------------
    # set mask
    # --------------------------
    # create binary mask. True for flow, False for blocked region.
    if len(Eval.shape) == 1:
        mask = np.reshape(a=Eval, newshape=(X.shape[1], X.shape[0]))
        mask = np.transpose(mask)
    else:
        mask = Eval

    return mask


def load_displacements(filename, displacement_uncertainty_method):
    # load the data:
    data = loadmat_functions.loadmat(filename=filename)  # , squeeze_me=True)

    # extract co-ordinates
    X_pix = data['X'].astype('float')
    Y_pix = data['Y'].astype('float')

    # extract displacements (sign conversion for the ray tracing code)
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


def main():
    
    # file containing displacements and uncertainties
    filename = 'sample-displacements.mat'
    
    # set integration method ('p' for poisson or 'w' for wls)
    density_integration_method = 'p'

    # displacement uncertainty method
    displacement_uncertainty_method = 'MC'

    # -------------------------------------------------
    # experimental parameters for density integration
    # -------------------------------------------------
    experimental_parameters = dict()

    # ambient density (kg/m^3)
    experimental_parameters['rho_0'] = 1.225

    # uncertainty in the free-stream density (kg/m^3)
    experimental_parameters['sigma_rho_0'] = 0.0 #0.05 * peak_density_offset #5% of peak difference

    # gladstone dale constant (m^3/kg)
    experimental_parameters['gladstone_dale'] = 0.225e-3

    # ambient refractive index
    experimental_parameters['n_0'] = 1.0 + experimental_parameters['gladstone_dale']*experimental_parameters['rho_0']

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

    # pixels corresponding to center of the FOV
    experimental_parameters['x0'] = 256 
    experimental_parameters['y0'] = 256 

    # size of a pixel on the camera sensor (m)
    experimental_parameters['pixel_pitch'] = 10e-6

    # focal length of camera lens (m)
    experimental_parameters['focal_length'] = 105e-3

    # non-dimensional magnification of the dot pattern (can also set it directly)
    experimental_parameters['magnification'] = experimental_parameters['focal_length'] / (
    experimental_parameters['object_distance'] - experimental_parameters['focal_length'])

    # non-dimensional magnification of the midpoint of the density gradient field
    experimental_parameters['magnification_grad'] = experimental_parameters['magnification'] \
                                                    * experimental_parameters['Z_B'] / experimental_parameters['Z_A']
    
    # --------------------------
    # processing
    # --------------------------
    # load displacements from file
    X_pix, Y_pix, U, V, sigma_U, sigma_V, Eval = load_displacements(filename, displacement_uncertainty_method)

    # set mask
    mask = create_mask(X_pix, Eval)
    
    # convert displacements to density gradients and co-ordinates to physical units 
    X, Y, rho_x, rho_y, sigma_rho_x, sigma_rho_y = convert_displacements_to_physical_units(X_pix, Y_pix, U, V, sigma_U, sigma_V, experimental_parameters, mask)

    # define dirichlet boundary points (minimum one point) - here defined to be all boundaries
    dirichlet_label, rho_dirichlet, sigma_rho_dirichlet = set_bc(X, experimental_parameters)
        
    # calculate density and uncertainty
    if density_integration_method == 'p':
        rho, sigma_rho = Density_integration_Poisson_uncertainty(X, Y, mask,rho_x, rho_y,
                                                            dirichlet_label, rho_dirichlet,
                                                            uncertainty_quantification=True,
                                                            sigma_grad_x=sigma_rho_x, sigma_grad_y=sigma_rho_y,
                                                            sigma_dirichlet=sigma_rho_dirichlet)
    elif density_integration_method == 'w':
        rho, sigma_rho = Density_integration_WLS_uncertainty(X, Y, mask,rho_x, rho_y,
                                                            dirichlet_label, rho_dirichlet,
                                                            uncertainty_quantification=True,
                                                            sigma_grad_x=sigma_rho_x, sigma_grad_y=sigma_rho_y,
                                                            sigma_dirichlet=sigma_rho_dirichlet)

    # # save the results to file
    # savemat(file_name='sample-result.mat', mdict={'X': X, 'Y': Y, 'rho': rho, 'sigma_rho': sigma_rho,
    #                     'dirichlet_label': dirichlet_label, 'rho_dirichlet':rho_dirichlet, 'sigma_rho_dirichlet':sigma_rho_dirichlet
    #                     }, long_field_names=True)

    # Plot the results
    fig1 = plt.figure(1, figsize=(12,8))
    plt.figure(1)
    ax1 = fig1.add_subplot(2,2,1)
    ax2 = fig1.add_subplot(2,2,2)
    ax3 = fig1.add_subplot(2,2,3)
    ax4 = fig1.add_subplot(2,2,4)

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

    # save plot to file
    plt.savefig('sample-result.png')
    plt.close()

if __name__ == '__main__':
    main()
















