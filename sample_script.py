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

import helper_functions

def main():
    
    # file containing displacements and uncertainties
    filename = 'sample-displacements.mat'
    
    # displacement estimation method ('c' for correlation and 't' for tracking)
    displacement_estimation_method = 'c'
    
    # displacement uncertainty method ('MC' for correlation and 'crlb' for tracking)
    displacement_uncertainty_method = 'MC'

    # set integration method ('p' for poisson or 'w' for wls)
    density_integration_method = 'w'
    
    # dataset type (syntehtic or experiment)
    dataset_type = 'synthetic'

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

    # uncertainty in magnification
    experimental_parameters['sigma_M'] = 0.1

    # uncertainty in Z_D (m)
    experimental_parameters['sigma_Z_D'] = 1e-3


    # non-dimensional magnification of the mid-z-PLANE of the density gradient field
    experimental_parameters['magnification_grad'] = experimental_parameters['magnification'] \
                                                    * experimental_parameters['Z_B'] / experimental_parameters['Z_A']
    
    # --------------------------
    # processing
    # --------------------------
    # load displacements and uncertainties from file        
    if displacement_estimation_method == 'c':
        # correlation
        X_pix, Y_pix, U, V, sigma_U, sigma_V, Eval = helper_functions.load_displacements_correlation(filename, displacement_uncertainty_method)    
    elif displacement_estimation_method == 't':
        # tracking
        X_pix, Y_pix, U, V, sigma_U, sigma_V = helper_functions.load_displacements_tracking(filename, experimental_parameters['dot_spacing'], displacement_uncertainty_method)    

    # account for sign convention
    if dataset_type == 'synthetic':
        U *= -1
        V *= -1

    # create mask array (1 for flow, 0 elsewhere) - only implemented for Correlation at the moment
    if displacement_estimation_method == 'c':
        mask = helper_functions.create_mask(X_pix.shape[0], X_pix.shape[1], Eval)
    elif displacement_estimation_method == 't':        
        mask = np.ones_like(a=U)

    # convert displacements to density gradients and co-ordinates to physical units 
    X, Y, rho_x, rho_y, sigma_rho_x, sigma_rho_y = helper_functions.convert_displacements_to_physical_units(X_pix, Y_pix, U, V, sigma_U, sigma_V, experimental_parameters, mask)

    # define dirichlet boundary points (minimum one point) - here defined to be all boundaries
    # This is specific to the current dataset
    dirichlet_label, rho_dirichlet, sigma_rho_dirichlet = helper_functions.set_bc(X_pix.shape[0], X_pix.shape[1], experimental_parameters['rho_0'], experimental_parameters['sigma_rho_0'])
        
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
    fig = helper_functions.plot_figures(X, Y, rho_x, rho_y, rho, sigma_rho)
    
    # save plot to file
    fig.savefig('sample-result.png')
    plt.close()


if __name__ == '__main__':
    main()
















