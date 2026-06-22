#!/usr/bin/env python3
# Support functions for HW4 and HW5
# Created 29 April 2026 by Sam Gardner <samuel.gardner@ttu.edu>

import numpy as np
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import chi2
from pyxlma import coords
from pyhigh import get_elevation_batch
from shapely import wkt
import re

EPSILON_0 = 8.8541e-12 # F/m


def monopole_E_change(x, y, z, q, xi, yi, zi):
    """
    Calculate the change in electric field due to a monopole charge at a given position.

    Parameters
    ----------
    x, y, z, q : array-like
        The x, y, z coordinates of the charge (stroke) positions. Shape should be (N_strokes,).
    xi, yi, zi : array-like
        The x, y, z coordinates of the station positions. Shape should be (N_stations,).

    Returns
    -------
    delta_E : ndarray
        The change in electric field at each station due to each charge. Shape will be (N_strokes, N_stations).
    """
    # reshape so that all 'stroke' parameters are on axis 0
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    z = np.array(z).reshape(-1, 1)
    q = np.array(q).reshape(-1, 1)
    # reshape so that all 'station' parameters are on axis 1
    xi = np.array(xi).reshape(1, -1)
    yi = np.array(yi).reshape(1, -1)
    zi = np.array(zi).reshape(1, -1)
    # compute r vector from stroke to station
    dx = x - xi
    dy = y - yi
    dz = z - zi
    r = np.array((dx, dy, dz))
    # calculate E-field change using monopole formula, \frac{1}{4\pi\epsilon_0} \frac{2qz}{r^3}
    k_const = 1/(4*np.pi*EPSILON_0)
    delta_E = k_const * (2*q*dz / (np.linalg.norm(r, axis=0)**3) )
    return delta_E # V/m

def dipole_E_change(x, y, z, q, xi, yi, zi, dr):
    """
    Calculate the change in electric field due to a dipole charge at a given position.

    Parameters
    ----------
    x, y, z, q : array-like
        The x, y, z coordinates of the charge (stroke) positions. Shape should be (N_strokes,).
    xi, yi, zi : array-like
        The x, y, z coordinates of the station positions. Shape should be (N_stations,).
    dr : array-like
        The dipole moment vector components (dr_x, dr_y, dr_z). Shape should be (3, N_strokes).

    Returns
    -------
    delta_E : ndarray
        The change in electric field at each station due to each charge. Shape will be (N_strokes, N_stations).
    """
    # reshape so that all 'stroke' parameters are on axis 0
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    z = np.array(z).reshape(-1, 1)
    q = np.array(q).reshape(-1, 1)
    drx = dr[0, :].reshape(-1, 1)
    dry = dr[1, :].reshape(-1, 1)
    drz = dr[2, :].reshape(-1, 1)
    # reshape so that all 'station' parameters are on axis 1
    xi = np.array(xi).reshape(1, -1)
    yi = np.array(yi).reshape(1, -1)
    zi = np.array(zi).reshape(1, -1)
    # compute r vector from stroke to station
    dx = x - xi
    dy = y - yi
    dz = z - np.zeros_like(xi)
    # compute charge moment vector, p = dr * q
    p = np.array((drx, dry, drz)) * q
    r = np.array((dx, dy, dz))
    r_mag = np.linalg.norm(r, axis=0)
    # dot product of r and p, r_dot_p = r_x*p_x + r_y*p_y + r_z*p_z
    r_dot_p = np.sum(r*p, axis=0)
    # calculate E-field change using dipole formula, \frac{1}{4\pi\epsilon_0} (\frac{2p_z}{r^3} - \frac{6z}{r^5} (r \cdot p))
    k_const = 1/(4*np.pi*EPSILON_0)
    delta_E = k_const * ((2*p[2, :]/(r_mag**3)) - (6*dz)/(r_mag**5) * r_dot_p)
    return delta_E # V/m

def delta_E_error(params, observed_E, station_info):
    """
    Calculate the error between the observed electric field changes and the predicted electric field changes for a given set of monopole parameters.

    Parameters
    ----------
    params : array-like
        An array containing the monopole parameters (q, x, y, z) to be optimized. Shape should be (4,).
    observed_E : array-like
        A 1D array of shape (N_stations,) containing the observed electric field changes at each station for the stroke.
    station_info : pd.DataFrame
        A DataFrame containing the station information, including 'x' and 'y' and 'z' columns for the station coordinates.

    Returns
    -------
    error : ndarray
        A 1D array of shape (N_stations,) containing the error between the observed electric field changes and the predicted electric field changes for the given monopole parameters.
    """
    q, x, y, z = params
    predicted_E = monopole_E_change(x, y, z, q, station_info['x'], station_info['y'], station_info['z'])
    return observed_E - np.reshape(predicted_E, observed_E.shape)

def charge_retrieval(initial_guess_q, initial_guess_x, initial_guess_y, initial_guess_z, stroke_obs, station_df):
    """
    Retrieve the location and charge removed by a monopole discharge stroke.

    Parameters
    ----------
    initial_guess_q : float
        An initial guess for the charge of the stroke.
    initial_guess_x : float
        An initial guess for the x coordinate of the stroke.
    initial_guess_y : float
        An initial guess for the y coordinate of the stroke.
    initial_guess_z : float
        An initial guess for the z coordinate of the stroke.
    stroke_obs : np.ndarray
        A 1D array of shape (N_stations,) containing the observed electric field changes at each station for the stroke.
    station_df : pd.DataFrame
        A DataFrame containing the station information, including 'x' and 'y' and 'z' columns for the station coordinates.

    """
    z_min = station_df['z'].min()
    retrieved_opt = least_squares(delta_E_error,
                   x0=np.array([initial_guess_q, initial_guess_x, initial_guess_y, initial_guess_z]),
                   bounds=([-np.inf, -np.inf, -np.inf, z_min], [np.inf, np.inf, np.inf, np.inf]),
                   args=(stroke_obs,
                         station_df)).x
    return retrieved_opt

def multi_monopole_retrieval(all_monopoles, station_df, initial_guess_q=0):
    """
    Retrieve the location and charge removed by multiple monopole discharge strokes.

    
    Parameters
    ----------
    all_monopoles : np.ndarray
        A 2D array of shape (N_strokes, N_stations) containing the observed electric field changes at each station for each stroke.
    station_df : pd.DataFrame
        A DataFrame containing the station information, including 'x' and 'y' and 'z' columns for the station coordinates.
    initial_guess_q : float, optional
        An optional initial guess for the charge of each stroke. Default is 0.

    Returns
    -------
    retrieved_x : np.ndarray
        A 1D array of shape (N_strokes,) containing the retrieved x coordinates of each stroke.
    retrieved_y : np.ndarray
        A 1D array of shape (N_strokes,) containing the retrieved y coordinates of each stroke.
    retrieved_z : np.ndarray
        A 1D array of shape (N_strokes,) containing the retrieved z coordinates of each stroke.
    retrieved_q : np.ndarray
        A 1D array of shape (N_strokes,) containing the retrieved charges of each stroke.
    """
    retrieved_x = np.zeros(all_monopoles.shape[0])
    retrieved_y = np.zeros(all_monopoles.shape[0])
    retrieved_z = np.zeros(all_monopoles.shape[0])
    retrieved_q = np.zeros(all_monopoles.shape[0])

    for i in range(all_monopoles.shape[0]):
        print(f'Retrieving parameters for stroke {i+1}/{all_monopoles.shape[0]}')
        this_charge_obs = all_monopoles[i, :] # select a set of observations for a single charge
        station_largest_ob_idx = np.argmax(np.abs(this_charge_obs)) # find the station with the largest observation for this charge, and use that as the initial guess for the retrieval
        initial_guess = station_df.iloc[station_largest_ob_idx] # get station info for initial guess
        # retrieve the charge parameters using the charge retrieval function
        try:
            this_retrieval = charge_retrieval(initial_guess_q=1,
                                              initial_guess_x=initial_guess['x'],
                                              initial_guess_y=initial_guess['y'],
                                              initial_guess_z=initial_guess['z'] + 3000, # assume flash happens above the station
                                              stroke_obs=this_charge_obs, station_df=station_df)
        except ValueError as e:
            print(f"Failed to retrieve parameters for stroke {i}")
            print(f"Initial Guess: q={initial_guess_q}, x={initial_guess['x']}, y={initial_guess['y']}, z={initial_guess['z']}")
            print(f"Observations: {this_charge_obs}")
            print(f"Error: {e}")
            raise e
        # store the result of the retrieval in the appropriate arrays
        retrieved_x[i] = this_retrieval[1]
        retrieved_y[i] = this_retrieval[2]
        retrieved_z[i] = this_retrieval[3]
        retrieved_q[i] = this_retrieval[0]
    return retrieved_x, retrieved_y, retrieved_z, retrieved_q


def add_error_to_monopoles(all_monopoles, error_scale=150):
    """
    Retrieve the location and charge removed by multiple monopole discharge strokes.
    Adds normally distributed random noise to the observations for each station before performing the retrieval, to simulate realistic instrument error.

    Parameters
    ----------
    all_monopoles : np.ndarray
        A 2D array of shape (N_strokes, N_stations) containing the observed electric field changes at each station for each stroke.
    error_scale : float, optional
        The standard deviation of the normally distributed error to be added to the observations for each station. Default is 150 (V/m).

    Returns
    -------
    ret_err : np.ndarray
        A 4D array of shape (4, N_x, N_y, N_z) containing the retrieved x, y, z, and q values for each point in the grid after
        adding error to the observations and performing the retrieval.
    """
    all_monopoles_with_error = all_monopoles.copy()
    for station_idx in range(all_monopoles.shape[1]):
        this_station_error = np.random.normal(loc=0, scale=error_scale, size=all_monopoles.shape[0]) # add some normally distributed error to the observations for this station
        all_monopoles_with_error[:, station_idx] += this_station_error
    # ret_x_err, ret_y_err, ret_z_err, ret_q_err = multi_monopole_retrieval(all_monopoles_with_error, station_df, initial_guess_q=3)
    # ret_err = np.array([ret_x_err.reshape(x_grid.shape), ret_y_err.reshape(x_grid.shape), ret_z_err.reshape(x_grid.shape), ret_q_err.reshape(x_grid.shape)])
    return all_monopoles_with_error


if __name__ == "__main__":
    ## PARAMETERS:
    L_x, n_x = 2*15, 31
    L_y, n_y = 2*15, 31
    L_z, n_z = 10, 10
    Q_coloumbs = 3
    error_scale = 150 # V/m
    # Step 0: read in station dataframe
    rows = []
    with open('./stations.csv') as f:
        next(f)  # skip header line
        for line in f:
            line = line.rstrip('\n')
            # WKT is always the first quoted field — capture it and everything after the comma
            m = re.match(r'^"([^"]+)",(.*)', line)
            if m:
                rows.append({'WKT': m.group(1), 'name': m.group(2)})

    station_df = pd.DataFrame(rows)
    station_df[['name', 'description']] = station_df['name'].str.split(',', n=1, expand=True)
    station_df = station_df.drop(columns=['description'])
    station_df['geometry'] = station_df['WKT'].apply(wkt.loads)
    station_df['lon'] = station_df['geometry'].apply(lambda x: x.x)
    station_df['lat'] = station_df['geometry'].apply(lambda x: x.y)
    station_df['alt'] = get_elevation_batch(list(zip(station_df['lat'].values, station_df['lon'].values)))
    grid_ctr_lat = station_df['lat'].mean()
    grid_ctr_lon = station_df['lon'].mean()
    grid_ctr_alt = station_df['alt'].mean()
    geosys = coords.GeographicSystem()
    tpcs = coords.TangentPlaneCartesianSystem(ctrLat=grid_ctr_lat, ctrLon=grid_ctr_lon, ctrAlt=grid_ctr_alt)
    station_X, station_Y, station_Z = geosys.toECEF(station_df['lon'].values, station_df['lat'].values, station_df['alt'].values)
    station_x, station_y, station_z = tpcs.fromECEF(station_X, station_Y, station_Z)
    station_df['x'] = station_x
    station_df['y'] = station_y
    station_df['z'] = station_z
    # step 1 -- create grids of X, Y, Z, Q in shape (Nx, Ny, Nz)
    x_vals = np.linspace(-L_x/2, L_x/2, n_x)*1e3
    y_vals = np.linspace(-L_y/2, L_y/2, n_y)*1e3
    z_vals = np.linspace(0, L_z, n_z)*1e3
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals)

    x_grid_flat, y_grid_flat, z_grid_flat = x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
    q = np.full_like(x_grid, Q_coloumbs).flatten()
    # step 2 -- calculate "perfect" delta E at station locations shape Nx*Ny*Nz, N_stations
    delta_e_ideal = monopole_E_change(x_grid_flat, y_grid_flat, z_grid_flat, q, station_df['x'].values, station_df['y'].values, station_df['z'].values)
    # step 3 -- add gaussian noise to the delta E values to simulate instrument error
    delta_e_with_error = add_error_to_monopoles(delta_e_ideal, error_scale=error_scale)
    print(delta_e_with_error.shape)
    # step 4 -- use noisy delta E values to retrieve x, y, z, and q of shape (Nx, Ny, Nz)
    retrieved_x, retrieved_y, retrieved_z, retrieved_q = multi_monopole_retrieval(delta_e_with_error, station_df, initial_guess_q=0)


    # store all actual and retrieved values in dictionaries
    retrieved_x = retrieved_x.reshape(x_grid.shape)
    retrieved_y = retrieved_y.reshape(x_grid.shape)
    retrieved_z = retrieved_z.reshape(x_grid.shape)
    retrieved_q = retrieved_q.reshape(x_grid.shape)
    retrieved_all = np.array([retrieved_x, retrieved_y, retrieved_z, retrieved_q])
