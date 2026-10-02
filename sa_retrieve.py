#!/usr/bin/env python3
# Support functions for slow antenna charge retrieval
# Created 29 April 2026 by Sam Gardner <samuel.gardner@ttu.edu>


import numpy as np
from scipy.optimize import least_squares
from functools import partial


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


def monopole_delta_E_error(params, observed_E, station_info):
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


def monopole_charge_retrieval(initial_guess_q, initial_guess_x, initial_guess_y, initial_guess_z, stroke_obs, station_df, constrain_xyz=False):
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
    constrain_xyz : bool, optional
        If True, require that the solution use the provided initial guesses for x, y, and z (solving only for q). Default is False.
    """
    z_min = station_df['z'].min()
    try:
        if constrain_xyz:
            prefilled = partial(monopole_delta_E_error, x=initial_guess_x, y=initial_guess_y, z=initial_guess_z)
            retrieved_opt = least_squares(prefilled, x0=np.array([initial_guess_q]), bounds=([-np.inf], [np.inf]), args=(stroke_obs, station_df)).x
        else:
            retrieved_opt = least_squares(monopole_delta_E_error,
                        x0=np.array([initial_guess_q, initial_guess_x, initial_guess_y, initial_guess_z]),
                        bounds=([-np.inf, -np.inf, -np.inf, z_min], [np.inf, np.inf, np.inf, np.inf]),
                        args=(stroke_obs,
                        station_df)).x
    except ValueError as e:
                print(f"Initial Guess: q={initial_guess_q}, x={initial_guess_x}, y={initial_guess_y}, z={initial_guess_z}")
                print(f"Observations: {stroke_obs}")
                print(f"Error: {e}")
                return np.array([np.nan, np.nan, np.nan, np.nan])
    
    return retrieved_opt

def multi_monopole_retrieval_dask(all_monopoles, station_df, initial_guess_q=1, chunk_size=50):
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
    import dask.array as da
    this_charge_obs = da.from_array(all_monopoles, chunks=(chunk_size, -1)) # observations for all charges, shape (N_strokes, N_stations)
    station_largest_ob_idx = np.argmax(np.abs(all_monopoles), axis=1)
    initial_guess = da.from_array(station_df[['x', 'y', 'z']].to_numpy()[station_largest_ob_idx], chunks=(chunk_size, -1)) # station x, y, z for each initial guess, shape (N_strokes, 3)
    # retrieve the charge parameters for every stroke in parallel using the charge retrieval function
    all_retrievals = da.apply_gufunc(partial(monopole_charge_retrieval, station_df=station_df), '(),(),(),(),(n)->(p)',
                                        initial_guess_q, # initial_guess_q
                                        initial_guess[:, 0], # initial_guess_x
                                        initial_guess[:, 1], # initial_guess_y
                                        initial_guess[:, 2] + 1000, # initial_guess_z, 1km above station location
                                        this_charge_obs, # stroke_obs
                                        vectorize=True, output_dtypes=float, output_sizes={'p': 4}).persist()
    retrieved_x = all_retrievals[:, 1]
    retrieved_y = all_retrievals[:, 2]
    retrieved_z = all_retrievals[:, 3]
    retrieved_q = all_retrievals[:, 0]
    return retrieved_x, retrieved_y, retrieved_z, retrieved_q