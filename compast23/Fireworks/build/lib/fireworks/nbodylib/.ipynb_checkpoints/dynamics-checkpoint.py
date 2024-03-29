"""
====================================================================================================================
Collection of functions to estimate the Gravitational forces and accelerations (:mod:`fireworks.nbodylib.dynamics`)
====================================================================================================================

This module contains a collection of functions to estimate acceleration due to
gravitational  forces.

Each method implemented in this module should follow the input-output structure show for the
template function  :func:`~acceleration_estimate_template`:

Every function needs to have two input parameters:

    - particles, that is an instance of the class :class:`~fireworks.particles.Particles`
    - softening, it is the gravitational softening. The parameters need to be included even
        if the function is not using it. Use a default value of 0.

The function needs to return a tuple containing three elements:

    - acc, the acceleration estimated for each particle, it needs to be a Nx3 numpy array,
        this element is mandatory it cannot be 0.
    - jerk, time derivative of the acceleration, it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx3 numpy array.
    - pot, gravitational potential at the position of each particle. it is an optional value, if the method
        does not estimate it, just set this element to None. If it is not None, it has
        to be a Nx1 numpy array.


"""
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from ..particles import Particles,Particles_tf
import tensorflow as tf
from tqdm import tqdm
tf.config.optimizer.set_jit(True)
from numba import prange, njit

try:
    import pyfalcon
    pyfalcon_load=True
except:
    pyfalcon_load=False

def acceleration_estimate_template(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    This an empty functions that can be used as a basic template for
    implementing the other functions to estimate the gravitational acceleration.
    Every function of this kind needs to have two input parameters:

        - particles, that is an instance of the class :class:`~fireworks.particles.Particles`
        - softening, it is the gravitational softening. The parameters need to be included even
          if the function is not using it. Use a default value of 0.

    The function needs to return a tuple containing three elements:

        - acc, the acceleration estimated for each particle, it needs to be a Nx3 numpy array,
            this element is mandatory it cannot be 0.
        - jerk, time derivative of the acceleration, it is an optional value, if the method
            does not estimate it, just set this element to None. If it is not None, it has
            to be a Nx3 numpy array.
        - pot, gravitational potential at the position of each particle. it is an optional value, if the method
            does not estimate it, just set this element to None. If it is not None, it has
            to be a Nx1 numpy array.

    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None
    """

    acc  = np.zeros(len(particles))
    jerk = None
    pot = None

    return (acc,jerk,pot)

def acceleration_jerk_direct(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None
    """
    N = len(particles)
    
    acc = np.zeros((N,3))

    for i in range(N):
        for j in range(i+1,N):
            #distances between particles i and j
            dx = particles.pos[i,0]-particles.pos[j,0]
            dy = particles.pos[i,1]-particles.pos[j,1]
            dz = particles.pos[i,2]-particles.pos[j,2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            acc_ij = np.zeros(3)
            acc_ij[0] = dx/r**3
            acc_ij[1] = dy/r**3
            acc_ij[2] = dz/r**3

            #add accelleration to bodies
            for k in range(3):
                acc[i,k] -= acc_ij[k]*particles.mass[j]
                acc[j,k] += acc_ij[k]*particles.mass[i]

            #calculating relative velocities
            vx = particles.vel[i,0]-particles.vel[j,0]
            vy = particles.vel[i,1]-particles.vel[j,1]
            vz = particles.vel[i,2]-particles.vel[j,2]

            jerk_ij = np.zeros(3)
            jerk_ij[0] = -((vx / (r ** 3 + softening ** 3)) - (3 * (np.dot(dx, vx) * dx) / (r ** 5 + softening ** 5)))
            jerk_ij[1] = -((vy / (r ** 3 + softening ** 3)) - (3 * (np.dot(dy, vy) * dy) / (r ** 5 + softening ** 5)))
            jerk_ij[2] = -((vz / (r ** 3 + softening ** 3)) - (3 * (np.dot(dz, vz) * dz) / (r ** 5 + softening ** 5)))


            for k in range(3):
                jerk[i,k] += jerk_ij[k]*particles.mass[j]
                jerk[j,k] -= jerk_ij[k]*particles.mass[i]


    acc  = acc
    jerk = jerk
    pot = None

    return (acc,jerk,pot)

def acceleration_direct(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None
    """
    N = len(particles)

    acc = np.zeros((N,3))

    for i in range(N):
        for j in range(i+1,N):
            #distances between particles i and j
            dx = particles.pos[i,0]-particles.pos[j,0]
            dy = particles.pos[i,1]-particles.pos[j,1]
            dz = particles.pos[i,2]-particles.pos[j,2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            acc_ij = np.zeros(3)
            acc_ij[0] = dx / (r ** 3 + softening ** 3)
            acc_ij[1] = dy / (r ** 3 + softening ** 3)
            acc_ij[2] = dz / (r ** 3 + softening ** 3)

            #add accelleration to bodies
            for k in range(3):
                acc[i,k] -= acc_ij[k]*particles.mass[j]
                acc[j,k] += acc_ij[k]*particles.mass[i]

    acc  = acc
    jerk = None
    pot = None

    return (acc,jerk,pot)

def acceleration_direct_vectorized(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None
    """


    x = particles.pos[:,0:1]
    y = particles.pos[:,1:2]
    z = particles.pos[:,2:3]

    dx = x.T-x
    dy = y.T-y
    dz = z.T-z

    r = np.sqrt(dx**2+dy**2+dz**2)

    acc_x = np.sum(np.nan_to_num(dx * particles.mass / (r ** 3 + softening ** 3)), axis=1)
    acc_y = np.sum(np.nan_to_num(dy * particles.mass / (r ** 3 + softening ** 3)), axis=1)
    acc_z = np.sum(np.nan_to_num(dz * particles.mass / (r ** 3 + softening ** 3)), axis=1)
    acc = zip(acc_x,acc_y,acc_z)

    acc  = np.array(list(acc))
    jerk = None
    pot = None

    return (acc,jerk,pot)


def acceleration_pyfalcon(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    Estimate the acceleration following the fast-multipole gravity Dehnen2002 solver (https://arxiv.org/pdf/astro-ph/0202512.pdf)
    as implementd in pyfalcon (https://github.com/GalacticDynamics-Oxford/pyfalcon)

    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - Acceleration: a NX3 numpy array containing the acceleration for each particle
        - Jerk: None, the jerk is not estimated
        - Pot: a Nx1 numpy array containing the gravitational potential at each particle position
    """

    if not pyfalcon_load: return ImportError("Pyfalcon is not available")

    acc, pot = pyfalcon.gravity(particles.pos,particles.mass,softening)
    jerk = None

    return acc, jerk, pot
###################################### TensorFlow ############################################################
def acceleration_tf(particles: Particles_tf, softening: float = 0.) \
        -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
    """
    Estimate the acceleration using TensorFlow operations.

    :param particles: An instance of the class Particles_tf
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - acc, a Nx3 TensorFlow tensor containing the acceleration for each particle
        - jerk, None, the jerk is not estimated
        - pot, None, the potential is not estimated
    """
    pos = particles.pos
    mass = particles.mass

    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    # print(f'datatpye inside acc_tf is {type(x)}')
    dx = tf.transpose(x) - x
    dy = tf.transpose(y) - y
    dz = tf.transpose(z) - z

    r_squared = dx ** 2 + dy ** 2 + dz ** 2
    # Avoid division by zero by adding a small epsilon value
    epsilon = tf.constant(1e-10, dtype=tf.float64)
    r_inv_cube = tf.math.rsqrt(r_squared + epsilon) ** 3

    acc_x = tf.reduce_sum(dx * r_inv_cube * mass, axis=1)
    acc_y = tf.reduce_sum(dy * r_inv_cube * mass, axis=1)
    acc_z = tf.reduce_sum(dz * r_inv_cube * mass, axis=1)
    acc = tf.stack([acc_x, acc_y, acc_z], axis=1)

    jerk = None
    pot = None

    return acc, jerk, pot

from tqdm import tqdm

def acceleration_tf_batch(particles: Particles_tf, batch_size: int, softening: float = 0.) \
        -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
    pos = particles.pos
    mass = particles.mass
    num_particles = tf.shape(pos)[0]
    num_batches = tf.cast(tf.math.ceil(num_particles / batch_size), tf.int64)

    acc_list = []
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start = i * batch_size
        end = tf.minimum((i + 1) * batch_size, num_particles)

        pos_batch = pos[start:end]
        mass_batch = mass[start:end]

        dx = tf.transpose(pos_batch[:, 0:1]) - pos_batch[:, 0:1]
        dy = tf.transpose(pos_batch[:, 1:2]) - pos_batch[:, 1:2]
        dz = tf.transpose(pos_batch[:, 2:3]) - pos_batch[:, 2:3]

        r_squared = dx ** 2 + dy ** 2 + dz ** 2
        epsilon = tf.constant(1e-10, dtype=tf.float64)
        r_inv_cube = tf.math.rsqrt(r_squared + epsilon) ** 3

        acc_x = tf.reduce_sum(dx * r_inv_cube * mass_batch, axis=1)
        acc_y = tf.reduce_sum(dy * r_inv_cube * mass_batch, axis=1)
        acc_z = tf.reduce_sum(dz * r_inv_cube * mass_batch, axis=1)
        acc = tf.stack([acc_x, acc_y, acc_z], axis=1)
        acc_list.append(acc)

    acc = tf.concat(acc_list, axis=0)

    jerk = None
    pot = None

    return acc, jerk, pot

##################################### NUMBA JIT ##################################################################
@njit
def acceleration_direct_njit(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None
    """
    pos = particles.pos
    mass = particles.mass
    N = len(particles)

    acc = np.zeros_like(pos)

    for i in range(N):
        for j in range(i+1,N):
            #distances between particles i and j
            dx = pos[i,0]-pos[j,0]
            dy = pos[i,1]-pos[j,1]
            dz = pos[i,2]-pos[j,2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            acc_ij = np.zeros(3)
            acc_ij[0] = dx / (r ** 3 + softening ** 3)
            acc_ij[1] = dy / (r ** 3 + softening ** 3)
            acc_ij[2] = dz / (r ** 3 + softening ** 3)

            #add accelleration to bodies
            for k in range(3):
                acc[i,k] -= acc_ij[k]*mass[j]
                acc[j,k] += acc_ij[k]*mass[i]

    acc  = acc
    jerk = None
    pot = None

    return (acc,jerk,pot)


@njit(parallel=True)
def acceleration_direct_njit_parallel(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None
    """
    pos = particles.pos
    mass = particles.mass
    N = len(particles)

    acc = np.zeros_like(pos)

    for i in prange(N):
        for j in range(i+1,N):
            #distances between particles i and j
            dx = pos[i,0]-pos[j,0]
            dy = pos[i,1]-pos[j,1]
            dz = pos[i,2]-pos[j,2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            acc_ij = np.zeros(3)
            acc_ij[0] = dx / (r ** 3 + softening ** 3)
            acc_ij[1] = dy / (r ** 3 + softening ** 3)
            acc_ij[2] = dz / (r ** 3 + softening ** 3)

            #add accelleration to bodies
            for k in range(3):
                acc[i,k] -= acc_ij[k]*mass[j]
                acc[j,k] += acc_ij[k]*mass[i]

    acc  = acc
    jerk = None
    pot = None

    return (acc,jerk,pot)

@njit
def acceleration_direct_vectorized_njit(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64],Optional[npt.NDArray[np.float64]],Optional[npt.NDArray[np.float64]]]:
    """
    :param particles: An instance of the class Particles
    :param softening: Softening parameter
    :return: A tuple with 3 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle
        - jerk, Nx3 numpy array storing the time derivative of the acceleration, can be set to None
        - pot, Nx1 numpy array storing the potential at each particle position, can be set to None
    """
    pos = particles.pos
    mass = particles.mass

    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    dx = x.copy().T-x
    dy = y.copy().T-y
    dz = z.copy().T-z

    r = np.sqrt(dx**2+dy**2+dz**2)

    acc_x = np.sum(np.nan_to_num(dx * mass / (r ** 3 + softening ** 3)), axis=1)
    acc_y = np.sum(np.nan_to_num(dy * mass / (r ** 3 + softening ** 3)), axis=1)
    acc_z = np.sum(np.nan_to_num(dz * mass / (r ** 3 + softening ** 3)), axis=1)
    acc = zip(acc_x,acc_y,acc_z)

    acc  = np.array(list(acc))
    jerk = None
    pot = None

    return (acc,jerk,pot)
