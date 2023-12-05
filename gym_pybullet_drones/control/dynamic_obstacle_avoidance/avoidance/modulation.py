"""
Library for the Modulation of Linear Systems
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021
import warnings

import numpy as np
import numpy.linalg as LA

from gym_pybullet_drones.control.vartools.directional_space import get_directional_weighted_sum
from gym_pybullet_drones.control.vartools.dynamical_systems import DynamicalSystem

from gym_pybullet_drones.control.dynamic_obstacle_avoidance.utils import get_relative_obstacle_velocity
from gym_pybullet_drones.control.dynamic_obstacle_avoidance.utils import get_orthogonal_basis
from gym_pybullet_drones.control.dynamic_obstacle_avoidance.utils import *

from .base_avoider import BaseAvoider


class ModulationAvoider(BaseAvoider):
    def __init__(
        self,
        initial_dynamics: DynamicalSystem = None,
        # convergence_system: DynamicalSystem = None,
        obstacle_environment=None,
    ):
        """Initial dynamics, convergence direction and obstacle list are used."""
        super().__init__(
            initial_dynamics=initial_dynamics, obstacle_environment=obstacle_environment
        )

        # if convergence_system is None:
        #     self.convergence_system = self.initial_dynamics
        # else:
        #     self.convergence_system = convergence_system

    def avoid(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """Obstacle avoidance based on 'local' rotation and the directional weighted mean."""
        return obs_avoidance_interpolation_moving(
            position, velocity, self.obstacle_environment
        )


def get_sticky_surface_imiation(relative_velocity, Gamma, E_orth, obs):
    # TODO: test & review sticky surface feature [!]
    relative_velocity_norm = np.linalg.norm(relative_velocity)

    if relative_velocity_norm:  # Nonzero
        # Normalize relative_velocity_hat
        mag = np.linalg.norm(relative_velocity_hat[:, n])
        if mag:  # nonzero
            relative_velocity_hat[:, n] = relative_velocity_hat[:, n] / mag

        # Limit maximum magnitude with respect to the tangent value
        sticky_surface_power = 2

        # Treat inside obstacle as on the surface
        Gamma_mag = max(Gamma[n], 1)
        if abs(Gamma[n]) < 1:
            # if abs(Gamma_mag) < 1:
            eigenvalue_magnitude = 0
        else:
            eigenvalue_magnitude = 1 - 1.0 / abs(Gamma[n]) ** sticky_surface_power
            # eigenvalue_magnitude = 1 - 1./abs(Gamma_mag)**sticky_surface_power

        if not evaluate_in_global_frame:
            relative_velocity_temp = obs[n].transform_global2relative_dir(
                relative_velocity_hat[:, n]
            )
        else:
            relative_velocity_temp = relative_velocity_hat[:, n]

        tang_vel = np.abs(E_orth[:, :, n].T.dot(relative_velocity_temp)[0])

        eigenvalue_magnitude = (
            min(eigenvalue_magnitude / tang_vel, 1) if tang_vel else 0
        )

        relative_velocity_hat[:, n] = (
            relative_velocity_hat[:, n] * relative_velocity_norm * eigenvalue_magnitude
        )

        if not evaluate_in_global_frame:
            relative_velocity_hat[:, n] = obs[n].transform_relative2global_dir(
                relative_velocity_hat[:, n]
            )


def compute_diagonal_matrix(
    Gamma,
    dim,
    is_boundary=False,
    rho=1,
    repulsion_coeff=1.0,
    tangent_eigenvalue_isometric=True,
    tangent_power=5,
    treat_obstacle_special=True,
    self_priority=1,
):
    """Compute diagonal Matrix"""
    if Gamma <= 1 and treat_obstacle_special:
        # Point inside the obstacle
        delta_eigenvalue = 1
    else:
        delta_eigenvalue = 1.0 / abs(Gamma) ** (self_priority / rho)
    eigenvalue_reference = 1 - delta_eigenvalue * repulsion_coeff

    if tangent_eigenvalue_isometric:
        eigenvalue_tangent = 1 + delta_eigenvalue
    else:
        # Decreasing velocity in order to reach zero on surface
        eigenvalue_tangent = 1 - 1.0 / abs(Gamma) ** tangent_power
    return np.diag(
        np.hstack((eigenvalue_reference, np.ones(dim - 1) * eigenvalue_tangent))
    )

def get_orthogonal_basis(v):
  v = np.array(v)
  v_norm = np.linalg.norm(v)
  assert v_norm > 0, 'v must be non-zero'
  v = v / v_norm
  basis = np.zeros((2, 2))
  basis[:, 0] = v
  basis[:, 1] = [v[1], -v[0]]
  return basis, basis


def compute_decomposition_matrix(obs, x_t, in_global_frame=False, dot_margin=0.02):
    """Compute decomposition matrix and orthogonal matrix to basis"""
    normal_vector = obs.get_normal_direction(x_t, in_global_frame=in_global_frame)
    reference_direction = obs.get_reference_direction(
        x_t, in_global_frame=in_global_frame
    )

    dot_prod = np.dot(normal_vector, reference_direction)
    if obs.is_non_starshaped and np.abs(dot_prod) < dot_margin:
        # Adapt reference direction to avoid singularities
        # WARNING: full convergence is not given anymore, but impenetrability
        if not np.linalg.norm(normal_vector):  # zero
            normal_vector = -reference_direction
        else:
            weight = np.abs(dot_prod) / dot_margin
            dir_norm = np.copysign(1, dot_prod)
            reference_direction = get_directional_weighted_sum(
                reference_direction=normal_vector,
                directions=np.vstack((reference_direction, dir_norm * normal_vector)).T,
                weights=np.array([weight, (1 - weight)]),
            )

    E_orth = get_orthogonal_basis(normal_vector, normalize=True)
    E = np.copy((E_orth))
    E[:, 0] = -reference_direction

    return E, E_orth


def transform_global2relative(position, center_position, dim, rotation_matrix=None, orientation=None):
    """Transform a position from the global frame of reference
    to the obstacle frame of reference"""
    # TODO: transform this into wrapper / decorator
    if not position.shape[0] == dim:
        raise ValueError("Wrong position dimensions")

    if dim == 2:
        if len(position.shape) == 1:
            position = position - np.array(center_position)
            if rotation_matrix is None:
                return position
            return rotation_matrix.T.dot(position)

        elif len(position.shape) == 2:
            n_points = position.shape[1]
            position = position - np.tile(center_position, (n_points, 1)).T
            if rotation_matrix is None:
                return position
            return rotation_matrix.T.dot(position)

        else:
            raise ValueError("Unexpected position-shape")

    elif dim == 3:
        if len(position.shape) == 1:
            position = position - center_position
            if orientation is None:
                return position
            return orientation.inv().apply(position)

        elif len(position.shape) == 2:
            n_points = position.shape[1]
            position = position.T - np.tile(center_position, (n_points, 1))
            if orientation is None:
                return position.T
            return orientation.inv().apply(position).T
        else:
            raise ValueError("Unexpected position shape.")

    else:
        warnings.warn(
            "Rotation for dimensions {} need to be implemented".format(dim)
        )
        return position


def obs_avoidance_interpolation_moving(
    position,
    initial_velocity,
    Gamma,
    obs_vel,
    obs_center,
    velocity_limit,
    is_boundary=False,
    repulsion_coeff=1,
    tail_effect=False,
    weightPow=2,
    repulsive_gammaMargin=0.01,
    repulsive_obstacle=False,
    evaluate_in_global_frame=False,
    zero_vel_inside=False,
    cut_off_gamma=1e6,
    tangent_eigenvalue_isometric=True,
    gamma_distance=None,
    xd=None,
    self_priority=1,
):
    """
    This function modulates the dynamical system at position x and dynamics xd
    such that it avoids all obstacles obs. It can furthermore be forced to
    converge to the attractor.

    Parameters
    ----------
    x [dim]: position at which the modulation is happening
    xd [dim]: initial dynamical system at position x
    obs [list of obstacle_class]: a list of all obstacles and their properties, which
        present in the local environment
    attractor [list of [dim]]]: list of positions of all attractors
    weightPow [int]: hyperparameter which defines the evaluation of the weight

    Return
    ------
    xd [dim]: modulated dynamical system at position x
    """
    N_obs = len(Gamma)

    if not N_obs:  # No obstacles
        print("here")
        return initial_velocity

    dim = len(obs_center[0])
    # print(N_obs)
    # print(dim)

    if evaluate_in_global_frame:
        pos_relative = np.tile(position, (N_obs, 1)).T

    else:
        pos_relative = np.zeros((dim, N_obs))
        for n in range(N_obs):
            # Move to obstacle centered frame
            pos_relative[:, n] = transform_global2relative(position=position, center_position=obs_center[n], dim=dim)

    # Two (Gamma) weighting functions lead to better behavior when agent &
    # obstacle size differs largely.
    # Gamma = np.zeros((N_obs))
    # for n in range(N_obs):
    #     Gamma[n] = obs[n].get_gamma(
    #         pos_relative[:, n], in_global_frame=evaluate_in_global_frame
    #     )

    #     if obs[n].is_boundary:
    #         pass
        # warnings.warn('Not... Artificially increasing boundary influence.')
        # Gamma[n] = pow(Gamma[n], 1.0/3.0)
        # else:
        # pass

    # Worst case of being at the center
    if any(Gamma == 0):
        return np.zeros(dim)

    if zero_vel_inside and any(Gamma < 1):
        return np.zeros(dim)

    ind_obs = Gamma < cut_off_gamma
    if any(~ind_obs):
        print("returned")
        return initial_velocity

    weight = compute_weights(Gamma, N_obs)

    # Modulation matrices
    E = np.zeros((dim, dim, N_obs))
    D = np.zeros((dim, dim, N_obs))
    E_orth = np.zeros((dim, dim, N_obs))

    for n in np.arange(N_obs)[ind_obs]:
        # x_t = obs[n].transform_global2relative(x) # Move to obstacle centered frame
        D[:, :, n] = compute_diagonal_matrix(
            Gamma[n],
            dim,
            repulsion_coeff=1,
            tangent_eigenvalue_isometric=tangent_eigenvalue_isometric,
            rho=1,
            self_priority=self_priority,
        )

        # E[:, :, n], E_orth[:, :, n] = compute_decomposition_matrix(
        #     obs[n],
        #     pos_relative[:, n],
        #     in_global_frame=evaluate_in_global_frame,
        # )
        E[:, :, n], E_orth[:, :, n] = get_orthogonal_basis(
            # obs[n],
            pos_relative[:, n],
            # in_global_frame=evaluate_in_global_frame,
        )

    xd_obs = get_relative_obstacle_velocity(
        position=position,
        obs_vel=obs_vel,
        obs_center=obs_center,
        is_boundary=is_boundary,
        E_orth=E_orth,
        gamma_list=Gamma,
        weights=weight,
    )


    # Computing the relative velocity with respect to the obstacle
    relative_velocity = initial_velocity - xd_obs

    rel_velocity_norm = np.linalg.norm(relative_velocity)
    if rel_velocity_norm:
        rel_velocity_normalized = relative_velocity / rel_velocity_norm

    else:
        # Zero velocity
        return xd_obs

    # Keep either way, since avoidance from attractor might be needed
    relative_velocity_hat = np.zeros((dim, N_obs))
    relative_velocity_hat_magnitude = np.zeros((N_obs))

    n = 0
    for n in np.arange(N_obs)[ind_obs]:
        if repulsion_coeff > 1 and E_orth[:, 0, n].T.dot(relative_velocity) < 0:
            # Only consider boundary when moving towards (normal direction)
            # OR if the object has positive repulsion-coefficient (only consider
            # it at front)
            relative_velocity_hat[:, n] = relative_velocity
            relative_velocity_hat_magnitude[n] = np.sqrt(
            np.sum(relative_velocity_hat[:, n] ** 2)
            )

        else:
            # Matrix inversion cost between O(n^2.373) - O(n^3)
            # if not evaluate_in_global_frame:
            #     relative_velocity_temp = obs[n].transform_global2relative_dir(
            #         relative_velocity
            #     )
            # else:
            #     relative_velocity_temp = np.copy(relative_velocity)

            relative_velocity_temp = np.copy(relative_velocity)
            
        # Modulation with M = E @ D @ E^-1
        #convert relative_velocity to normal to obstacle and tangent to obstacle directions
            relative_velocity_trafo = np.linalg.pinv(E[:, :, n]).dot(
                    relative_velocity_temp
            )

            if repulsion_coeff < 0:
                # Negative Repulsion Coefficient at the back of an obstacle
                if E_orth[:, 0, n].T.dot(relative_velocity) < 0:
                    # Adapt in reference direction
                    D[0, 0, n] = 2 - D[0, 0, n]

            # relative_velocity_trafo[0]>0
            elif not tail_effect and (
                (relative_velocity_trafo[0] > 0 and not is_boundary)
                or (relative_velocity_trafo[0] < 0 and is_boundary)
            ):
            #if the obstacle is moving away from the agent
                D[0, 0, n] = 1  # No effect in 'radial direction'
            stretched_velocity = D[:, :, n].dot(relative_velocity_trafo)

            speed_obs = np.linalg.norm(xd_obs)

            robot_current_velocity = E[:, :, n].dot(stretched_velocity)
            robot_obs_direction = robot_current_velocity.dot(xd_obs)

            #robot and obs going in the same direction 
            #global velocity is robot_relative_obs + speed_obs
            if(robot_obs_direction>0):
                normal_speed = abs(stretched_velocity[0])+abs(speed_obs)
            else:
                normal_speed = abs(abs(stretched_velocity[0])-abs(speed_obs))

            robot_actual_velocity = [normal_speed,stretched_velocity[1]]
            robot_actual_speed = np.linalg.norm(np.array(robot_actual_velocity))

            if (robot_actual_speed > velocity_limit):
                if (np.linalg.norm(xd_obs)>velocity_limit):
                    print("No Solution! Limit too tight.")
                    return np.zeros(dim)
                elif (1.2*abs(normal_speed)<velocity_limit):                    
                    tangent_speed= np.sqrt(velocity_limit**2-normal_speed**2)
                else:
                    for i in range(10):
                        stretched_velocity[0]=0.7*stretched_velocity[0]
                        if(robot_obs_direction>0):
                            normal_speed = abs(stretched_velocity[0])+abs(speed_obs)
                        else:
                            normal_speed = abs(abs(stretched_velocity[0])-abs(speed_obs))
                        if (abs(normal_speed)<velocity_limit):
                            tangent_speed= np.sqrt(velocity_limit**2-normal_speed**2)
                            break;

                        if(i==9):
                            print("error")
                            stretched_velocity[0]=0
                            tangent_speed = np.sqrt(velocity_limit**2-speed_obs**2)

                if stretched_velocity[1]<0:
                    stretched_velocity[1] = -tangent_speed;
                else:
                    stretched_velocity[1] = tangent_speed;

            # if (np.linalg.norm(stretched_velocity)>velocity_limit):
            #     print("line 470")
            #     print(np.linalg.norm(stretched_velocity))
            #     print()


            # if D[0, 0, n] < 0:
            #     # Repulsion in tangent direction, too, have really active repulsion
            #     factor_tangent_repulsion = 2
            #     tang_vel_norm = LA.norm(relative_velocity_trafo[1:])
            #     stretched_velocity[0] += (
            #         (-1) * D[0, 0, n] * tang_vel_norm * factor_tangent_repulsion
            #     )

            relative_velocity_hat[:, n] = E[:, :, n].dot(stretched_velocity)

            # if not evaluate_in_global_frame:
            #     relative_velocity_hat[:, n] = obs[n].transform_relative2global_dir(
            #         relative_velocity_hat[:, n]
            #     )

            # get_sticky_surface_imiation()

        # if repulsive_obstacle:
        #     # Emergency move away from center in case of a collision
        #     # Not the cleanest solution...
        #     if Gamma[n] < (1 + repulsive_gammaMargin):
        #         repulsive_power = 5
        #         repulsive_factor = 5
        #         repulsive_gamma = 1 + repulsive_gammaMargin

        #         repulsive_speed = (
        #             (repulsive_gamma / Gamma[n]) ** repulsive_power - repulsive_gamma
        #         ) * repulsive_factor
        #         if not obs[n].is_boundary:
        #             repulsive_speed *= -1

        #         pos_rel = obs[n].get_reference_direction(position, in_global_frame=True)

        #         repulsive_velocity = pos_rel * repulsive_speed
        #         relative_velocity_hat[:, n] = repulsive_velocity
            relative_velocity_hat_magnitude[n] = np.sqrt(
            np.sum(relative_velocity_hat[:, n] ** 2)
            )

            # if (relative_velocity_hat_magnitude[n]>velocity_limit):
            #     print("line 511")



    relative_velocity_hat_normalized = np.zeros(relative_velocity_hat.shape)
    ind_nonzero = relative_velocity_hat_magnitude > 0
    if np.sum(ind_nonzero):
        relative_velocity_hat_normalized[:, ind_nonzero] = relative_velocity_hat[
            :, ind_nonzero
        ] / np.tile(relative_velocity_hat_magnitude[ind_nonzero], (dim, 1))

    if rel_velocity_norm:
        weighted_direction = get_directional_weighted_sum(
            null_direction=rel_velocity_normalized,
            directions=relative_velocity_hat_normalized,
            weights=weight,
        )

    else:
        # TODO: Better solution / smooth switching when velocity is nonzero
        # e.g. it could be part of the reltavie-movement
        weighted_direction = np.sum(
            np.tile(weight, (1, relative_velocity_hat_normalized.shape[0])).T
            * relative_velocity_hat_normalized,
            axis=0,
        )

    relative_velocity_magnitude = np.sum(relative_velocity_hat_magnitude * weight)
    vel_final = relative_velocity_magnitude * weighted_direction.squeeze()

    vel_final = vel_final + xd_obs
    return vel_final
