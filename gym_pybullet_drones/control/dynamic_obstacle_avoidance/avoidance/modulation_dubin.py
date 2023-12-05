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

def get_orthogonal_basis(position,obs_center,dhdx):
  v = np.array(-obs_center+position)
  v_norm = np.linalg.norm(v)
  dhdx_norm = np.linalg.norm(dhdx)
  assert v_norm > 0, 'v must be non-zero'
  assert dhdx_norm > 0, 'dhdx must be non-zero'
  v = v / v_norm
  dhdx = dhdx/dhdx_norm
  basis = np.zeros((2, 2))
  basis_orth = np.zeros((2, 2))
  basis_orth[:, 0] = dhdx
  basis_orth[:, 1] = [dhdx[1], -dhdx[0]]
  basis[:, 0] = v
  basis[:, 1] = [dhdx[1], -dhdx[0]]

  return basis, basis_orth


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



def transform_global2relative(position, center_position, dim, orientation=None):
    """Transform a position from the global frame of reference
    to the obstacle frame of reference"""
    # TODO: transform this into wrapper / decorator
    if not position.shape[0] == dim:
        raise ValueError("Wrong position dimensions")

    if dim == 2:
        if len(position.shape) == 1:
            position = position - np.array(center_position)
            if orientation is None:
                return position
            rotation_matrix = np.array([[np.cos(orientation),-np.sin(orientation)],[np.sin(orientation), np.cos(orientation)]])
            return rotation_matrix.T.dot(position)

        elif len(position.shape) == 2:
            n_points = position.shape[1]
            position = position - np.tile(center_position, (n_points, 1)).T
            if orientation is None:
                return position
            rotation_matrix = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta), np.cos(theta)]])
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

def compute_orientation(v_xy, negative_velocity = False):
    if negative_velocity == True:
        v_xy = -v_xy
    if v_xy[0]:
        if v_xy[0]>0:
            theta = np.arctan(v_xy[1]/v_xy[0])
        else:
            theta = np.pi+np.arctan(v_xy[1]/v_xy[0])

        if theta > np.pi:
            theta = theta -2*np.pi
    else:
        theta = np.sign(v_xy[1])*np.pi/2

    return theta

def compute_orientation_subtraction(rad_start,rad_end):
    if abs(rad_start)>np.pi or abs(rad_end)>np.pi:
        obprint("orientation reduction error!!")
        while rad_end>np.pi:
            rad_end = rad_end -2*np.pi
        while rad_end<-np.pi:
            rad_end = rad_end + 2*np.pi
        while rad_start>np.pi:
            rad_start = rad_start -2*np.pi
        while rad_start<-np.pi:
            rad_start = rad_start + 2*np.pi
    difference = rad_end -rad_start
    if abs(difference)>np.pi:
        difference = -np.sign(difference)*(2*np.pi-abs(difference))
    return difference

def compute_orientation_addition(rad_start,delta_rad):
    rad_end = rad_start + delta_rad
    while rad_end>np.pi:
        rad_end = rad_end -2*np.pi
    while rad_end<-np.pi:
        rad_end = rad_end + 2*np.pi
    return rad_end

def compute_velocity(theta, v):
    vx = v*np.cos(theta)
    vy = v*np.sin(theta)
    return np.array([vx,vy])


def obs_avoidance_interpolation_moving(
    position,
    desired_velocity,
    current_velocity,
    current_orientation,
    current_angular_velocity,
    Gamma,
    dhdx,
    obs_vel,
    obs_angular_velocity,
    obs_center,
    obs_orientation,
    linear_acceleration_limit,
    angular_acceleration_limit,
    dt,
    velocity_limit=False,
    is_boundary=False,
    repulsion_coeff=1,
    tail_effect=False,
    weightPow=2,
    repulsive_gammaMargin=0.01,
    repulsive_obstacle=False,
    evaluate_in_global_frame=False,
    zero_vel_inside=True,
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
    isFailure = False
    preferred_speed = 10

    if not N_obs:  # No obstacles
        print("here")
        return desired_velocity,isFailure

    dim = len(obs_center[0])
    # print(N_obs)
    # print(dim)

    if evaluate_in_global_frame:
        pos_relative = np.tile(position, (N_obs, 1)).T

    else:
        pos_relative = np.zeros((dim, N_obs))
        for n in range(N_obs):
            # Move to obstacle centered frame
            pos_relative[:, n] = transform_global2relative(
                position=position, 
                center_position=obs_center[n], 
                dim=dim,
                orientation=obs_orientation[n],
            )

    # Worst case of being at the center
    if any(Gamma == 0):
        isFailure = True
        return np.zeros(dim), isFailure

    if zero_vel_inside and any(Gamma < 1):
        isFailure = True
        return np.zeros(dim),isFailure 

    ind_obs = Gamma < cut_off_gamma
    if any(~ind_obs):
        # print(initial_velocity)
        return desired_velocity, isFailure

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

        #!!!!!!!!!! not pos_relative, but xc and position
        E[:, :, n], E_orth[:, :, n] = get_orthogonal_basis(
            position = position,
            obs_center = obs_center[n],
            dhdx = dhdx[n,:],
            # in_global_frame=evaluate_in_global_frame,
        )

    xd_obs = get_relative_obstacle_velocity(
        position=position,
        obs_vel=obs_vel,
        obs_center=obs_center,
        obs_angular_velocity = obs_angular_velocity,
        is_boundary=is_boundary,
        E_orth=E_orth,
        gamma_list=Gamma,
        weights=weight,
    )

    # print(obs_vel)
    # print(xd_obs)
    # print()

    # Computing the relative velocity with respect to the obstacle
    initial_velocity = compute_velocity(current_orientation,current_velocity)
    relative_velocity = desired_velocity - xd_obs


    rel_velocity_norm = np.linalg.norm(relative_velocity)
    if rel_velocity_norm:
        rel_velocity_normalized = (initial_velocity-xd_obs) / rel_velocity_norm

    else:
        # Zero velocity
        return xd_obs, isFailure

    # Keep either way, since avoidance from attractor might be needed
    relative_velocity_hat = np.zeros((dim, N_obs))
    relative_velocity_hat_magnitude = np.zeros((N_obs))
    relative_velocity_direction = np.zeros(relative_velocity_hat_magnitude.shape)
    new_velocity_min = max(current_velocity + linear_acceleration_limit[0]*dt,0)
    new_velocity_max = current_velocity + linear_acceleration_limit[1]*dt
    new_velocity_range = [new_velocity_min,new_velocity_max]
    # if new_velocity_max >preferred_speed:
    #     new_velocity_backup = preferred_speed
    # else:
    #     new_velocity_backup=new_velocity_max

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
            relative_velocity_hat[:, n] = E[:, :, n].dot(stretched_velocity)
            stretched_velocity_orth = np.linalg.pinv(E_orth[:, :, n]).dot(relative_velocity_hat[:,n])
            obs_vel_E = np.linalg.pinv(E_orth[:, :, n]).dot(xd_obs)
            robot_velocity_global = relative_velocity_hat[:,n] + xd_obs

            new_orientation = compute_orientation(robot_velocity_global)
            orth_orientation = compute_orientation(dhdx[n])
            delta_curr_to_new_theta = compute_orientation_subtraction(current_orientation,new_orientation)
            delta_new_to_orth_theta = compute_orientation_subtraction(new_orientation,orth_orientation)
            delta_curr_to_orth_theta = compute_orientation_subtraction(current_orientation,orth_orientation)

            orth_angular_acceleration = 2*(delta_curr_to_orth_theta - current_angular_velocity*dt)/dt**2
            angular_acceleration = 2*(delta_curr_to_new_theta - current_angular_velocity*dt)/dt**2
            angular_acceleration_required = [angular_acceleration,orth_angular_acceleration]

            for i in range(2):

                if angular_acceleration_required[i] < angular_acceleration_limit[0]:
                    angular_acceleration = angular_acceleration_limit[0]
                elif angular_acceleration_required[i] > angular_acceleration_limit[1]:
                    angular_acceleration = angular_acceleration_limit[1]
                else:
                    angular_acceleration = angular_acceleration_required[i]

                new_delta_theta = current_angular_velocity*dt + 0.5*angular_acceleration*dt**2
                new_orientation = compute_orientation_addition(current_orientation,new_delta_theta)
                delta_new_to_orth_theta = compute_orientation_subtraction(new_orientation,orth_orientation)

                #Here maybe should be delta_theta_orth_to_new_orientation instead
                current_vxy = compute_velocity(current_orientation,current_velocity)
                current_velocity_n = np.linalg.pinv(E_orth[:, :, n]).dot(current_vxy-xd_obs)
                linear_acceleration_n = max(np.cos(delta_new_to_orth_theta)*linear_acceleration_limit[0],np.cos(delta_new_to_orth_theta)*linear_acceleration_limit[1])
                # delta = 0.5*np.sqrt(linear_acceleration_n**2*dt**2+4*linear_acceleration_n*(2*(Gamma[n]-1)+current_velocity_n[0]*dt))
                # safe_approaching_velocity = 0.5*linear_acceleration_n*dt-delta
                # breakpoint()
                s_in_t_range = current_velocity*dt + 0.5*linear_acceleration_limit[1]*dt**2
                if Gamma[n]-1-s_in_t_range >= 0:
                    safe_approaching_velocity = -np.sqrt(2*linear_acceleration_n*(Gamma[n]-1-s_in_t_range))
                else:
                    safe_approaching_velocity = np.sqrt(-2*linear_acceleration_n*(Gamma[n]-1-s_in_t_range))

                # robot_traj_vector = compute_velocity(new_orientation,1)
                # normal2_robot_traj_from_xc = obs_center[n]+[robot_traj_vector[1],-robot_traj_vector[0]]
                # beta = np.dot(normal2_robot_traj_from_xc,dhdx[n])/(np.linalg.norm(dhdx[n])*np.linalg.norm(normal2_robot_traj_from_xc)
                # if beta>np.pi/2:
                #     beta = np.pi-beta
                # linear_acceleration_n1 = linear_acceleration_limit[1]*np.sin(0.5*beta)
                # linear_acceleration_n2 = 2*(Gamma[n]*(np.sin(beta)-np.cos(beta)*np.tan(0.5*beta)) - current_velocity*dt)/dt**2
                # linear_acceleration_n = min(linear_acceleration_n1,linear_acceleration_n2)
                # safe_approaching_velocity = -np.sqrt(2*linear_acceleration_n*(Gamma[n]-1-s_in_t_range))
                
                # safe_approaching_velocity = min(safe_approaching_velocity_1,safe_approaching_velocity_2)



                if stretched_velocity_orth[0]<= safe_approaching_velocity:
                    # stretched_velocity_orth[1] = (stretched_velocity_orth[1]+obs_vel_E[1])*(safe_approaching_velocity+obs_vel_E[0])/(stretched_velocity_orth[0]+obs_vel_E[0]) - obs_vel_E[1]
                    stretched_velocity_orth[0] = safe_approaching_velocity

                # if stretched_velocity_orth[0]>= safe_approaching_velocity:
                #     # stretched_velocity_orth[1] = (stretched_velocity_orth[1]+obs_vel_E[1])*(safe_approaching_velocity+obs_vel_E[0])/(stretched_velocity_orth[0]+obs_vel_E[0]) - obs_vel_E[1]
                #     stretched_velocity_orth[0] = safe_approaching_velocity

                new_unit_velocity= compute_velocity(new_orientation,1)
                new_unit_vel_proj_2_orth = np.linalg.pinv(E_orth[:, :, n]).dot(new_unit_velocity)

                if new_unit_vel_proj_2_orth[0]:
                    new_velocity_1 = (stretched_velocity_orth[0]+obs_vel_E[0])/new_unit_vel_proj_2_orth[0]
                    new_velocity_2 = (obs_vel_E[0])/new_unit_vel_proj_2_orth[0]
                    new_velocity_safe = (safe_approaching_velocity+obs_vel_E[0])/new_unit_vel_proj_2_orth[0]

                    linear_acceleration_1 = (new_velocity_1 - current_velocity)/dt
                    linear_acceleration_2 = (new_velocity_2 - current_velocity)/dt
                    linear_acceleration_safe = (new_velocity_safe - current_velocity)/dt


                    # if new_velocity_1 < 0:
                    #     breakpoint()
                    # linear_acceleration_min = min(linear_acceleration_1,linear_acceleration_2,linear_acceleration_3)
                    # linear_acceleration_max = max(linear_acceleration_1, linear_acceleration_2,linear_acceleration_3)

                    # if (linear_acceleration_2>=linear_acceleration_3 and linear_acceleration_3> linear_acceleration_limit[1]):
                    #     if i == 1:
                    #         # breakpoint()
                    #         print("infeasible")
                    #         isFailure = True
                    #         return np.zeros(dim), isFailure

                    # elif (linear_acceleration_2<linear_acceleration_3 and linear_acceleration_3< linear_acceleration_limit[0]):
                    #     if i == 1:
                    #         # breakpoint()
                    #         print("infeasible")
                    #         isFailure = True
                    #         return np.zeros(dim), isFailure
                    # elif new_velocity_max<0:
                    #     if i == 1:
                    #         print("infeasible")
                    #         print(new_velocity_3)
                    #         print(new_velocity_2)
                    #         isFailure = True
                    #         return np.zeros(dim), isFailure
                    if new_unit_vel_proj_2_orth[0]>0 and new_velocity_safe>new_velocity_range[1]:
                        if i ==1:
                            print("infeasible 1")
                            isFailure = True
                            return np.zeros(dim), isFailure
                    elif new_unit_vel_proj_2_orth[0]<0 and new_velocity_safe<new_velocity_range[0]:
                        if i ==1:
                            print("infeasible 2")
                            print(new_velocity_safe)
                            print(new_velocity_range[0])
                            isFailure = True
                            return np.zeros(dim), isFailure

                    elif (linear_acceleration_1<=linear_acceleration_limit[1] and linear_acceleration_1>=linear_acceleration_limit[0]):
                        new_velocity = max(new_velocity_1,0)
                        break
                    else:
                        if linear_acceleration_1<linear_acceleration_limit[0]:
                            new_velocity = new_velocity_range[0]
                        elif linear_acceleration_1>linear_acceleration_limit[1]:
                            new_velocity = new_velocity_range[1]
                        else:
                            breakpoint()
                        break
                else:
                    new_velocity = new_velocity_range[1]
                    breakpoint()

            if new_velocity==0:
                if new_unit_vel_proj_2_orth[0]>0:
                    # if current_velocity > new_velocity_safe:
                    print("sol1")
                    #     new_velocity = current_velocity
                    # else:
                    #     print("sol2")
                    new_velocity = new_velocity_range[1]
                if new_unit_vel_proj_2_orth[0]<0:
                    print("sol2")
                    new_velocity = min(new_velocity_safe,new_velocity_range[1])
                    # if current_velocity < new_velocity_safe:
                    #     print("sol3")
                    #     new_velocity = min(new_velocity_safe,new_velocity_range[1])
                    # else:
                    #     if new_velocity_safe<new_velocity_range[1]:
                    #         print("sol4")
                    #         new_velocity = new_velocity_safe
                    #     else:
                    #         print("sol5")
                    #         new_velocity = new_velocity_range[1]
                print(new_velocity)
                # new_velocity = new_velocity_backup

            robot_velocity_global = compute_velocity(new_orientation,new_velocity)

            if (new_velocity==0):
                print("NOOOOOO!")
                # print(new_velocity)
            # _new_delta_theta = compute_orientation_subtraction(current_orientation,new_orientation)
            # _angular_acceleration = 2*(_new_delta_theta - current_angular_velocity*dt)/dt**2
            # _new_velocity = np.linalg.norm(robot_velocity_global)
            # _linear_acceleration = (_new_velocity-current_velocity)/dt

            # if(abs(_angular_acceleration)>(max(angular_acceleration_limit)+1)):
            #     print(linear_acceleration)
            #     print(angular_acceleration)

            relative_velocity_hat[:, n] = robot_velocity_global
            relative_velocity_hat_magnitude[n] = np.linalg.norm(relative_velocity_hat[:,n])
            relative_velocity_direction[n] = new_delta_theta

    # ind_nonzero = relative_velocity_hat_magnitude > 0
    # if np.sum(ind_nonzero):
    #     relative_velocity_hat_normalized[:, ind_nonzero] = relative_velocity_hat[:, ind_nonzero] / np.tile(relative_velocity_hat_magnitude[ind_nonzero], (dim, 1))

    # if rel_velocity_norm:
    #     weighted_direction = get_directional_weighted_sum(
    #         null_direction=rel_velocity_normalized,
    #         directions=relative_velocity_hat_normalized,
    #         weights=weight,
    #     )

    # else:
    #     # TODO: Better solution / smooth switching when velocity is nonzero
    #     # e.g. it could be part of the reltavie-movement
    #     weighted_direction = np.sum(
    #         np.tile(weight, (1, relative_velocity_hat_normalized.shape[0])).T
    #         * relative_velocity_hat_normalized,
    #         axis=0,
    #     )
    weighted_direction_theta = np.sum(relative_velocity_direction*weight)
    weighted_direction = compute_orientation_addition(current_orientation,weighted_direction_theta)
    relative_velocity_magnitude = np.sum(relative_velocity_hat_magnitude * weight)
    vel_final = compute_velocity(weighted_direction, relative_velocity_magnitude)

    # vel_final = relative_velocity_magnitude * weighted_direction.squeeze()
    # if np.linalg.norm(weighted_direction) >1.0:
    #     print(np.linalg.norm(weighted_direction))

    # vel_final = vel_final + xd_obs

    new_orientation = compute_orientation(vel_final)
    new_delta_theta = compute_orientation_subtraction(current_orientation,new_orientation)
    angular_acceleration = 2*(new_delta_theta - current_angular_velocity*dt)/dt**2
    new_velocity = np.linalg.norm(vel_final)
    linear_acceleration = (new_velocity-current_velocity)/dt

    # if(abs(linear_acceleration)>(max(linear_acceleration_limit)+10) or abs(angular_acceleration)>(max(angular_acceleration_limit)+10)):
    #     new_orientation = compute_orientation(vel_final,True)
    #     new_delta_theta = compute_orientation_subtraction(current_orientation,new_orientation)
    #     angular_acceleration = 2*(new_delta_theta - current_angular_velocity*dt)/dt**2
    #     new_velocity = -new_velocity
    #     linear_acceleration = (new_velocity-current_velocity)/dt

    if(abs(linear_acceleration)>(max(linear_acceleration_limit)+1) or abs(angular_acceleration)>(max(angular_acceleration_limit)+1)):
        print(angular_acceleration)


    return [linear_acceleration,angular_acceleration], isFailure
