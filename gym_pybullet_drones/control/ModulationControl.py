"""---------------------------------------------------------------------
Figueroa Robotics Lab
---------------------------------------------------------------------"""
import numpy as np
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

from modulation import obs_avoidance_interpolation_moving

class ModulationXYControl(BaseControl):
    """Modulation class for control on xyz 3D space.

    Modified from https://github.com/penn-figueroa-lab/ros_obstacle_avoidance/tree/main.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        #### Set general use constants #############################
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in ModulationControl.__init__(), ModulationControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        
        self.dt = 0.02

        self.d_obst_num = 0

        self.velocity_limit = None
        self.speed_thr = 2

        self.convex = True

        self.margin = 0.7
        self.c = 0.6
        self.b = 0.02

        self.reset()

    ################################################################################

    def reset(self):
        """Reset the control classes.

        A general use counter is set to zero.

        """
        super().reset()

    ################################################################################

    def _compute_orientation_subtraction(self, rad_start, rad_end):
        if abs(rad_start) > np.pi or abs(rad_end) > np.pi:
            print("orientation reduction error!!")

            while rad_end > np.pi:
                rad_end = rad_end - 2*np.pi
            while rad_end < -np.pi:
                rad_end = rad_end + 2*np.pi
            while rad_start > np.pi:
                rad_start = rad_start - 2*np.pi
            while rad_start < -np.pi:
                rad_start = rad_start + 2*np.pi

        difference = rad_end - rad_start
        if abs(difference) > np.pi:
            difference = -np.sign(difference)*(2*np.pi-abs(difference))

        return difference

    ################################################################################

    def _compute_orientation(self, v_xy, negative_velocity = False):
        if negative_velocity == True:
            v_xy = -v_xy

        if v_xy[0]:
            if v_xy[0] > 0:
                theta = np.arctan(v_xy[1]/v_xy[0])
            else:
                theta = np.pi + np.arctan(v_xy[1]/v_xy[0])

            if theta > np.pi:
                theta = theta - 2*np.pi
        else:
            theta = np.sign(v_xy[1])*np.pi/2

        return theta

    ################################################################################

    def _pos_global_to_relative(self, pos_rob, pos_obst, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return rotation_matrix.T @ (pos_rob - pos_obst)

    ################################################################################

    def _pos_relative_to_global(self, pos_rel, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return rotation_matrix @ pos_rel 

    ################################################################################

    def _grad_pos_h(self, cur_pos_xy, obst_pos_xy, obst_orit, convex=False, return_rel=False):
        dh_dpos = np.zeros((self.d_obst_num, 2))
        dh_dpos_rel = np.zeros((self.d_obst_num, 2))

        for i in range(self.d_obst_num):
            pos = self._pos_global_to_relative(cur_pos_xy, obst_pos_xy[i], obst_orit[i])
            if convex:
                dist = np.sqrt(pos[0]**2 + pos[1]**2)
                dh_dpos_rel = [pos[0]/dist, pos[1]/dist]
            elif self.d_obst_num == 1:
                dh_dpos_rel = [4*pos[0]**3 - 10*pos[0], 4*pos[1]**3]
            else:
                dist = np.sqrt(np.power((pos[0]**2 - self.c)**2 + pos[1]**4, 3/4))
                dh_dpos_rel = [0.25/dist*(4*pos[0]**3 - 2*self.c*pos[0]), 0.25/dist*4*pos[1]**3]

            if return_rel:
                dh_dpos[i] = dh_dpos_rel
            else:
                dh_dpos[i] = self._pos_relative_to_global(dh_dpos_rel, obst_orit[i])

        return dh_dpos

    ################################################################################

    def _h(self, cur_pos_xy, obst_pos_xy, obst_orit, convex=True):
        h = np.zeros((self.d_obst_num, ))
        for i in range(self.d_obst_num):
            pos = self._pos_global_to_relative(cur_pos_xy, obst_pos_xy[i], obst_orit[i])
            if convex:
                h[i] = np.sqrt(pos[0]**2 + pos[1]**2) - 1 - self.margin
            else:
                h[i] = np.power((pos[0]**2 - self.c)**2 + pos[1]**4, 1/4) - np.power(self.c**2 + self.b, 1/4)
        return h

    ################################################################################

    def _nominal_ctrl(self, cur_pos, target_pos):
        """Control the velocity direction

        a - any matrix with negative eigenvalues

        """
        a = np.array([[-1, 0],[0, -1]])
        vel = a @ (cur_pos - target_pos)

        if np.linalg.norm(vel) > self.speed_thr:
            return self.speed_thr * vel / np.linalg.norm(vel)
        
        return vel

    ################################################################################

    def _modulationXY(self, cur_pos_xy, cur_yaw, obst_pos_xy, obst_orit, obst_vel_xy, obst_ang_vel, target_pos_xy):
        """
        Parameters
        ----------
        cur_pos_xy : ndarray 
            (2,1)-shaped array of floats containing the current x, y position.
        cur_yaw : float
            float containing the current yaw rotation (orientation).
        obst_pos_xy : ndarray
            (2,1)-shaped array of floats containing the obstacles x, y position.
        obst_orit : float
            float containing the obstacles orientation.
        obst_vel_xy : ndarray
            (2,1)-shaped array of floats containing the obstacles x, y velocity.
        obst_ang_vel : float
            float containing the obstacles angular velocity.
        target_pos : ndarray
            (2,1)-shaped array of floats containing the desired x, y position.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        float
            The target yaw rate (angular velocity).
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The traget yaw value.

        """

        u_nom = self._nominal_ctrl(cur_pos_xy, target_pos_xy)

        velocity, _= obs_avoidance_interpolation_moving(
				position = np.array(cur_pos_xy),
				initial_velocity = u_nom,
				Gamma = self._h(cur_pos_xy, obst_pos_xy, obst_orit, convex=self.convex) +1,
				dhdx = self._grad_pos_h(cur_pos_xy, obst_pos_xy, obst_orit, convex=self.convex, return_rel=False),
				obs_vel = obst_vel_xy,
				obs_angular_velocity = obst_ang_vel,
				obs_center = obst_pos_xy,
				obs_orientation = obst_orit,
				velocity_limit = self.velocity_limit,
	    )
        rob_vel = np.linalg.norm(velocity)

        position = velocity * self.dt + cur_pos_xy

        target_yaw = self._compute_orientation(velocity, negative_velocity = False)
        rob_ang_vel = self._compute_orientation_subtraction(cur_yaw, target_yaw) / self.dt
        
        return rob_vel, rob_ang_vel, position, target_yaw
    
    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Abstract method to compute the control action for a single drone.

        It must be implemented by each subclass of `BaseControl`.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        pos_e = target_pos - cur_pos

        if pos_e[2] != 0:
            print("[ERROR] in ModulationXYControl.computeControl(), ModulationXYControl only works for xy-planar control")
            exit()

        self.control_counter += 1
        cur_rpy = p.getEulerFromQuaternion(cur_quat)

        # TODO: Create dynamics obstacles list and update lab environment
        d_obstacles = []
        self.d_obst_num = d_obstacles.len()
        obst_pos = [0, 0, 0]
        obst_orit = 0
        obst_vel = [0, 0, 0]
        obst_ang_vel = 0
        
        velocity, angular_velocity, _, computed_target_yaw = self._(cur_pos, cur_rpy, obst_pos, obst_orit, obst_vel, obst_ang_vel, target_pos)

        # TODO: Need to use the velocity and angular_velocity as input of PID to
        # calculate the thrust and rpm. See https://github.com/KevinHuang8/DATT/blob/main/controllers/pid_controller.py
        rpm = [0, 0, 0, 0]

        return rpm, pos_e, computed_target_yaw - cur_rpy[2]