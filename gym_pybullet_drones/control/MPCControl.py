"""---------------------------------------------------------------------
Figueroa Robotics Lab
---------------------------------------------------------------------"""
import os
import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel

import numpy as np
import control
import cvxpy as cp
from math import sin, cos, atan2

class MPCControl(BaseControl):
    """MPC class for control.

    Modified from https://github.com/TylerReimer13/6DOF_Quadcopter_MPC/tree/main.

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
            print("[ERROR] in MPCControl.__init__(), MPCControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        
        self.g = g
        self.Ix = 1.
        self.Iy = 1.
        self.Iz = 1.5
        self.m = self._getURDFParameter('m')

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([ [0, -1,  -1], [+1, 0, 1], [0,  1,  -1], [-1, 0, 1] ])

        #### Set MPC constants #############################
        self.DT = 0.1   # Time step
        self.N = 20     # MPC horizon length

        self.x_pos = 0.0
        self.y_pos = 0.0
        self.z_pos = 0.0
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.z_vel = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0

        self.A_zoh = np.eye(12)
        self.B_zoh = np.zeros((12, 4))

        self.states = np.array([self.roll, self.pitch, self.yaw, self.roll_rate, self.pitch_rate, self.yaw_rate,
                                self.x_vel, self.y_vel, self.z_vel, self.x_pos, self.y_pos, self.z_pos]).T

        self.reset()

    ################################################################################
    
    @property
    def A(self):
        """ Linear state transition matrix
        """
        A = np.zeros((12, 12))
        A[0, 3] = 1.
        A[1, 4] = 1.
        A[2, 5] = 1.
        A[6, 1] = -self.g
        A[7, 0] = self.g
        A[9, 6] = 1.
        A[10, 7] = 1.
        A[11, 8] = 1.
        return A
    
    ################################################################################
    
    @property
    def B(self):
        """ Control matrix
        """
        B = np.zeros((12, 4))
        B[3, 1] = 1/self.Ix
        B[4, 2] = 1/self.Iy
        B[5, 3] = 1/self.Iz
        B[8, 0] = 1/self.m
        return B

    ################################################################################
    
    @property
    def C(self):
        C = np.eye(12)
        return C

    ################################################################################
    
    @property
    def D(self):
        D = np.zeros((12, 4))
        return D

    ################################################################################
    
    @property
    def Q(self):
        # State cost
        Q = np.eye(12)
        Q[8, 8] = 5.  # z vel
        Q[9, 9] = 10.  # x pos
        Q[10, 10] = 10.  # y pos
        Q[11, 11] = 100.  # z pos
        return Q

    ################################################################################
    
    @property
    def R(self):
        # Actuator cost
        R = np.eye(4)*.001
        return R

    ################################################################################
    
    def zoh(self):
        """ Convert continuous time dynamics into discrete time
        """
        sys = control.StateSpace(self.A, self.B, self.C, self.D)
        sys_discrete = control.c2d(sys, self.DT, method='zoh')

        self.A_zoh = np.array(sys_discrete.A)
        self.B_zoh = np.array(sys_discrete.B)

    ################################################################################
    
    def run_mpc(self, xr):
        N = self.N
        cost = 0.
        constr = [x[:, 0] == x_init]
        for t in range(N):
            cost += cp.quad_form(xr - x[:, t], self.Q) + cp.quad_form(u[:, t], self.R)  # Linear Quadratic cost
            constr += [xmin <= x[:, t], x[:, t] <= xmax]  # State constraints
            constr += [x[:, t + 1] == self.A_zoh * x[:, t] + self.B_zoh * u[:, t]]

        cost += cp.quad_form(x[:, N] - xr, self.Q)  # End of trajectory error cost
        problem = cp.Problem(cp.Minimize(cost), constr)
        return problem

    ################################################################################

    def reset(self):
        """Reset the control classes.

        A general use counter is set to zero.

        """
        self.control_counter = 0

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
        self.control_counter += 1
        pos_e = target_pos - cur_pos

        x_pos = cur_pos[0]
        y_pos = cur_pos[1]
        z_pos = cur_pos[2]
        x_vel = cur_vel[0]
        y_vel = cur_vel[1]
        z_vel = cur_vel[2]
        roll  = cur_quat[0]
        pitch = cur_quat[1]
        yaw   = cur_quat[2]
        roll_rate  = cur_ang_vel[0]
        pitch_rate = cur_ang_vel[1]
        yaw_rate   = cur_ang_vel[2]

        x_pos_target = target_pos[0]
        y_pos_target = target_pos[1]
        z_pos_target = target_pos[2]

        # Inital solver states
        x0 = np.array([roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate,
                       x_vel, y_vel, z_vel, x_pos, y_pos, z_pos])

        # Desired states
        xr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, x_pos_target, y_pos_target, z_pos_target])

        # Convex optimization solver variables
        [nx, nu] = self.B.shape
        x = cp.Variable((nx, N+1))
        u = cp.Variable((nu, N))
        x_init = cp.Parameter(nx)

        # Run optimization for N horizons
        prob = quad.run_mpc(xr)

        # Solve convex optimization problem
        x_init.value = x0
        prob.solve(solver=cp.OSQP, warm_start=True)
        x0 = quad.A_zoh.dot(x0) + quad.B_zoh.dot(u[:, 0].value)

        thrust = self.GRAVITY
        target_torque = u[0, 0].value
        states = self._updateStates(thrust + u[0, 0].value, u[1, 0].value, u[2, 0].value, u[3, 0].value)

        computed_target_rpy = states[0:3]
        target_rpy_rates = states[3:6]

        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

        ################################################################################

        def _updateStates(self, ft, tx, ty, tz):
            roll_ddot = ((self.Iy - self.Iz) / self.Ix) * (self.pitch_rate * self.yaw_rate) + tx / self.Ix
            pitch_ddot = ((self.Iz - self.Ix) / self.Iy) * (self.roll_rate * self.yaw_rate) + ty / self.Iy
            yaw_ddot = ((self.Ix - self.Iy) / self.Iz) * (self.roll_rate * self.pitch_rate) + tz / self.Iz
            x_ddot = -(ft/self.m) * (sin(self.roll) * sin(self.yaw) + cos(self.roll) * cos(self.yaw) * sin(self.pitch))
            y_ddot = -(ft/self.m) * (cos(self.roll) * sin(self.yaw) * sin(self.pitch) - cos(self.yaw) * sin(self.roll))
            z_ddot = -1*(self.g - (ft/self.m) * (cos(self.roll) * cos(self.pitch)))

            DT = self.DT

            self.roll_rate += roll_ddot * DT
            self.roll += self.roll_dot * DT

            self.pitch_rate += pitch_ddot * DT
            self.pitch += self.pitch_dot * DT

            self.yaw_rate += yaw_ddot * DT
            self.yaw += self.yaw_dot * DT

            self.x_vel += x_ddot * DT
            self.x_pos += self.x_dot * DT

            self.y_vel += y_ddot * DT
            self.y_pos += self.y_dot * DT

            self.z_vel += z_ddot * DT
            self.z_pos += self.z_dot * DT

            self.states = np.array([self.roll, self.pitch, self.yaw, self.roll_rate, self.pitch_rate, self.yaw_rate,
                                self.x_vel, self.y_vel, self.z_vel, self.x_pos, self.y_pos, self.z_pos]).T

        return states

        