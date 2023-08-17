"""---------------------------------------------------------------------
Figueroa Robotics Lab
------------------------------------------------------------------------

Example     Init the env by calling FLabCtrlAviary()

Notes       `FLabCtrlAviary` environment extend 'BaseAviary' environment.
            Add customized obstacles.

------------------------------------------------------------------------
Modified from envs/sCtrlAviary.py by Qi Xue (qixue@seas.upenn.edu).
---------------------------------------------------------------------"""

import numpy as np
import pybullet as p
from gym import spaces
from robot import KUKASAKE

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class FLabCtrlAviary(BaseAviary):
    """Multi-drone environment class for control applications with Figueroa Lab enviroment"""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """           
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=False,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )

        self.obstacles_list = []
        self.obstacles_id_list = []
        if obstacles:
            self.obstacles_id_list, self.obstacles_list = self._addObstacles()

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        # samurai_id = p.loadURDF("samurai.urdf",
        #            physicsClientId=self.CLIENT
        #            )
        # duck_id = p.loadURDF("duck_vhacd.urdf",
        #            [-.5, -.5, .05],
        #            p.getQuaternionFromEuler([0, 0, 0]),
        #            physicsClientId=self.CLIENT
        #            )
        sphere_id = p.loadURDF("sphere2.urdf",
                   [.7, 0, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )

        cube1_id = p.loadURDF("cube_no_rotation.urdf",
                   [-3.5, 0, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube2_id = p.loadURDF("cube_no_rotation.urdf",
                   [-3.5, 1, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube3_id = p.loadURDF("cube_no_rotation.urdf",
                   [-3.5, 2, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube4_id = p.loadURDF("cube_no_rotation.urdf",
                   [-3.5, 3, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube5_id = p.loadURDF("cube_no_rotation.urdf",
                   [-3.5, 4, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube1t_id = p.loadURDF("cube_no_rotation.urdf",
                   [-3.5, 0, 1.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube2t_id = p.loadURDF("cube_no_rotation.urdf",
                   [-3.5, 1, 1.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube3t_id = p.loadURDF("cube_no_rotation.urdf",
                   [-3.5, 2, 1.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube4t_id = p.loadURDF("cube_no_rotation.urdf",
                   [-3.5, 3, 1.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube5t_id = p.loadURDF("cube_no_rotation.urdf",
                   [-3.5, 4, 1.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )

        cube6_id = p.loadURDF("cube_no_rotation.urdf",
                   [2.5, 0, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube7_id = p.loadURDF("cube_no_rotation.urdf",
                   [2.5, 1, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube8_id = p.loadURDF("cube_no_rotation.urdf",
                   [2.5, 2, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube9_id = p.loadURDF("cube_no_rotation.urdf",
                   [2.5, 3, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube10_id = p.loadURDF("cube_no_rotation.urdf",
                   [2.5, 4, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube6t_id = p.loadURDF("cube_no_rotation.urdf",
                   [2.5, 0, 1.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube7t_id = p.loadURDF("cube_no_rotation.urdf",
                   [2.5, 1, 1.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube8t_id = p.loadURDF("cube_no_rotation.urdf",
                   [2.5, 2, 1.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube9t_id = p.loadURDF("cube_no_rotation.urdf",
                   [2.5, 3, 1.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        cube10t_id = p.loadURDF("cube_no_rotation.urdf",
                   [2.5, 4, 1.5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )        
                   
        
        table1_id = p.loadURDF("objects/table.urdf",
                   [-1, 2, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True,
                   physicsClientId=self.CLIENT
                   )
        table2_id = p.loadURDF("objects/table.urdf",
                   [-1, 4, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   useFixedBase=True,
                   physicsClientId=self.CLIENT
                   )

        kuka1 = KUKASAKE(p, [-1, 2, .5],
                        p.getQuaternionFromEuler([0, 0, 0]))
        kuka1_id = kuka1.robot_id

        kuka2 = KUKASAKE(p, [-1, 4, .5],
                        p.getQuaternionFromEuler([0, 0, 0]))
        kuka2_id = kuka2.robot_id
    
        obstacles_id = [
            sphere_id,

            cube1_id,
            cube2_id,
            cube3_id,
            cube4_id,
            cube5_id,
            cube6_id,
            cube7_id,
            cube8_id,
            cube9_id,
            cube10_id,

            table1_id,
            table2_id,
            kuka1_id,
            kuka2_id
        ]

        obstacles = [
            p.getBasePositionAndOrientation(sphere_id),

            p.getBasePositionAndOrientation(cube1_id),
            p.getBasePositionAndOrientation(cube2_id),
            p.getBasePositionAndOrientation(cube3_id),
            p.getBasePositionAndOrientation(cube4_id),
            p.getBasePositionAndOrientation(cube5_id),
            p.getBasePositionAndOrientation(cube6_id),
            p.getBasePositionAndOrientation(cube7_id),
            p.getBasePositionAndOrientation(cube8_id),
            p.getBasePositionAndOrientation(cube9_id),
            p.getBasePositionAndOrientation(cube10_id),

            p.getBasePositionAndOrientation(table1_id),
            p.getBasePositionAndOrientation(table2_id),
            p.getBasePositionAndOrientation(kuka1_id),
            p.getBasePositionAndOrientation(kuka2_id)
        ]

        return obstacles_id, obstacles
    
    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([0.,           0.,           0.,           0.])
        act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Dict({str(i): spaces.Box(low=act_lower_bound,
                                               high=act_upper_bound,
                                               dtype=np.float32
                                               ) for i in range(self.NUM_DRONES)})
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
                                                                     high=obs_upper_bound,
                                                                     dtype=np.float32
                                                                     ),
                                                 "neighbors": spaces.MultiBinary(self.NUM_DRONES)
                                                 }) for i in range(self.NUM_DRONES)})

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        adjacency_mat = self._getAdjacencyMatrix()
        return {str(i): {"state": self._getDroneStateVector(i), "neighbors": adjacency_mat[i, :]} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        clipped_action = np.zeros((self.NUM_DRONES, 4))
        for k, v in action.items():
            clipped_action[int(k), :] = np.clip(np.array(v), 0, self.MAX_RPM)
        return clipped_action

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
