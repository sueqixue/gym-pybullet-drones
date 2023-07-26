"""---------------------------------------------------------------------
Figueroa Robotics Lab
------------------------------------------------------------------------

Example     Init the kuka by calling KUKASAKE()

Notes       Script to include the 3D model of the kuka robot as obstacles
            in PyBullet environment.

------------------------------------------------------------------------
Implemented by Ho Jin Choi (cr139139@seas.upenn.edu).
---------------------------------------------------------------------"""
import numpy as np
from typing import List


class ROBOT(object):
    def __init__(self, bc):
        self.bc = bc
        self.robot_id: int = 0
        self.ee_id: int = 0
        self.arm_joint_ids: List[int] = [0]
        self.arm_rest_poses: List[float] = [0]
        self.gripper_link_ids: List[int] = [0]
        self.gripper_link_sign: List[float] = [1]
        self.gripper_link_limit: List[float] = [0, 1]
        self.arm_velocity: float = .35
        self.arm_force: float = 100.
        self.gripper_force: float = 20.
        self.ee_pos: List[float] = [0.537, 0.0, 0.5]
        self.ee_orn: List[float] = [0, -np.pi, 0]
        self.gripper_angle: float = 0

    def gripper_constraint(self):
        for i in range(len(self.gripper_link_ids)):
            if i != 0:
                c = self.bc.createConstraint(self.robot_id, self.gripper_link_ids[0],
                                             self.robot_id, self.gripper_link_ids[i],
                                             jointType=self.bc.JOINT_GEAR,
                                             jointAxis=[0, 1, 0],
                                             parentFramePosition=[0, 0, 0],
                                             childFramePosition=[0, 0, 0])
                gearRatio = -self.gripper_link_sign[0] * self.gripper_link_sign[i]
                self.bc.changeConstraint(c, gearRatio=gearRatio, maxForce=3, erp=1)

            gripper_link_limit = sorted([limit * self.gripper_link_sign[i] for limit in self.gripper_link_limit])
            self.bc.changeDynamics(self.robot_id, self.gripper_link_ids[i],
                                   jointLowerLimit=gripper_link_limit[0],
                                   jointUpperLimit=gripper_link_limit[1])

    def reset_arm_poses(self):
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_joint_ids):
            self.bc.resetJointState(self.robot_id, joint_id, rest_pose)
            self.bc.setJointMotorControl2(self.robot_id, joint_id, self.bc.POSITION_CONTROL,
                                          targetPosition=rest_pose, force=self.arm_force)

    def reset_gripper(self):
        for i in range(len(self.gripper_link_ids)):
            self.bc.resetJointState(self.robot_id, self.gripper_link_ids[i],
                                    self.gripper_link_limit[1] * self.gripper_link_sign[i])

    def control_gripper(self, position):
        position = np.clip(position, self.gripper_link_limit[0], self.gripper_link_limit[1])
        self.bc.setJointMotorControlArray(self.robot_id, self.gripper_link_ids, self.bc.POSITION_CONTROL,
                                          targetPositions=[i * position for i in self.gripper_link_sign],
                                          positionGains=[1 for _ in range(len(self.gripper_link_ids))],
                                          forces=[self.gripper_force for _ in range(len(self.gripper_link_ids))])

    def control_arm(self, positions):
        for position, joint_id in zip(positions, self.arm_joint_ids):
            self.bc.setJointMotorControl2(self.robot_id, joint_id, self.bc.POSITION_CONTROL,
                                          targetPosition=position, force=self.arm_force)

    def applyAction(self, motor_commands):
        self.ee_pos = np.array(self.ee_pos) + np.array(motor_commands[:3])
        self.ee_pos[0] = np.clip(self.ee_pos[0], 0.50, 0.65)
        self.ee_pos[1] = np.clip(self.ee_pos[1], -0.17, 0.22)
        self.ee_orn[2] += motor_commands[3]
        self.gripper_angle -= motor_commands[4]

        ee_orn_quaternion = self.bc.getQuaternionFromEuler(self.ee_orn)
        joint_poses = self.bc.calculateInverseKinematics(self.robot_id, self.ee_id,
                                                         self.ee_pos, ee_orn_quaternion)
        self.control_arm(joint_poses)
        self.control_gripper(self.gripper_angle)

    def get_joint_limits(self, body_id, joint_ids):
        """Query joint limits as (lo, hi) tuple, each with length same as
        `joint_ids`."""
        joint_limits = []
        for joint_id in joint_ids:
            joint_info = self.bc.getJointInfo(body_id, joint_id)
            joint_limit = joint_info[8], joint_info[9]
            joint_limits.append(joint_limit)
        joint_limits = np.transpose(joint_limits)  # Dx2 -> 2xD
        return joint_limits


# class UR5RG2(ROBOT):
#     def __init__(self, bc, pos, orn):
#         super().__init__(bc)
#         self.robot_id: int = bc.loadURDF('/robots/ur5_rg2.urdf', pos, orn,
#                                          useFixedBase=True, flags=bc.URDF_USE_SELF_COLLISION)
#         self.ee_id: int = 7  # NOTE(choi): End-effector joint ID for UR5 robot
#         self.arm_joint_ids: List[int] = [1, 2, 3, 4, 5, 6]  # NOTE(choi): Hardcoded arm joint id for UR5 robot
#         self.arm_joint_limits = self.get_joint_limits(self.robot_id, self.arm_joint_ids)
#         self.arm_rest_poses: List[float] = self.bc.calculateInverseKinematics(self.robot_id, self.ee_id,
#                                                                               self.ee_pos,
#                                                                               self.bc.getQuaternionFromEuler(self.ee_orn),
#                                                                               maxNumIterations=100)
#         self.gripper_z_offset: float = 0.221
#         self.gripper_link_ids: List[int] = [10, 11, 12, 13, 14, 15]
#         self.gripper_link_sign: List[float] = [-1, -1, 1, 1, 1, -1]
#         self.gripper_link_limit: List[float] = [0, 1.343904]
#         self.gripper_angle: float = self.gripper_link_limit[1]
#
#         # Disable closed linkage collision (gripper's inner fingers)
#         bc.setCollisionFilterPair(self.robot_id, self.robot_id, 9, 12, 0)
#         bc.setCollisionFilterPair(self.robot_id, self.robot_id, 9, 15, 0)
#         self.gripper_constraint()
#
#         self.reset_arm_poses()
#         self.reset_gripper()


class KUKASAKE(ROBOT):
    def __init__(self, bc, pos, orn):
        super().__init__(bc)
        self.robot_id: int = bc.loadURDF('/robots/iiwa7_sake.urdf', pos, orn,
                                         useFixedBase=True, flags=bc.URDF_USE_SELF_COLLISION)

        self.ee_id: int = 7  # NOTE(choi): End-effector joint ID for UR5 robot
        self.arm_joint_ids: List[int] = [0, 1, 2, 3, 4, 5, 6]  # NOTE(choi): Hardcoded arm joint id for UR5 robot
        self.arm_joint_limits = self.get_joint_limits(self.robot_id, self.arm_joint_ids)
        self.arm_rest_poses: List[float] = self.bc.calculateInverseKinematics(self.robot_id, self.ee_id,
                                                                              self.ee_pos,
                                                                              self.bc.getQuaternionFromEuler(
                                                                                  self.ee_orn),
                                                                              maxNumIterations=100)
        self.gripper_z_offset: float = 0.17
        self.gripper_link_ids: List[int] = [10, 11, 13, 14]
        self.gripper_link_sign: List[float] = [-1, 0, -1, 0]
        self.gripper_link_limit: List[float] = [-0.27, 1.57075]
        self.gripper_angle: float = self.gripper_link_limit[1]
        self.gripper_constraint()

        self.reset_arm_poses()
        self.reset_gripper()


class KUKAROBOTIQ(ROBOT):
    def __init__(self, bc, pos, orn):
        super().__init__(bc)
        self.robot_id: int = bc.loadURDF('/robots/iiwa7_robotiq.urdf', pos, orn,
                                         useFixedBase=True, flags=bc.URDF_USE_SELF_COLLISION)
        self.ee_id: int = 7  # NOTE(choi): End-effector joint ID for UR5 robot
        self.arm_joint_ids: List[int] = [0, 1, 2, 3, 4, 5, 6]  # NOTE(choi): Hardcoded arm joint id for UR5 robot
        self.arm_joint_limits = self.get_joint_limits(self.robot_id, self.arm_joint_ids)
        self.arm_rest_poses: List[float] = self.bc.calculateInverseKinematics(self.robot_id, self.ee_id,
                                                                              self.ee_pos,
                                                                              self.bc.getQuaternionFromEuler(
                                                                                  self.ee_orn),
                                                                              maxNumIterations=100)
        self.gripper_z_offset: float = 0.14
        self.gripper_link_ids: List[int] = [8, 10, 11, 12, 13, 14]
        self.gripper_link_sign: List[float] = [-1, -1, 1, 1, -1, 1]
        self.gripper_link_limit: List[float] = [-0.725, 0]
        self.gripper_angle: float = self.gripper_link_limit[1]

        # Disable closed linkage collision (gripper's inner fingers)
        bc.setCollisionFilterPair(self.robot_id, self.robot_id, 9, 11, 0)
        bc.setCollisionFilterPair(self.robot_id, self.robot_id, 13, 15, 0)
        self.gripper_constraint()

        self.reset_arm_poses()
        self.reset_gripper()
