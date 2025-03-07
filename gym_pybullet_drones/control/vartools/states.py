"""
Basic state to base anything on.
"""
# Author: Lukas Huber
# Mail: lukas.huber@epfl.ch
# License: BSD (c) 2021
# import time
# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt

from scipy.spatial.transform import Rotation  # scipy rotation


class BaseState:
    def __init__(self, position, orientation, velocity, angular_velocity):
        pass


class Time:
    pass


class Stamp:
    def __init__(self, seq: int = None, timestamp: Time = None, frame_id: str = None):
        self.seq = seq
        self.timestamp = timestamp
        self.frame_id = frame_id


class ObjectTwist:
    def __repr__(self):
        return f"Linear {self.linear} \n" + f"Angular: {self.angular}"

    def __init__(
        self,
        linear: np.ndarray = None,
        angular: np.ndarray = None,
        dimension: float = None,
    ):

        if dimension is None:
            if linear is None:
                self.dimension = 2
            else:
                self.dimension = dimension

        self.linear = linear
        self.angular = angular

    @property
    def linear(self):
        return self._linear

    @linear.setter
    def linear(self, value):
        if value is None:
            self._linear = np.zeros(self.dimension)
        else:
            self._linear = np.array(value)


class ObjectPose:
    """(ROS)-inspired pose of an object of dimension
    Attributes
    ----------
    Position

    """

    def __repr__(self):
        return (
            super().__repr__()
            + " with \n"
            + f"position {repr(self.position)} \n"
            + f"orientation: {self.orientation}"
        )

    def __init__(
        self,
        position: npt.ArrayLike,
        orientation: Optional[np.ndarray] = None,
        stamp: Optional[Stamp] = None,
        dimension: Optional[int] = None,
    ):
        # Assign values
        self.position = np.array(position)
        self.stamp = stamp

        if orientation is None:
            if self.dimension == 2:
                self.orientation = 0

            elif self.dimension == 3:
                self.orientation = Rotation.from_euler("x", [0])

            else:
                # Keep none
                self.orientation = orientation
        else:
            self.orientation = orientation

    @property
    def dimension(self) -> int:
        if self.position is None:
            return None
        return self.position.shape[0]

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: npt.ArrayLike):
        if value is None:
            self._position = value
            return
        self._position = np.array(value)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        """Value is of type 'float' for 2D
        or `numpy.array`/`scipy.spatial.transform.Rotation` for 3D and higher."""
        if value is None:
            self._orientation = value
            return

        if self.dimension == 2:
            self._orientation = value
            # self.rotation_matrix = get_rotation_matrix(self.orientation)

        elif self.dimension == 3:
            if not isinstance(value, Rotation):
                raise TypeError("Use 'scipy - Rotation' type for 3D orientation.")
            self._orientation = value

        else:
            if value is not None and np.sum(np.abs(value)):  # nonzero value
                warnings.warn("Rotation for dimensions > 3 not defined.")
            self._orientation = value

    @property
    def rotation_matrix(self):
        if self.dimension != 2:
            warnings.warn("Orientation matrix only used for useful for 2-D rotations.")
            return

        if self.orientation is None:
            return np.eye(self.dimension)

        _cos = np.cos(self.orientation)
        _sin = np.sin(self.orientation)
        return np.array([[_cos, (-1) * _sin], [_sin, _cos]])

    def update(self, delta_time: float, twist: ObjectTwist):
        if twist.linear is not None:
            self.position = position + twist.linear * delta_time

        if twist.angular is not None:
            breakpoint()
            # Not implemented
            self.angular = position + twist.agnular * delta_time

    def transform_position_from_reference_to_local(self, *args, **kwargs):
        # TODO: is being renamed -> remove original]
        return self.transform_position_from_relative(*args, **kwargs)

    def transform_pose_to_relative(self, pose: ObjectPose) -> ObjectPose:
        pose.position = self.transform_position_to_relative(pose.position)

        if self.orientation is None:
            return pose

        if pose.orientation is None:
            pose.orientation = pose.orientation - self.orientation
            return pose

        if self.dimension != 2:
            raise NotImplementedError()

        pose.orientation += self.orientation

    def transform_pose_from_relative(self, pose: ObjectPose) -> ObjectPose:
        pose.position = self.transform_position_from_relative(pose.position)

        if self.orientation is None:
            return pose

        if pose.orientation is None:
            pose.orientation = pose.orientation + self.orientation
            return pose

        if self.dimension != 2:
            raise NotImplementedError()

        pose.orientation -= self.orientation

        return pose

    def transform_position_from_relative(self, position: np.ndarray) -> np.ndarray:
        """Transform a position from the global frame of reference
        to the obstacle frame of reference"""
        position = self.transform_direction_from_relative(direction=position)

        if self.position is not None:
            position = position + self.position

        return position

    def transform_positions_from_relative(self, positions: np.ndarray) -> np.ndarray:
        positions = self.transform_directions_from_relative(direction=positions)
        if not self.position is None:
            positions = positions + np.tile(self.position, (positions.shape[1], 1)).T

        return positions

    def transform_position_from_local_to_reference(
        self, position: np.ndarray
    ) -> np.ndarray:
        return self.transform_position_to_relative(position)

    def transform_position_to_relative(self, position: np.ndarray) -> np.ndarray:
        """Transform a position from the obstacle frame of reference
        to the global frame of reference"""
        if self.position is not None:
            position = position - self.position

        position = self.transform_direction_to_relative(direction=position)
        return position

    def transform_positions_to_relative(self, positions: np.ndarray) -> np.ndarray:
        if not self.position is None:
            positions = positions - np.tile(self.position, (positions.shape[1], 1)).T

        positions = self.transform_directions_to_relative(directions=positions)
        return positions

    def transform_direction_from_reference_to_local(
        self, direction: np.ndarray
    ) -> np.ndarray:
        """Transform a direction, velocity or relative position to the global-frame."""
        raise
        # return self.apply_rotation_reference_to_local(direction)

    def transform_direction_from_local_to_reference(
        self, direction: np.ndarray
    ) -> np.ndarray:
        """Transform a direction, velocity or relative position to the obstacle-frame"""
        raise
        # return self.apply_rotation_local_to_reference(direction)

    def transform_direction_from_relative(self, direction: np.ndarray) -> np.ndarray:
        if self._orientation is None:
            return direction

        if self.dimension == 2:
            return self.rotation_matrix.dot(direction)

        elif self.dimension == 3:
            return self._orientation.apply(direction.T).flatten()
        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction

    def transform_directions_from_relative(self, directions: np.ndarray) -> np.ndarray:
        return self.transform_direction_from_relative(directions)

    def transform_direction_to_relative(self, direction: np.ndarray) -> np.ndarray:
        if self._orientation is None:
            return direction

        if self.dimension == 2:
            return self.rotation_matrix.T.dot(direction)

        elif self.dimension == 3:
            return self._orientation.inv().apply(direction.T).flatten()
        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction

    def transform_directions_to_relative(self, directions: np.ndarray) -> np.ndarray:
        return self.transform_direction_to_relative(directions)

    def apply_rotation_reference_to_local(self, direction: np.ndarray) -> np.ndarray:
        if self._orientation is None:
            return direction

        if self.dimension == 2:
            return self.rotation_matrix.T.dot(direction)

        elif self.dimension == 3:
            return self._orientation.inv().apply(direction.T).T
        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction

    def apply_rotation_local_to_reference(self, direction: np.ndarray) -> np.ndarray:
        if self._orientation is None:
            return direction

        if self.dimension == 2:
            return self.rotation_matrix.dot(direction)

        elif self.dimension == 3:
            return self._orientation.apply(direction.T).flatten()

        else:
            warnings.warn("Not implemented for higer dimensions")
            return direction


class Wrench:
    def __init__(self, linear, angular):
        pass
