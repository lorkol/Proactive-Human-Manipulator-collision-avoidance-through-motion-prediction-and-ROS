
from typing import Annotated, Dict, List, TypeAlias, Tuple
import cupy as cp
from cupyx.scipy.spatial import KDTree
from cupy.typing import NDArray

#---------------------------------------------------------- Type Aliasing
#Holonomic
RobotAnglesVector: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (6,) representing the 6 joint angles of the robot'''
RobotJointPositions: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (6, 3) representing the x,y,z positions of each of the 6 robot joints'''
HumanPoseSequence: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (N, 15, 3) where N is the number of time steps - Currently 1, each pose has 15 joints with (x,y,z) coordinates'''
HumanPose: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (15, 3) human pose in a single time step each pose has 15 joints with (x,y,z) coordinates'''
Position: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (3,) representing a 3D position vector'''
Link: TypeAlias = Tuple[Position, Position, float]
'''A body link represented by two end positions and a radius'''

# Kinodynamic
RobotVelocitiesVector: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (6,) representing the 6 joint angle velocities of the robot'''
RobotKinodynamicState: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (12,) representing the 6 joint angles and 6 joint velocities of the robot'''

class RobotState:
    """Represents the kinodynamic state of the robot: joint angles and joint velocities."""
    #Static fields (class variables)
    DOF: int = 6
    '''Degrees of freedom'''
    MAX_POSITIONS: RobotAnglesVector = cp.array([6.28, 6.28, 3.14, 6.28, 6.28, 6.28])
    '''Max angular positions in rad according to the official UR16e documentation'''
    MAX_VELOCITIES: RobotVelocitiesVector = cp.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14])
    '''Max angular velocities in rad/s according to the official UR16e documentation'''
    MAX_ACCELERATIONS: RobotAnglesVector = cp.array([6.28, 6.28, 6.28, 6.28, 6.28, 6.28])
    '''Example: max angular accelerations in rad/s^2 according to the official UR16e documentation'''

    def __init__(self, angles: RobotAnglesVector, velocities: RobotVelocitiesVector) -> None:
        self.angles: RobotAnglesVector = angles.copy()
        self.velocities: RobotVelocitiesVector = velocities.copy()

        
    ############Methods to convert to vectors
    def angles_to_vector(self) -> RobotAnglesVector:
        """returns the joint angles as a cupy array of shape (6,)"""
        return self.angles
    
    def velocities_to_vector(self) -> RobotVelocitiesVector:
        """returns the joint velocities as a cupy array of shape (6,)"""
        return self.velocities

    def to_vector(self) -> RobotKinodynamicState:
        """returns the joint angles and velocities concatenated as a single cupy array of shape (12,)"""
        return cp.concatenate((self.angles_to_vector(), self.velocities_to_vector()))


ControlInput: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray] 
'''A control input is represented as a vector of joint angle accelerations'''