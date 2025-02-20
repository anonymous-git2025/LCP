import math
import numpy as np
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff
from habitat.tasks.utils import cartesian_to_polar
import sys
from scipy.spatial.transform import Rotation as R
import quaternion

def heading_from_quaternion(quat: np.array):
    quat = quaternion_from_coeff(quat)      
    heading_vector = quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))     
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]     
    return phi % (2 * np.pi)

def heading_from_quaternion_wo_coeff(quat: quaternion.quaternion):
    heading_vector = quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))     
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]      
    return phi % (2 * np.pi)


def calculate_vp_rel_pos(p1, p2, base_heading=0, base_elevation=0):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    

    heading = np.arcsin(-dx / xz_dist)  
    if p2[2] > p1[2]:
        heading = np.pi - heading
    heading -= base_heading
    
    while heading < 0:
        heading += 2*np.pi
    heading = heading % (2*np.pi)

    return heading, xz_dist     


def quat_from_heading(heading, elevation=0):
    array_h = np.array([0, heading, 0])
    array_e = np.array([0, elevation, 0])
    rotvec_h = R.from_rotvec(array_h)
    rotvec_e = R.from_rotvec(array_e)
    quat = (rotvec_h * rotvec_e).as_quat()
    return quat

def edge_vec_to_ang_dis(edge_vec):
    
    angle = -np.arctan2(1.0, 0.0) + np.arctan2(edge_vec[1], edge_vec[0])        
    if angle < 0.0:
        angle += 2 * math.pi

    distance = np.linalg.norm(edge_vec)     

    return angle, distance


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '_' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def navigable(habitat_pos, heading, dis, sim, step_size):
    theta = -(heading - np.pi)/2
    rotation = np.quaternion(np.cos(theta),0,np.sin(theta),0)      
    sim.set_agent_state(habitat_pos,rotation)
    for i in range(int(dis//step_size)):
        sim.step_without_obs(1)     
        if sim.previous_step_collided:
            return False
    return True


def navigable_with_check(habitat_pos, node_b_pos, heading, dis, sim, step_size):
    theta = -(heading - np.pi)/2
    rotation = np.quaternion(np.cos(theta),0,np.sin(theta),0)       
    sim.set_agent_state(habitat_pos,rotation)
    for i in range(int(dis//step_size)):
        sim.step_without_obs(1)
        if sim.previous_step_collided:
             final_pos = sim.get_agent_state().position[[0,2]]
             return False, None
    final_pos = sim.get_agent_state().position[[0,2]]
    if np.linalg.norm((final_pos - node_b_pos)) < 0.015:
        return True, True
    return True, False




from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)


import numpy as np
from gym import spaces
from gym.spaces.box import Box
from numpy import ndarray

if TYPE_CHECKING:
    from torch import Tensor

from habitat_sim.simulator import MutableMapping, MutableMapping_T
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.core.registry import registry
from habitat.core.simulator import (
    Config,
    VisualObservation,
)
from habitat.core.spaces import Space

@registry.register_simulator(name="Sim-v1")
class Simulator(HabitatSim):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def step_without_obs(self,
        action: Union[str, int, MutableMapping_T[int, Union[str, int]]],
        dt: float = 1.0 / 60.0,):
        self._num_total_frames += 1
        if isinstance(action, MutableMapping):
            return_single = False
        else:
            action = cast(Dict[int, Union[str, int]], {self._default_agent_id: action})     
            return_single = True
        collided_dict: Dict[int, bool] = {}
        for agent_id, agent_act in action.items():
            agent = self.get_agent(agent_id)
            collided_dict[agent_id] = agent.act(agent_act)
            self.__last_state[agent_id] = agent.get_state()


        multi_observations = {}
        for agent_id in action.keys():
            agent_observation = {}
            agent_observation["collided"] = collided_dict[agent_id]
            multi_observations[agent_id] = agent_observation


        if return_single:
            sim_obs = multi_observations[self._default_agent_id]
        else:
            sim_obs = multi_observations

        self._prev_sim_obs = sim_obs
