from typing import Any, List, Optional, Tuple

import math
import numpy as np

from habitat.core.embodied_task import (
    SimulatorTaskAction,
)
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar

from vlnce_baselines.GT_waypoint.utils import calculate_vp_rel_pos, heading_from_quaternion_wo_coeff, quat_from_heading



@registry.register_task_action
class MoveHighToLowAction(SimulatorTaskAction):
    def step(self, *args: Any, 
            angle: float, distance: float,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        init_state = self._sim.get_agent_state()

        
        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = self._sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        
        self._sim.set_agent_state(init_state.position, rotation)

        
        ksteps = int(distance//init_forward)
        for k in range(ksteps):
            if k == ksteps - 1:
                output = self._sim.step(forward_action)
            else:
                self._sim.step_without_obs(forward_action)

        return output
    
@registry.register_task_action
class MoveHighToLowAction_GT_waypoint(MoveHighToLowAction):
    def step(self, *args: Any, 
            angle: float, distance: float,
            pos,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        init_state = self._sim.get_agent_state()

       
        forward_action = HabitatSimActions.MOVE_FORWARD
       
        init_forward = self._sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount
        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        
        self._sim.set_agent_state(init_state.position, rotation)

        
        ksteps = int(distance//init_forward)
        for k in range(ksteps):
            if k == ksteps - 1:
                output = self._sim.step(forward_action)
            else:
                self._sim.step_without_obs(forward_action)

        self._sim.set_agent_state(pos, rotation)
        output = self._sim.get_observations_at(pos, rotation, False)
    

        return output


@registry.register_task_action
class MoveHighToLowActionEval(SimulatorTaskAction):
    def step(self, *args: Any, 
            angle: float, distance: float,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        init_state = self._sim.get_agent_state()

        positions = []
        collisions = []
        
        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = self._sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)

        self._sim.set_agent_state(init_state.position, rotation)

        ksteps = int(distance//init_forward)
        for k in range(ksteps):
            if k == ksteps - 1:
                output = self._sim.step(forward_action)
            else:
                self._sim.step_without_obs(forward_action)
            positions.append(self._sim.get_agent_state().position)
            collisions.append(self._sim.previous_step_collided)

        output['positions'] = positions
        output['collisions'] = collisions

        return output
    
@registry.register_task_action
class MoveHighToLowActionEval_GT_waypoint(MoveHighToLowActionEval):
    def step(self, *args: Any, 
            angle: float, distance: float,
            pos,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        init_state = self._sim.get_agent_state()

        positions = []
        collisions = []
        
        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = self._sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + (angle) / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)

        self._sim.set_agent_state(init_state.position, rotation)


        ksteps = int(distance//init_forward)
        for k in range(ksteps):
            if k == ksteps - 1:
                output = self._sim.step(forward_action)
            else:
                self._sim.step_without_obs(forward_action)
            positions.append(self._sim.get_agent_state().position)
            collisions.append(self._sim.previous_step_collided)

        self._sim.set_agent_state(pos, rotation)
        positions.append(self._sim.get_agent_state().position)
        collisions.append(self._sim.previous_step_collided)
        output = self._sim.get_observations_at(pos, rotation, False)

        output['positions'] = positions
        output['collisions'] = collisions

        return output


@registry.register_task_action
class MoveHighToLowActionInfer(SimulatorTaskAction):
    def step(self, *args: Any, 
            angle: float, distance: float,
            **kwargs: Any):
        r"""This control method is called from ``Env`` on each ``step``.
        """
        init_state = self._sim.get_agent_state()

        def get_info(sim):
            agent_state = sim.get_agent_state()
            heading_vector = quaternion_rotate_vector(
                agent_state.rotation.inverse(), np.array([0, 0, -1])
            )
            heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
            return {
                "position": agent_state.position.tolist(),
                "heading": heading,
                "stop": False,
            }
        infos = []
       
        forward_action = HabitatSimActions.MOVE_FORWARD
        
        init_forward = self._sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        
        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)

        self._sim.set_agent_state(init_state.position, rotation)
       

        ksteps = int(distance//init_forward)
        for k in range(ksteps):
            if k == ksteps - 1:
                output = self._sim.step(forward_action)
            else:
                self._sim.step_without_obs(forward_action)
            infos.append(get_info(self._sim))

        output['infos'] = infos

        return output 