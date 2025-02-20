from habitat_baselines.common.baseline_registry import baseline_registry
import math
import random
import habitat
import numpy as np
import cv2
import os
from habitat import Config, Dataset
from typing import Any, Dict, Optional, Tuple, List, Union

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from scipy.spatial.transform import Rotation as R

from vlnce_baselines.common.environments import VLNCEDaggerEnv


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


@baseline_registry.register_env(name="DataCollectEnv")
class  DataCollectEnv(VLNCEDaggerEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)


    def get_pos_ori(self):
        agent_state = self._env.sim.get_agent_state()
        pos = agent_state.position
        ori = np.array([*(agent_state.rotation.imag), agent_state.rotation.real])
        return (pos, ori)
    
    def get_observation_at(self,
        source_position: List[float],
        source_rotation: List[Union[int, np.float64]],
        keep_agent_at_new_pose: bool = False):
        
        obs = self._env.sim.get_observations_at(source_position, source_rotation, keep_agent_at_new_pose)
        obs.update(self._env.task.sensor_suite.get_observations(
            observations=obs, episode=self._env.current_episode, task=self._env.task
        ))
        return obs
    
    
    
    def my_step(self, action, vis_info, *args, **kwargs):
        act = action['act']

        if act == 4: 
            if self.video_option:
                self.get_plan_frame(vis_info)       

            
            if action['back_path'] is None:     
                self.teleport(action['front_pos'])      
            else:
                self.multi_step_with_heading(action['back_path'], vis_info)    
            agent_state = self._env.sim.get_agent_state()       
            observations = self.get_observation_at(agent_state.position, agent_state.rotation)      

            
            self.single_step_with_heading(action['ghost_pos'], vis_info)       
            agent_state = self._env.sim.get_agent_state()
            observations = self.get_observation_at(agent_state.position, agent_state.rotation)

        elif act == 0:   
            if self.video_option:
                self.get_plan_frame(vis_info)

            
            if action['back_path'] is None:
                self.teleport(action['stop_pos'])
            else:
                self.multi_step_with_heading(action['back_path'], vis_info)

            
            observations = self._env.step(act)
            if self.video_option:       
                info = self.get_info(observations)
                self.video_frames.append(
                    navigator_video_frame(
                        observations,
                        info,
                        vis_info,
                    )
                )
                self.get_plan_frame(vis_info)       

        else:
            raise NotImplementedError                

        reward = self.get_reward(observations)      
        done = self.get_done(observations)         
        info = self.get_info(observations)          

        if self.video_option and done:      
            
            generate_video(
                video_option=self.video_option,
                video_dir=self.video_dir,
                images=self.video_frames,
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2],
                checkpoint_idx=0,
                metrics={"SPL": round(info["spl"], 3)},
                tb_writer=None,
                fps=8,
            )
            
            metrics={ 
                        "spl": round(info["spl"], 3),
                    }
            metric_strs = []
            for k, v in metrics.items():
                metric_strs.append(f"{k}{v:.2f}")
            episode_id=self._env.current_episode.episode_id
            scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2]
            tmp_name = f"{scene_id}-{episode_id}-" + "-".join(metric_strs)
            tmp_name = tmp_name.replace(" ", "_").replace("\n", "_") + ".png"
            tmp_fn = os.path.join(self.video_dir, tmp_name)
            tmp = np.concatenate(self.plan_frames, axis=0)
            cv2.imwrite(tmp_fn, tmp)
            self.plan_frames = []

        return observations, reward, done, info
    
    def multi_step_with_heading(self, path, vis_info):
        for vp, vp_pos in path[:]:
            self.single_step_with_heading(vp_pos, vis_info)

    def single_step_with_heading(self, pos, vis_info):
        act_f = HabitatSimActions.MOVE_FORWARD
        uni_f = self._env.sim.get_agent(0).agent_config.action_space[act_f].actuation.amount
        agent_state = self._env.sim.get_agent_state()
        ang, dis = calculate_vp_rel_pos(agent_state.position, pos, heading_from_quaternion(agent_state.rotation))
        final_heading = ang+heading_from_quaternion(agent_state.rotation)
        self.teleport_with_heading(agent_state.position, final_heading)
        ksteps = int(dis // uni_f)
        for _ in range(ksteps):     
            self.wrap_act(act_f, vis_info)
        self.teleport_with_heading(pos, final_heading)

    def teleport_with_heading(self, pos, ang):
        self._env.sim.set_agent_state(pos, quat_from_heading(ang))

    def teleport_with_quat(self, pos, quat):
        self._env.sim.set_agent_state(pos, quat)
